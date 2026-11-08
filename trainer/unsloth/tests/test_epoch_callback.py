"""The per-epoch callback evaluates mid-training and saves every epoch, adapter-only for LoRA models."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model

from satyrn.trainer.unsloth import epoch_callback
from satyrn.trainer.unsloth.epoch_callback import make_epoch_end_callback, save_model


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.config = SimpleNamespace(to_dict=lambda: {}, use_return_dict=True)

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True)
        torch.save(self.state_dict(), Path(path, "full_model.bin"))


class FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        Path(path, "tokenizer.json").write_text("{}")


def end_epoch(callback, epoch: float, global_step: int, max_steps: int) -> None:
    state = SimpleNamespace(epoch=epoch, global_step=global_step, max_steps=max_steps, is_world_process_zero=True)
    control = SimpleNamespace(should_training_stop=global_step >= max_steps)
    callback.on_epoch_end(SimpleNamespace(), state, control)


def run_epochs(callback, num_epochs: int, steps_per_epoch: int) -> None:
    for epoch in range(1, num_epochs + 1):
        end_epoch(callback, float(epoch), epoch * steps_per_epoch, num_epochs * steps_per_epoch)


def test_evaluates_every_epoch_but_the_last(tmp_path: Path) -> None:
    calls: list[tuple[int, str]] = []
    callback = make_epoch_end_callback("sdft", TinyModel(), FakeTokenizer(), lambda *a: calls.append(a), None)

    run_epochs(callback, num_epochs=3, steps_per_epoch=10)

    assert calls == [(10, "epoch1"), (20, "epoch2")]
    assert not tmp_path.joinpath("checkpoints").exists()


def test_saves_every_epoch_but_the_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[Path] = []
    monkeypatch.setattr(epoch_callback, "save_model", lambda model, tokenizer, path: saved.append(path))
    callback = make_epoch_end_callback("sdft", TinyModel(), FakeTokenizer(), None, tmp_path)

    run_epochs(callback, num_epochs=3, steps_per_epoch=5)

    assert saved == [tmp_path / "sdft-epoch1", tmp_path / "sdft-epoch2"]


def test_partial_epoch_cut_short_by_max_steps_is_left_to_the_trainer(tmp_path: Path) -> None:
    calls: list[tuple[int, str]] = []
    callback = make_epoch_end_callback("sdft", TinyModel(), FakeTokenizer(), lambda *a: calls.append(a), tmp_path)

    end_epoch(callback, epoch=0.4, global_step=4, max_steps=4)

    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_saves_full_model_mid_training(tmp_path: Path) -> None:
    callback = make_epoch_end_callback("sdft", TinyModel(), FakeTokenizer(), None, tmp_path)

    end_epoch(callback, epoch=1.0, global_step=5, max_steps=10)

    assert tmp_path.joinpath("sdft-epoch1", "full_model.bin").exists()


def test_lora_model_saves_only_the_adapter(tmp_path: Path) -> None:
    model = get_peft_model(TinyModel(), LoraConfig(r=2, target_modules=["proj"]))

    save_model(model, FakeTokenizer(), tmp_path / "ckpt")

    files = {path.name for path in (tmp_path / "ckpt").iterdir()}
    assert "adapter_config.json" in files
    assert files & {"adapter_model.safetensors", "adapter_model.bin"}
    assert "full_model.bin" not in files


def test_full_model_saves_everything(tmp_path: Path) -> None:
    save_model(TinyModel(), FakeTokenizer(), tmp_path / "ckpt")

    assert {path.name for path in (tmp_path / "ckpt").iterdir()} == {"full_model.bin", "tokenizer.json"}
