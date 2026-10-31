"""Tests for saving the CPT checkpoint and resuming from it, without loading a model."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from satyrn.trainer.unsloth.config import ExperimentConfig, validate_config
from satyrn.trainer.unsloth.run import (
    CONFIG_DIR,
    check_adapter_matches_config,
    check_cpt_checkpoint_is_free,
    save_cpt_checkpoint,
)

BASE_EXPERIMENT = "py3.15-qwen3.8-27b"
DISTILLED_EXPERIMENT = "py3.15-qwen3.8-27b-distilled"
CPT_ONLY = ["datasets.sft=null", "datasets.rl=null"]


def load_config(experiment_name: str, overrides: list[str]) -> ExperimentConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR):
        return validate_config(compose(config_name=f"experiment/{experiment_name}", overrides=overrides))


def test_save_cpt_checkpoint_accepted_on_a_cpt_only_run() -> None:
    config = load_config(BASE_EXPERIMENT, ["save_cpt_checkpoint=true", *CPT_ONLY])

    assert config.save_cpt_checkpoint
    assert config.checkpoint_dir == "outputs/py3.15-Qwen3.8-27B"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["datasets.cpt=null", *CPT_ONLY], r"requires datasets.cpt"),
        (["datasets.rl=null"], r"datasets.sft and datasets.rl"),
        (["datasets.sft=null"], r"datasets.sft and datasets.rl"),
    ],
)
def test_save_cpt_checkpoint_rejected_with_other_datasets(overrides: list[str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_config(BASE_EXPERIMENT, ["save_cpt_checkpoint=true", *overrides])


def test_resume_adapter_accepted_without_cpt_and_need_not_exist() -> None:
    config = load_config(BASE_EXPERIMENT, ["resume_adapter=does/not/exist", "datasets.cpt=null"])

    assert config.resume_adapter == "does/not/exist"


def test_resume_adapter_rejected_with_cpt() -> None:
    with pytest.raises(ValidationError, match=r"resume_adapter requires datasets.cpt to be null"):
        load_config(BASE_EXPERIMENT, ["resume_adapter=does/not/exist"])


@pytest.mark.parametrize(
    ("experiment_name", "seed_run_name"),
    [
        ("py3.15-qwen3.8-27b", "py3.15-Qwen3.8-27B"),
        ("py3.15-ornith-1.5-9b", "py3.15-Ornith-1.5-9B"),
        ("py-mellum2-12b-a2.5", "py-mellum2-12b-a2.5"),
    ],
)
def test_distilled_experiment_resumes_from_its_base_experiment(experiment_name: str, seed_run_name: str) -> None:
    base = load_config(experiment_name, [])
    distilled = load_config(f"{experiment_name}-distilled", [])

    assert base.run_name == seed_run_name
    assert distilled.run_name != base.run_name
    assert distilled.resume_adapter == f"outputs/{seed_run_name}/cpt/adapter"
    assert not distilled.pre_model_eval
    assert distilled.datasets.cpt is None
    # max_seq_length comes from the inherited cpt block, so the model is loaded the same way
    assert distilled.max_seq_length == base.max_seq_length
    assert distilled.sft == base.sft
    assert len(distilled.datasets.sft) == len(base.datasets.sft)
    for distilled_path, base_path in zip(distilled.datasets.sft, base.datasets.sft, strict=True):
        assert Path(distilled_path).name == Path(base_path).name
        assert "/distilled/" in distilled_path


class FakeModel:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def save_pretrained(self, path: Path) -> None:
        self.calls.append(("model", path))

    def save_pretrained_merged(self, path: Path, processor: object, save_method: str) -> None:
        self.calls.append(("merged", path, save_method))


class FakeProcessor:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def save_pretrained(self, path: Path) -> None:
        self.calls.append(("processor", path))


def test_save_cpt_checkpoint_writes_adapter_and_merged_model(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = SimpleNamespace(checkpoint_dir=str(tmp_path))

    save_cpt_checkpoint(FakeModel(calls), FakeProcessor(calls), config)

    assert calls == [
        ("model", tmp_path / "cpt" / "adapter"),
        ("processor", tmp_path / "cpt" / "adapter"),
        ("merged", tmp_path / "cpt" / "merged", "merged_16bit"),
    ]


def test_existing_cpt_checkpoint_fails_at_startup(tmp_path: Path) -> None:
    (tmp_path / "cpt").mkdir()
    config = SimpleNamespace(checkpoint_dir=str(tmp_path), save_cpt_checkpoint=True)

    with pytest.raises(FileExistsError, match="already exists"):
        check_cpt_checkpoint_is_free(config)


def test_existing_cpt_checkpoint_ignored_when_not_saving(tmp_path: Path) -> None:
    (tmp_path / "cpt").mkdir()

    check_cpt_checkpoint_is_free(SimpleNamespace(checkpoint_dir=str(tmp_path), save_cpt_checkpoint=False))


def adapter_model(r: int, lora_alpha: int) -> SimpleNamespace:
    return SimpleNamespace(peft_config={"default": SimpleNamespace(r=r, lora_alpha=lora_alpha)})


def test_adapter_matching_config_passes() -> None:
    config = load_config(DISTILLED_EXPERIMENT, [])
    peft = config.model.peft

    check_adapter_matches_config(adapter_model(peft.r, peft.lora_alpha), config)


@pytest.mark.parametrize(("r_delta", "alpha_delta"), [(1, 0), (0, 1)])
def test_adapter_differing_from_config_raises(r_delta: int, alpha_delta: int) -> None:
    config = load_config(DISTILLED_EXPERIMENT, [])
    peft = config.model.peft

    with pytest.raises(ValueError, match="the config has"):
        check_adapter_matches_config(adapter_model(peft.r + r_delta, peft.lora_alpha + alpha_delta), config)
