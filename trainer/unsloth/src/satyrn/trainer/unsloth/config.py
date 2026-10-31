"""Pydantic schema for the composed Hydra experiment config."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

StageName = Literal["pre", "cpt", "sft", "rl"]


def basename(path: str) -> str:
    """Return the final path segment."""
    return Path(path).name


OmegaConf.register_resolver("mul", operator.mul)
OmegaConf.register_resolver("max", lambda *values: max(values))
OmegaConf.register_resolver("basename", basename)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpt: CptStageConfig
    sft: SftStageConfig
    rl: RlStageConfig
    datasets: DatasetsConfig
    model: ModelConfig
    mlflow: MlflowConfig

    eval_ratio: float
    load_in_4bit: bool
    logging_steps: int
    max_seq_length: int
    max_steps: int
    optim: str
    pre_model_eval: bool
    run_name: str

    push_to_hub: bool
    push_to_hub_path: str | None

    save_cpt_checkpoint: bool
    checkpoint_dir: str
    resume_adapter: str | None

    @model_validator(mode="after")
    def check_checkpoint_flags(self) -> ExperimentConfig:
        if self.save_cpt_checkpoint:
            if self.datasets.cpt is None:
                raise ValueError("save_cpt_checkpoint requires datasets.cpt")
            # merged_16bit may not leave the live PEFT model intact, so no stage may train after it.
            if self.datasets.sft is not None or self.datasets.rl is not None:
                raise ValueError("save_cpt_checkpoint requires datasets.sft and datasets.rl to be null")
        # The path is not checked here: checkpoints don't exist in CI, and from_pretrained reports a missing one.
        if self.resume_adapter is not None and self.datasets.cpt is not None:
            raise ValueError("resume_adapter requires datasets.cpt to be null")
        return self


class PeftConfig(BaseModel):
    """Arguments passed to FastModel.get_peft_model."""

    model_config = ConfigDict(extra="forbid")

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str] | None
    finetune_attention_modules: bool
    finetune_mlp_modules: bool
    finetune_language_layers: bool
    finetune_vision_layers: bool
    finetune_audio_layers: bool


class TemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction_part: str
    response_part: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    enable_thinking: bool
    peft: PeftConfig
    template: TemplateConfig


class MlflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str
    experiment_name: str


class DatasetsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpt: str | list[str] | None
    sft: str | list[str] | None
    rl: str | list[str] | None


class CptStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    packing: bool
    prepack_dataset: bool
    seq_len: int
    batch_size: int
    eval_batch_size: int
    eval_steps: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float


class SftStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq_len: int
    batch_size: int
    eval_batch_size: int
    eval_steps: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float


class RlStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_completion_length: int
    num_train_epochs: int
    learning_rate: float
    beta: float


def validate_config(cfg: DictConfig) -> ExperimentConfig:
    return ExperimentConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))


def log_config(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
