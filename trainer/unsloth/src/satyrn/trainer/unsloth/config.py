"""Pydantic schema for the composed Hydra experiment config."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

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
    sdft: SdftStageConfig
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
    sdft: str | list[str] | None
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


class SdftStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq_len: int
    batch_size: int
    eval_batch_size: int
    eval_steps: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float

    # SDFT loss parameters
    # Divergence interpolation coefficient for SDFT top-k logit distillation.
    distillation_alpha: float = 0.5
    # Distillation objective mode. SDFT defaults to top-k logit distillation.
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits"] = "topk_logits"
    # Number of top tokens used by the default SDFT top-k logit objective.
    distillation_topk: int = 100
    # Clipping coefficient for importance sampling in self-distillation. None disables clipping.
    distillation_is_clip: float = 2.0
    # Whether to add a tail bucket for non-top-k probability mass.
    distillation_add_tail: bool = False
    # Number of initial completion tokens to exclude from the distillation loss.
    num_loss_tokens_to_skip: int = 0

    # Teacher parameters
    # Semantic teacher choice for SDFT. base uses the initial student, live uses the current student, and ema uses an exponentially averaged teacher.
    teacher_model_kind: Literal["base", "live", "ema"] = "base"
    # EMA update rate used when teacher_model_kind="ema". A value of 1.0 reduces the update to a hard overwrite, periodically resyncing the teacher to the current student weights.
    teacher_update_rate: float = 0.05
    # Number of optimizer steps between teacher updates.
    teacher_sync_steps: int = 1

    # Teacher-conditioned generation
    # Whether on-policy generation should use the teacher-conditioned prompt instead of the student prompt.
    generate_from_teacher: bool = False
    # Template used to combine the student prompt and privileged context into the teacher prompt.
    teacher_prompt_template: str = "{prompt}\n\n{privileged_context}"
    




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
