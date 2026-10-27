"""Pydantic schema for the composed Hydra experiment config."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

StageName = Literal["pre", "cpt", "sft", "sdft", "rl"]


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

    # Privileged context
    # Instruction prepended to the teacher's privileged context, telling the teacher that the
    # provided demonstration (reasoning trace and answer) should be re-written in its own words.
    privileged_context_prompt: str = (
        "The reasoning trace and answer below are a demonstration for this task. "
        "They should be re-written in your own words rather than copied verbatim."
    )

    # Teacher-conditioned generation
    # Whether on-policy generation should use the teacher-conditioned prompt instead of the student prompt.
    generate_from_teacher: bool = False
    # Template used to combine the student prompt and privileged context into the teacher prompt.
    teacher_prompt_template: str = "{prompt}\n\n{privileged_context}"

    # Model parameters
    # Keyword arguments for transformers.AutoModelForCausalLM.from_pretrained, used when the model argument of the SDFTTrainer is provided as a string.
    model_init_kwargs: dict[str, Any] | None = None
    # Whether to allow loading models and tokenizers that ship custom Python code from the Hub.
    trust_remote_code: bool = False
    # Whether to disable dropout in the student and teacher models.
    disable_dropout: bool = True

    # Data preprocessing
    # Whether to only keep the columns required by the trainer in the dataset.
    remove_unused_columns: bool = False
    # Maximum prompt length. Longer prompts are truncated from the left.
    max_prompt_length: int | None = 512
    # Whether to shuffle the training dataset.
    shuffle_dataset: bool = True

    # Generation parameters
    # Number of generations to sample.
    num_generations: int = 8
    # Number of generations to sample during evaluation. None uses num_generations.
    num_generations_eval: int | None = None
    # Maximum length of the generated completion.
    max_completion_length: int | None = 256
    # Temperature for sampling.
    temperature: float = 1.0
    # Cumulative probability of the top tokens to consider.
    top_p: float = 1.0
    # Number of highest probability vocabulary tokens to keep for top-k-filtering. 0 disables it.
    top_k: int = 0
    # Minimum token probability, scaled by the probability of the most likely token.
    min_p: float | None = None
    # Float that penalizes new tokens based on whether they appear in the prompt and generated text so far.
    repetition_penalty: float = 1.0
    # Implementation of the cache method for faster generation when use_vllm is False.
    cache_implementation: str | None = None
    # Additional keyword arguments passed to GenerationConfig or SamplingParams when sampling completions.
    generation_kwargs: dict[str, Any] | None = None
    # Additional keyword arguments passed to apply_chat_template when generating completions.
    chat_template_kwargs: dict[str, Any] | None = None
    # Whether to gather the policy model weights for generation under DeepSpeed ZeRO-3.
    ds3_gather_for_generation: bool = True

    # vLLM generation acceleration
    # Whether to use vLLM for generating completions instead of the default model.generate().
    use_vllm: bool = False
    # Whether to compute teacher logprobs from the running vLLM generation server.
    use_teacher_server: bool = False
    # Mode to use for vLLM integration when use_vllm is True. One of 'server' or 'colocate'.
    vllm_mode: str = "colocate"
    # Model implementation to use for vLLM. One of 'transformers' or 'vllm'.
    vllm_model_impl: str = "vllm"
    # Enable vLLM sleep mode to offload weights/cache during the optimizer step.
    vllm_enable_sleep_mode: bool = False
    # Base URL for the vLLM server. If provided, vllm_server_host and vllm_server_port are ignored.
    vllm_server_base_url: str | None = None
    # Host of the vLLM server to connect to.
    vllm_server_host: str = "0.0.0.0"
    # Port of the vLLM server to connect to.
    vllm_server_port: int = 8000
    # Port number for the weight update group.
    vllm_group_port: int = 51216
    # Total timeout duration in seconds to wait for the vLLM server to be up.
    vllm_server_timeout: float = 240.0
    # Tensor parallel size for vLLM when vllm_mode is 'colocate'.
    vllm_tensor_parallel_size: int = 1
    # GPU memory utilization for vLLM when vllm_mode is 'colocate'.
    vllm_gpu_memory_utilization: float = 0.3
    # Context window for vLLM. If omitted, it is inferred from the model config.
    vllm_max_model_length: int | None = None

    # Training parameters
    # Number of iterations per batch.
    num_iterations: int = 1
    # Batch size to use for generation. None defaults to the effective training batch size.
    generation_batch_size: int | None = None
    # Number of steps per generation. None defaults to gradient_accumulation_steps.
    steps_per_generation: int | None = None


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
