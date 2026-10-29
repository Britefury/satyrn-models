"""Pydantic schema for the composed Hydra experiment config."""

from __future__ import annotations

import logging
import operator
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

StageName = Literal["pre", "cpt", "sft", "sdft", "rl"]


def _format_errors(stage: str, errors: list[str]) -> str:
    """Render every config problem at once so one startup run reports the full list."""
    listed = "\n".join(f"  - {error}" for error in errors)
    return f"Invalid {stage} config ({len(errors)} problem{'s' if len(errors) > 1 else ''}):\n{listed}"


def _check_group_multiple(batch_size: int, num_generations: int, name: str, errors: list[str]) -> None:
    """A batch holding rollouts must contain whole groups of num_generations completions."""
    if batch_size % num_generations != 0:
        errors.append(
            f"{name} ({batch_size}) must be a multiple of num_generations ({num_generations}): every generated batch "
            "holds whole groups of completions, which the trainer compares within a group."
        )


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
    # Jinja2 template rendering the teacher's privileged context, including the instruction that
    # the demonstration should be re-written in the model's own words. Available variables:
    # trace, code, answer, expected_output. `trace` is None unless thinking is enabled. Compiled
    # once per dataset.
    privileged_context_template: str = (
        "The reasoning trace and answer below are a demonstration for this task. "
        "They should be re-written in your own words rather than copied verbatim.\n\n"
        "{% if trace %}Reasoning trace:\n{{ trace }}\n\n{% endif %}"
        "Code:\n{{ code }}\n\n"
        "Answer:\n{{ answer }}\n\n"
        "Expected output:\n{{ expected_output }}"
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
    vllm_mode: Literal["server", "colocate"] = "colocate"
    # Model implementation to use for vLLM. One of 'transformers' or 'vllm'.
    vllm_model_impl: Literal["transformers", "vllm"] = "vllm"
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
    # Run the colocated vLLM engine without torch.compile and CUDA graphs. vLLM 0.23 fails to compile Qwen3.5's
    # linear attention ("Tried to erase Node size but it still had 2 users"). For vllm_mode='server', pass
    # --enforce-eager to the vLLM server instead.
    vllm_enforce_eager: bool = False

    # Training parameters
    # Number of iterations per batch.
    num_iterations: int = 1
    # Batch size to use for generation. None defaults to the effective training batch size.
    generation_batch_size: int | None = None
    # Number of steps per generation. None defaults to gradient_accumulation_steps.
    steps_per_generation: int | None = None

    @model_validator(mode="after")
    def _check_trainer_constraints(self) -> SdftStageConfig:
        """
        Reject combinations SDFTConfig/SDFTTrainer would only reject once the model is loaded.

        Every rule here mirrors a check or a documented requirement of the trainer (or of the batch layout its
        rollout grouping relies on), so a bad config fails at startup instead of after model load. Defaults and
        derivations assume a single training process, which is how unsloth runs (one GPU, plain `SDFTTrainer`).
        """
        errors: list[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        # Values the trainer divides by, indexes with, or uses as a loop bound. Checked first: later checks divide.
        for name in (
            "batch_size",
            "eval_batch_size",
            "eval_steps",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "num_generations",
            "num_iterations",
            "teacher_sync_steps",
        ):
            value = getattr(self, name)
            require(value >= 1, f"{name} must be at least 1, got {value}.")
        for name in ("max_prompt_length", "max_completion_length", "generation_batch_size", "steps_per_generation"):
            value = getattr(self, name)
            require(value is None or value >= 1, f"{name} must be at least 1 when set, got {value}.")
        require(self.num_loss_tokens_to_skip >= 0, "num_loss_tokens_to_skip must not be negative.")
        # The distillation loss is per token with no group advantage, so a single completion per prompt is valid.
        require(
            self.num_generations_eval is None or self.num_generations_eval >= 1,
            "num_generations_eval must be at least 1 when set, or None to reuse num_generations.",
        )

        # Generation and evaluation params, sampled by the transformers/vLLM samplers.
        require(self.temperature > 0, f"temperature must be positive, got {self.temperature}.")
        require(0 < self.top_p <= 1, f"top_p must be in (0, 1], got {self.top_p}.")
        require(self.top_k >= 0, f"top_k must not be negative (0 disables it), got {self.top_k}.")
        require(self.min_p is None or 0 <= self.min_p <= 1, f"min_p must be in [0, 1] when set, got {self.min_p}.")
        require(
            self.repetition_penalty > 0,
            f"repetition_penalty must be positive, got {self.repetition_penalty}.",
        )

        # Distillation objective.
        require(
            0 <= self.distillation_alpha <= 1,
            f"distillation_alpha must be in [0, 1], got {self.distillation_alpha}.",
        )
        if self.distillation_mode == "sampled_token":
            require(
                self.distillation_alpha == 1.0,
                "distillation_mode='sampled_token' only supports reverse KL, so it requires distillation_alpha=1.0, "
                f"got {self.distillation_alpha}.",
            )
        if self.distillation_mode == "topk_logits":
            require(
                self.distillation_topk is not None and self.distillation_topk >= 1,
                "distillation_mode='topk_logits' requires distillation_topk to be at least 1, "
                f"got {self.distillation_topk}.",
            )
        require(
            self.distillation_is_clip is None or self.distillation_is_clip > 0,
            f"distillation_is_clip must be positive when set, got {self.distillation_is_clip}.",
        )

        # Teacher.
        if self.teacher_model_kind == "ema":
            require(
                0 < self.teacher_update_rate <= 1,
                f"teacher_update_rate must be in (0, 1] for teacher_model_kind='ema', got {self.teacher_update_rate}.",
            )
        require(
            "{prompt}" in self.teacher_prompt_template,
            "teacher_prompt_template must contain the `{prompt}` placeholder.",
        )
        require(
            "{privileged_context}" in self.teacher_prompt_template,
            "teacher_prompt_template must contain the `{privileged_context}` placeholder.",
        )

        # vLLM.
        require(
            0 < self.vllm_gpu_memory_utilization <= 1,
            f"vllm_gpu_memory_utilization must be in (0, 1], got {self.vllm_gpu_memory_utilization}.",
        )
        require(
            self.vllm_tensor_parallel_size >= 1,
            f"vllm_tensor_parallel_size must be at least 1, got {self.vllm_tensor_parallel_size}.",
        )
        for name in ("vllm_server_port", "vllm_group_port"):
            port = getattr(self, name)
            require(1 <= port <= 65535, f"{name} must be a valid port, got {port}.")
        require(self.vllm_server_timeout > 0, f"vllm_server_timeout must be positive, got {self.vllm_server_timeout}.")
        if self.vllm_max_model_length is not None and self.max_prompt_length is not None:
            # Prompts are capped at max_prompt_length and completions at max_completion_length, so the context window
            # has to fit both or vLLM aborts mid-run once a long sample comes along.
            needed_length = self.max_prompt_length + (self.max_completion_length or 0)
            require(
                self.vllm_max_model_length >= needed_length,
                f"vllm_max_model_length ({self.vllm_max_model_length}) must fit max_prompt_length + "
                f"max_completion_length ({needed_length}).",
            )
        if self.use_teacher_server:
            require(self.use_vllm, "use_teacher_server requires use_vllm=True.")
            require(
                self.vllm_mode == "server",
                "use_teacher_server requires vllm_mode='server': teacher logprobs come from the vLLM server.",
            )
            require(
                self.teacher_model_kind == "live",
                f"use_teacher_server only supports teacher_model_kind='live', got {self.teacher_model_kind!r}.",
            )
            require(
                self.distillation_mode in ("sampled_token", "topk_logits"),
                "use_teacher_server supports distillation_mode in {'sampled_token', 'topk_logits'}: the server only "
                f"returns top-k logprobs, got {self.distillation_mode!r}.",
            )

        if errors:
            raise ValueError(_format_errors("sdft", errors))

        # Generation schedule. SDFTConfig resolves generation_batch_size from steps_per_generation (defaulting to
        # gradient_accumulation_steps) or vice versa, and the resulting generation batch must hold whole training
        # batches and whole groups of num_generations completions.
        if self.generation_batch_size is not None and self.steps_per_generation is not None:
            errors.append(
                "generation_batch_size and steps_per_generation are mutually exclusive: the trainer derives one from "
                "the other. Set only one of them."
            )
        elif self.generation_batch_size is not None:
            if self.generation_batch_size % self.batch_size != 0:
                errors.append(
                    f"generation_batch_size ({self.generation_batch_size}) must be a multiple of batch_size "
                    f"({self.batch_size}) so each optimizer step consumes a whole training batch."
                )
            else:
                _check_group_multiple(self.generation_batch_size, self.num_generations, "generation_batch_size", errors)
        else:
            steps_per_generation = self.steps_per_generation or self.gradient_accumulation_steps
            generation_batch_size = self.batch_size * steps_per_generation
            _check_group_multiple(
                generation_batch_size,
                self.num_generations,
                "generation_batch_size (batch_size * steps_per_generation)",
                errors,
            )

        # Evaluation generates one group per prompt too, with num_generations_eval completions per prompt.
        _check_group_multiple(
            self.eval_batch_size,
            self.num_generations_eval or self.num_generations,
            "eval_batch_size",
            errors,
        )

        if errors:
            raise ValueError(_format_errors("sdft", errors))
        return self


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
