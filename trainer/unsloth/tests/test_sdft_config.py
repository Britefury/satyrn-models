"""Startup validation of SDFT stage settings."""

from typing import Any

import pytest
from pydantic import ValidationError

from satyrn.trainer.unsloth.config import SdftStageConfig

BASE: dict[str, Any] = {
    "batch_size": 8,
    "eval_batch_size": 8,
    "eval_steps": 3,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
}


def build(**overrides: Any) -> SdftStageConfig:
    return SdftStageConfig.model_validate({**BASE, **overrides})


def error_text(**overrides: Any) -> str:
    with pytest.raises(ValidationError) as exit_info:
        build(**overrides)
    return str(exit_info.value)


def test_defaults_validate() -> None:
    """The defaults produce a generation batch that holds whole prompt groups."""
    config = build()

    assert config.num_generations == 8
    assert config.batch_size * config.gradient_accumulation_steps % config.num_generations == 0


def test_generation_batch_needs_whole_prompt_groups() -> None:
    """generation_batch_size must divide into groups of num_generations completions."""
    assert "must be a multiple of num_generations" in error_text(generation_batch_size=12, batch_size=4)


def test_generation_batch_needs_whole_training_batches() -> None:
    """generation_batch_size must cover whole optimizer-step batches."""
    assert "must be a multiple of batch_size" in error_text(generation_batch_size=8, batch_size=12)


def test_default_generation_batch_is_checked() -> None:
    """With neither field set the trainer derives the batch from batch_size and grad accumulation, so check that."""
    assert "must be a multiple of num_generations" in error_text(batch_size=12, gradient_accumulation_steps=1)


def test_steps_per_generation_derives_the_generation_batch() -> None:
    assert "must be a multiple of num_generations" in error_text(
        batch_size=4, steps_per_generation=3, num_generations=8
    )
    assert build(batch_size=4, steps_per_generation=4, num_generations=8).steps_per_generation == 4


def test_generation_batch_and_steps_are_exclusive() -> None:
    """The trainer derives one from the other, so setting both is ambiguous."""
    assert "mutually exclusive" in error_text(generation_batch_size=8, steps_per_generation=1)


def test_eval_batch_needs_whole_prompt_groups() -> None:
    assert "eval_batch_size" in error_text(eval_batch_size=4)


def test_eval_batch_uses_num_generations_eval_when_set() -> None:
    """Evaluation can sample fewer completions per prompt, which changes the batch requirement."""
    assert "eval_batch_size (8) must be a multiple of num_generations (3)" in error_text(num_generations_eval=3)
    assert build(num_generations_eval=4, eval_batch_size=8).num_generations_eval == 4


def test_single_completion_per_prompt_is_allowed() -> None:
    """The distillation loss has no group advantage, so one rollout per prompt is enough."""
    config = build(num_generations=1, num_generations_eval=1)

    assert config.num_generations == 1
    assert config.num_generations_eval == 1
    assert "num_generations must be at least 1" in error_text(num_generations=0)
    assert "num_generations_eval must be at least 1" in error_text(num_generations_eval=0)


def test_sampled_token_distillation_requires_reverse_kl() -> None:
    assert "distillation_alpha=1.0" in error_text(distillation_mode="sampled_token", distillation_alpha=0.5)
    assert build(distillation_mode="sampled_token", distillation_alpha=1.0).distillation_mode == "sampled_token"


def test_teacher_server_requires_vllm_server_and_live_teacher() -> None:
    """Teacher logprobs are served by the vLLM generation server, which only holds live student weights."""
    assert "use_teacher_server requires use_vllm=True" in error_text(use_teacher_server=True)

    assert "requires vllm_mode='server'" in error_text(use_teacher_server=True, use_vllm=True)

    assert "only supports teacher_model_kind='live'" in error_text(
        use_teacher_server=True, use_vllm=True, vllm_mode="server"
    )

    assert "distillation_mode in {'sampled_token', 'topk_logits'}" in error_text(
        use_teacher_server=True,
        use_vllm=True,
        vllm_mode="server",
        teacher_model_kind="live",
        distillation_mode="full_logits",
    )

    assert build(
        use_teacher_server=True, use_vllm=True, vllm_mode="server", teacher_model_kind="live"
    ).use_teacher_server


def test_ema_teacher_update_rate_is_a_rate() -> None:
    assert "teacher_update_rate" in error_text(teacher_model_kind="ema", teacher_update_rate=1.5)
    assert build(teacher_model_kind="ema", teacher_update_rate=1.0).teacher_update_rate == 1.0


def test_teacher_prompt_template_needs_both_placeholders() -> None:
    assert "{privileged_context}" in error_text(teacher_prompt_template="{prompt}")
    assert "{prompt}" in error_text(teacher_prompt_template="{privileged_context}")


def test_vllm_context_window_fits_prompt_and_completion() -> None:
    assert "vllm_max_model_length" in error_text(
        use_vllm=True, max_prompt_length=512, max_completion_length=256, vllm_max_model_length=512
    )
    assert (
        build(
            use_vllm=True, max_prompt_length=512, max_completion_length=256, vllm_max_model_length=768
        ).vllm_max_model_length
        == 768
    )


def test_sampling_parameters_are_in_range() -> None:
    assert "temperature" in error_text(temperature=0.0)
    assert "top_p" in error_text(top_p=0.0)
    assert "min_p" in error_text(min_p=1.5)
    assert "top_k" in error_text(top_k=-1)
    assert "repetition_penalty" in error_text(repetition_penalty=0.0)


def test_unknown_vllm_mode_is_rejected() -> None:
    assert "vllm_mode" in error_text(vllm_mode="colocated")


def test_every_problem_is_reported_at_once() -> None:
    """One startup run should list all mistakes, not just the first one found."""
    message = error_text(eval_batch_size=0, temperature=0.0, num_generations=0, use_teacher_server=True)

    assert "eval_batch_size must be at least 1" in message
    assert "temperature must be positive" in message
    assert "num_generations must be at least 1" in message
    assert "use_teacher_server requires use_vllm=True" in message
