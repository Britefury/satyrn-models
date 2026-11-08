"""Per-epoch evaluation and checkpointing for a training stage.

transformers must be imported after unsloth patches it, so the callback class is built by
`make_epoch_end_callback` rather than defined at module level.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import PreTrainedTokenizerBase, TrainerCallback

logger = logging.getLogger(__name__)

# evaluate(step, tag) runs the evals against the model as it stands
EvaluateFn = Callable[[int, str], None]


def save_model(model: Module, tokenizer: PreTrainedTokenizerBase, path: Path) -> None:
    """Save the model and tokenizer to path; a PEFT model saves only its adapter weights."""
    from peft import PeftModel

    if isinstance(model, PeftModel):
        logger.info("Saving LoRA adapter to %s", path)
    else:
        logger.info("Saving full model to %s", path)
    # PeftModel.save_pretrained writes only the adapter and its config.
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))


def make_epoch_end_callback(
    stage_name: str,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    evaluate: EvaluateFn | None,
    save_dir: Path | None,
) -> TrainerCallback:
    """Build a callback that, at the end of each epoch but the last, evaluates and/or saves the model.

    Saves go to `save_dir/<stage_name>-epoch<N>`. The final epoch is left to the normal mechanisms: the post-stage
    eval, and the checkpoint the Trainer always writes at the final step (`<output_dir>/checkpoint-<step>`).
    """
    from transformers import TrainerCallback

    class EpochEndCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control

            # The final epoch (or a partial one cut short by max_steps) is covered by the post-stage eval and the
            # Trainer's end-of-training checkpoint.
            if control.should_training_stop or state.global_step >= state.max_steps:
                return control

            epoch = round(state.epoch)

            if save_dir is not None:
                save_model(model, tokenizer, save_dir / f"{stage_name}-epoch{epoch}")

            if evaluate is not None:
                logger.info("Model evaluation after %s epoch %d (step %d)", stage_name, epoch, state.global_step)
                evaluate(state.global_step, f"epoch{epoch}")

            return control

    return EpochEndCallback()
