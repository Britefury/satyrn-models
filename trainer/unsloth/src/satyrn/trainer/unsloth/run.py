"""Entry point for the satyrn-unsloth trainer CLI."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from datasets import Dataset
from hydra.core.hydra_config import HydraConfig
from inspect_evals.humaneval import humaneval
from jinja2 import Environment
from omegaconf import DictConfig

from satyrn.trainer.unsloth.config import ExperimentConfig, StageName, basename, log_config, validate_config
from satyrn.trainer.unsloth.dataset_packing import pack_documents
from satyrn.trainer.unsloth.eval.inspect_runner import run_inspect_eval
from satyrn.trainer.unsloth.eval.python_eval import python_eval
from satyrn.trainer.unsloth.eval.qa import run_eval_qa
from satyrn.trainer.unsloth.log_capture import tee_output
from satyrn.trainer.unsloth.rl.prompt import to_chat_prompt
from satyrn.trainer.unsloth.rl.reward import reward_correct_code
from satyrn.trainer.unsloth.secrets import load_secrets

if TYPE_CHECKING:
    from torch.nn import Module
    from transformers import PreTrainedTokenizerBase, ProcessorMixin, Trainer

logger = logging.getLogger(__name__)
logging.getLogger("satyrn").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

CONFIG_DIR = str(Path(__file__).resolve().parents[4] / "configs")


def unsloth_init() -> None:
    """Initialize unsloth and patch the training libraries."""
    global FastModel, FastVisionModel, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer, SDFTConfig, SDFTTrainer, torch, train_on_responses_only

    # Unsloth must be imported first to patch transformers, accelerate, etc.
    from unsloth import FastModel, FastVisionModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.experimental.sdft import SDFTConfig, SDFTTrainer

    # In notebooks, NotebookProgressCallback logs loss to an IPython handle we cannot capture in the logs
    import transformers.trainer
    from transformers.trainer_callback import ProgressCallback

    transformers.trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallback


def load_dataset(paths: str | list[str]) -> Dataset:
    """Read one or more JSONL files into a single dataset."""
    if isinstance(paths, str):
        paths = [paths]

    rows = []
    for path in paths:
        with Path(path).open() as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return Dataset.from_list(rows)


def render_conversations_into_text_field(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase, enable_thinking: bool
) -> Dataset:
    """Render each prompt+completion row into the text column the trainer tokenizes."""

    def render(row: dict) -> dict:
        assistant = dict(row["completion"][0])
        if enable_thinking:
            assistant["reasoning_content"] = row["trace"]
        return {"text": tokenizer.apply_chat_template(row["prompt"] + [assistant], tokenize=False)}

    # A row still holding prompt and completion columns takes the trainer's prompt-completion
    # path, which masks by prompt length and ignores the text column.
    return dataset.map(render, remove_columns=dataset.column_names)


def render_sdft_conversations_into_text_field(
    dataset: Dataset, enable_thinking: bool, privileged_context_template: str
) -> Dataset:
    """
    Reshape each row into `{"prompt": <messages>, "privileged_context": <text>}` for SDFTTrainer.

    The prompt stays conversational: SDFTTrainer applies the chat template itself, with
    `add_generation_prompt=True` and its configured `chat_template_kwargs`, and the teacher
    context builder needs the message list to extract the last user turn. The privileged context is
    a plain string because the trainer's stringifier only reads message `content`, dropping any
    separate reasoning field; it is rendered from a Jinja2 template compiled once here."""
    template = Environment(keep_trailing_newline=True).from_string(privileged_context_template)

    def render(row: dict[str, Any]) -> dict[str, Any]:
        """
        `row` is a dict that takes the form `{
            "prompt": [{"role": "user", "content": <prompt>}],
            "completion": [{"role": "assistant", "content": <response>}],
            "filename": idea.doc_path.name,
            "pep": pep_identifier(idea.doc_path),
            "python_version": idea.python_version,
            "idea": idea.description,
            "code": code_block["code"],
            "trace": code_block["trace"],
            "expected_output": code_block["expected_output"],
        }`
        """
        return {
            "prompt": row["prompt"],
            "privileged_context": template.render(
                trace=row["trace"] if enable_thinking else None,
                code=row["code"],
                answer=row["completion"][0].get("content", ""),
                expected_output=row["expected_output"],
            ),
        }

    # Keep only the columns SDFTTrainer declares in _signature_columns; a lingering completion
    # column would take the trainer's prompt-completion path instead of the rollout path.
    return dataset.map(render, remove_columns=dataset.column_names)


def build_trainer(
    name: StageName,
    model: Module,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    cfg: ExperimentConfig,
    packing: bool = False,
    packing_strategy: str = "bfd",  # best-fit, decreasing document size order; truncates over max_length
) -> SFTTrainer:
    """Split the dataset and build the trainer for one stage."""
    stage = getattr(cfg, name)
    split = dataset.train_test_split(test_size=cfg.eval_ratio, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]

    training_args = SFTConfig(
        output_dir=f"outputs/{name}",
        per_device_train_batch_size=stage.batch_size,
        per_device_eval_batch_size=stage.eval_batch_size,
        gradient_accumulation_steps=stage.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=stage.eval_steps,
        report_to="mlflow",
        max_length=stage.seq_len,
        num_train_epochs=stage.num_train_epochs,
        max_steps=cfg.max_steps,
        learning_rate=stage.learning_rate,
        shuffle_dataset=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=cfg.optim,
        packing=packing,
        packing_strategy=packing_strategy,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    return trainer


def log_dataset_params(dataset_path: str | list[str], trainer: Trainer) -> None:
    """Record the stage's dataset path and training row count on the active MLflow run."""
    mlflow.log_params({"dataset_path": dataset_path, "dataset_train_rows": len(trainer.train_dataset)})


def evaluate_model(stage_name: StageName, model: Module, tokenizer: PreTrainedTokenizerBase) -> None:
    """Run every eval against the model as it stands after stage_name."""
    run_eval_qa(stage_name, model, tokenizer)
    run_inspect_eval(stage_name, model, tokenizer, humaneval(sandbox="local"))
    run_inspect_eval(stage_name, model, tokenizer, python_eval())


def upload_to_hub(model: Module, processor: ProcessorMixin, config: ExperimentConfig) -> None:
    """Push the trained model to the Hugging Face Hub."""
    hub_model_id = config.push_to_hub_path or f"{os.environ['HF_USERNAME']}/{basename(config.model.name)}"
    logger.info("Pushing merged model to Hugging Face Hub: https://huggingface.co/%s", hub_model_id)
    model.push_to_hub_merged(hub_model_id, processor, save_method="merged_16bit")
    model.push_to_hub_gguf(hub_model_id, processor, quantization_method="q4_k_m")


@hydra.main(config_path=CONFIG_DIR)
def main(cfg: DictConfig) -> None:
    if not cfg:
        print("Usage: satyrn-unsloth --config-name experiment/<name>")
        return

    log_path = Path(HydraConfig.get().runtime.output_dir, "run.log")
    with tee_output(log_path) as log_file:
        log_config(cfg)
        # Validate before the expensive unsloth/transformers import so a bad config fails in seconds.
        config = validate_config(cfg)

        unsloth_init()
        load_secrets()

        logger.info("Downloading model %s", config.model.name)
        model, processor = FastVisionModel.from_pretrained(
            model_name=config.model.name,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
        )
        # Some model configs ship use_cache=false, which makes generate() recompute every token.
        model.generation_config.use_cache = True
        # Multimodal models return a Processor; text-only training uses its tokenizer.
        tokenizer = getattr(processor, "tokenizer", processor)

        model = FastModel.get_peft_model(
            model,
            r=config.model.peft.r,
            lora_alpha=config.model.peft.lora_alpha,
            lora_dropout=config.model.peft.lora_dropout,
            target_modules=config.model.peft.target_modules,
            finetune_attention_modules=config.model.peft.finetune_attention_modules,
            finetune_mlp_modules=config.model.peft.finetune_mlp_modules,
            finetune_language_layers=config.model.peft.finetune_language_layers,
            finetune_vision_layers=config.model.peft.finetune_vision_layers,
            finetune_audio_layers=config.model.peft.finetune_audio_layers,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        # logger.info("PEFT model %s", model)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        trainable_percentage = 100 * trainable_params / all_params
        logger.info("Trainable params: %s", f"{trainable_params:,}")
        logger.info("All params: %s", f"{all_params:,}")
        logger.info("Trainable: %.4f%%", trainable_percentage)

        mlflow.enable_system_metrics_logging()
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

        with mlflow.start_run(run_name=config.run_name):
            try:
                mlflow.log_params(
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "base_model": config.model.name,
                        "lora_rank": config.model.peft.r,
                        "lora_alpha": config.model.peft.lora_alpha,
                        "lora_dropout": config.model.peft.lora_dropout,
                        "lora_targets": ",".join(config.model.peft.target_modules or []),
                        "finetune_attention_modules": config.model.peft.finetune_attention_modules,
                        "finetune_mlp_modules": config.model.peft.finetune_mlp_modules,
                        "finetune_language_layers": config.model.peft.finetune_language_layers,
                        "finetune_vision_layers": config.model.peft.finetune_vision_layers,
                        "finetune_audio_layers": config.model.peft.finetune_audio_layers,
                        "params_train": trainable_params,
                        "params_total": all_params,
                        "params_train_pct": trainable_percentage,
                    }
                )

                if config.pre_model_eval:
                    logger.info("Model evaluation before training")
                    evaluate_model("pre", model, tokenizer)

                if config.datasets.cpt is not None:
                    logger.info("Starting Continuous Pre-Training (CPT) stage")
                    dataset = load_dataset(config.datasets.cpt)
                    packing = config.cpt.packing

                    if packing and config.cpt.prepack_dataset:
                        packed = pack_documents(dataset, tokenizer, config.cpt.seq_len)
                        logger.info("Packed cpt: %d documents into %d sequences", len(dataset), len(packed))
                        dataset = packed
                        logger.warning("Prepacking done, setting packing=false to avoid double packing")
                        packing = False

                    trainer = build_trainer(
                        "cpt",
                        model,
                        tokenizer,
                        dataset,
                        config,
                        packing=packing,
                        packing_strategy="bfd_split",  # best-fit, decreasing doc size order; splits over max_length
                    )

                    with mlflow.start_run(run_name="cpt", nested=True):
                        log_dataset_params(config.datasets.cpt, trainer)
                        trainer.train()

                    logger.info("Model evaluation after Continuous Pre-Training (CPT)")
                    evaluate_model("cpt", model, tokenizer)

                if config.datasets.sft is not None:
                    logger.info("Starting Supervised Fine-Tuning (SFT) stage")
                    dataset = load_dataset(config.datasets.sft)
                    dataset = render_conversations_into_text_field(dataset, tokenizer, config.model.enable_thinking)

                    trainer = build_trainer("sft", model, tokenizer, dataset, config)
                    trainer = train_on_responses_only(
                        trainer,
                        instruction_part=config.model.template.instruction_part,
                        response_part=config.model.template.response_part,
                    )

                    with mlflow.start_run(run_name="sft", nested=True):
                        log_dataset_params(config.datasets.sft, trainer)
                        trainer.train()

                    logger.info("Model evaluation after Supervised Fine-Tuning (SFT)")
                    evaluate_model("sft", model, tokenizer)

                if config.datasets.sdft is not None:
                    logger.info("Starting Self-distillation Fine-Tuning (SDFT) stage")
                    dataset = load_dataset(config.datasets.sdft)
                    dataset = render_sdft_conversations_into_text_field(
                        dataset,
                        config.model.enable_thinking,
                        config.sdft.privileged_context_template,
                    )

                    sdft_args = SDFTConfig(
                        output_dir="outputs/sdft",
                        per_device_train_batch_size=config.sdft.batch_size,
                        per_device_eval_batch_size=config.sdft.eval_batch_size,
                        gradient_accumulation_steps=config.sdft.gradient_accumulation_steps,
                        logging_steps=cfg.logging_steps,
                        eval_strategy="steps",
                        eval_steps=config.sdft.eval_steps,
                        report_to="mlflow",
                        num_train_epochs=config.sdft.num_train_epochs,
                        max_steps=config.max_steps,
                        learning_rate=config.sdft.learning_rate,
                        bf16=torch.cuda.is_bf16_supported(),
                        fp16=not torch.cuda.is_bf16_supported(),
                        optim=config.optim,
                        # SDFT loss parameters
                        distillation_alpha=config.sdft.distillation_alpha,
                        distillation_mode=config.sdft.distillation_mode,
                        distillation_topk=config.sdft.distillation_topk,
                        distillation_is_clip=config.sdft.distillation_is_clip,
                        distillation_add_tail=config.sdft.distillation_add_tail,
                        num_loss_tokens_to_skip=config.sdft.num_loss_tokens_to_skip,
                        # Teacher parameters
                        teacher_model_kind=config.sdft.teacher_model_kind,
                        teacher_update_rate=config.sdft.teacher_update_rate,
                        teacher_sync_steps=config.sdft.teacher_sync_steps,
                        # Teacher-conditioned generation
                        generate_from_teacher=config.sdft.generate_from_teacher,
                        teacher_prompt_template=config.sdft.teacher_prompt_template,
                        # Model parameters
                        model_init_kwargs=config.sdft.model_init_kwargs,
                        trust_remote_code=config.sdft.trust_remote_code,
                        disable_dropout=config.sdft.disable_dropout,
                        # Data preprocessing
                        remove_unused_columns=config.sdft.remove_unused_columns,
                        max_prompt_length=config.sdft.max_prompt_length,
                        shuffle_dataset=config.sdft.shuffle_dataset,
                        # Generation parameters
                        num_generations=config.sdft.num_generations,
                        num_generations_eval=config.sdft.num_generations_eval,
                        max_completion_length=config.sdft.max_completion_length,
                        temperature=config.sdft.temperature,
                        top_p=config.sdft.top_p,
                        top_k=config.sdft.top_k,
                        min_p=config.sdft.min_p,
                        repetition_penalty=config.sdft.repetition_penalty,
                        cache_implementation=config.sdft.cache_implementation,
                        generation_kwargs=config.sdft.generation_kwargs,
                        # enable_thinking drives the template the trainer applies; an explicit
                        # chat_template_kwargs entry may override it.
                        chat_template_kwargs={
                            "enable_thinking": config.model.enable_thinking,
                            **(config.sdft.chat_template_kwargs or {}),
                        },
                        ds3_gather_for_generation=config.sdft.ds3_gather_for_generation,
                        # vLLM generation acceleration
                        use_vllm=config.sdft.use_vllm,
                        use_teacher_server=config.sdft.use_teacher_server,
                        vllm_mode=config.sdft.vllm_mode,
                        vllm_model_impl=config.sdft.vllm_model_impl,
                        vllm_enable_sleep_mode=config.sdft.vllm_enable_sleep_mode,
                        vllm_server_base_url=config.sdft.vllm_server_base_url,
                        vllm_server_host=config.sdft.vllm_server_host,
                        vllm_server_port=config.sdft.vllm_server_port,
                        vllm_group_port=config.sdft.vllm_group_port,
                        vllm_server_timeout=config.sdft.vllm_server_timeout,
                        vllm_tensor_parallel_size=config.sdft.vllm_tensor_parallel_size,
                        vllm_gpu_memory_utilization=config.sdft.vllm_gpu_memory_utilization,
                        vllm_max_model_length=config.sdft.vllm_max_model_length,
                        # Training parameters
                        num_iterations=config.sdft.num_iterations,
                        generation_batch_size=config.sdft.generation_batch_size,
                        steps_per_generation=config.sdft.steps_per_generation,
                    )

                    sdft_train = SDFTTrainer(
                        model=model,
                        processing_class=tokenizer,
                        train_dataset=dataset,
                        args=sdft_args,
                    )

                    with mlflow.start_run(run_name="sdft", nested=True):
                        log_dataset_params(config.datasets.sdft, sdft_train)
                        sdft_train.train()

                    logger.info("Model evaluation after Self-distillation Fine-Tuning (SDFT)")
                    evaluate_model("sdft", model, tokenizer)

                if config.datasets.rl is not None:
                    logger.info("Starting Reinforcement Learning (RL) stage")
                    dataset = load_dataset(config.datasets.rl)
                    dataset = dataset.map(to_chat_prompt)

                    training_args = GRPOConfig(
                        output_dir="outputs/rl",
                        per_device_train_batch_size=config.rl.batch_size,
                        gradient_accumulation_steps=config.rl.gradient_accumulation_steps,
                        num_generations=config.rl.num_generations,
                        max_completion_length=config.rl.max_completion_length,
                        num_train_epochs=config.rl.num_train_epochs,
                        learning_rate=config.rl.learning_rate,
                        beta=config.rl.beta,
                        logging_steps=config.logging_steps,
                        max_steps=config.max_steps,
                        optim=config.optim,
                        report_to="mlflow",
                        bf16=torch.cuda.is_bf16_supported(),
                        fp16=not torch.cuda.is_bf16_supported(),
                        disable_tqdm=True,  # ProgressCallback breaks in marimo notebooks
                    )
                    trainer = GRPOTrainer(
                        model=model,
                        processing_class=tokenizer,
                        reward_funcs=[reward_correct_code],
                        args=training_args,
                        train_dataset=dataset,
                    )

                    with mlflow.start_run(run_name="rl", nested=True):
                        log_dataset_params(config.datasets.rl, trainer)
                        trainer.train()

                    logger.info("Model evaluation after Reinforcement Learning (RL)")
                    evaluate_model("rl", model, tokenizer)

                if config.push_to_hub:
                    upload_to_hub(model, processor, config)

            except Exception:
                logger.exception("Run failed")
                raise
            finally:
                # Send the Hydra job log to the tracking server
                log_file.flush()
                mlflow.log_text(log_path.read_text(), "train.log")

    mlflow.end_run()


if __name__ == "__main__":
    main()
