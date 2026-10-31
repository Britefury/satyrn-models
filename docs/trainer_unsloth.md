# Model training

Use the `satyrn-unsloth` package to fine-tune models with Unsloth.

## Install

Prerequisites:
- make sure you have a Python version matching `>=3.13.9,<3.14`

Install the `satyrn-unsloth` command into your environment from the `trainer/unsloth` package.

```sh
uv pip install -e ./trainer/unsloth
```

Unsloth installs `trl==0.24.0` which has a known bug [trl#6105](https://github.com/huggingface/trl/issues/6105).
We need to upgrade `trl`:

```sh
uv pip install "trl==1.7.0"
```

### Transformers version

**Note:** Different `transformers` versions are required for different models.

For Mellum 2 we need `transformers>=5.10.1`. Tested on `5.15.0`, install it using:

```sh
uv pip install "transformers==5.15.0" "torch==2.11.0" "torchvision==0.26.0"
```

Gemma 4 does not work with `transformers==5.15.0`, stick with the Unsloth-shipped version:

```sh
uv pip install "transformers==5.5.0" "torch==2.11.0" "torchvision==0.26.0"
```

### CUDA build

PyPI `torch==2.11.0` is a CUDA 13 build (`+cu130`). If using older 570-series Nvidia drivers and CUDA 12.8, you will need a build from `pytorch.org` instead:

```sh
uv pip install "torch==2.11.0" "torchvision==0.26.0" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

## Run a fine-tuning experiment

Pass the name of an experiment config to `satyrn-unsloth`.

```sh
satyrn-unsloth --config-name experiment/pep750-qwen2.5-0.5b
```

In experiment config `py3.15` you can swap the model with `model=`, naming any config in 
`trainer/unsloth/configs/model`:

```sh
satyrn-unsloth --config-name experiment/py3.15 model=qwen3.6-27b
```

Anything set in the experiment config can be overridden on the same line:

```sh
satyrn-unsloth --config-name experiment/py3.15 model=qwen3-coder-30b-a3b load_in_4bit=true cpt.batch_size=2
```

## Self-distillation

Self-distillation runs CPT, rewrites the SFT datasets with the resulting model, then resumes with SFT
on the rewritten data. Each step is a separate process, so the GPU is free for the next one.

1. Run CPT only, saving a checkpoint:

   ```sh
   satyrn-unsloth --config-name experiment/py3.15-qwen3.8-27b \
     save_cpt_checkpoint=true datasets.sft=null datasets.rl=null
   ```

   This writes two formats to `outputs/<run_name>/cpt`: `adapter/`, for resuming, and `merged/`, a
   full model for vLLM. The run fails at startup if that directory already exists.
   `merged/` takes about 2 bytes per parameter, roughly 54 GB for Qwen3.8-27B, and can be deleted
   after distillation.

2. Serve `merged/` and distill every SFT file the experiment uses. See
   [Distill SFT datasets](dataset_generator.md#distill-sft-datasets). Stop the server to free the GPU.

3. Resume with SFT on the distilled data:

   ```sh
   satyrn-unsloth --config-name experiment/py3.15-qwen3.8-27b-distilled
   ```

The `-distilled` experiments set `resume_adapter`, which continues training the same LoRA adapter,
matching an in-process CPT → SFT run. `r` and `lora_alpha` must match the config. The stages that
follow, including RL where the base experiment has it, run as usual.

Compare each distilled run with its vanilla run in MLflow, within a model and with the same CPT
settings. `humaneval` is the proxy for general ability, which should improve or hold; `python_eval`
and `eval_qa` cover the target task. Record the share of distilled rows next to each result.
