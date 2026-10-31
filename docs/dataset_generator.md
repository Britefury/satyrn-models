# Dataset generation

Use the `corpus_builder` package to generate fine-tuning datasets.

## Install

Install the `satyrn-dataset` command into your environment from the `corpus_builder` package.

Activate a Python virtual environment and install the package. Use an editable install (`-e`) 
if you plan to change the `corpus_builder` code, so edits take effect without reinstalling:

```sh
$ source /path/to/venv/bin/activate
(venv) $ pip install -e ./corpus_builder
```

## Collecting input material

Before generating a dataset, gather the source documents it draws from.

### Gather Python Enhancement Proposals (PEPs)

Download the PEPs for a Python version into `datasets/python3.15/input/docs/`

```sh
satyrn-dataset download-inputs 3.15
```

### Gather recent changes in CPython Docs

To collect and download the documentation changes for CPython, perform the following steps.

Clone CPython into any path on your system:

```sh
git clone --depth 1 https://github.com/python/cpython.git /path/to/cpython
```

Collect the documentation sections tied to changes in a Python version. Make sure that you use `Doc` 
directory of the CPython repo as the input:

```sh
satyrn-dataset collect-doc-changes 3.15 /path/to/cpython/Doc
```

Writes to `datasets/python3.15/input/docs/changes/`.

## Generating datasets

Each subcommand writes JSONL to the `--output` file.
`cpt` reads source material from `--input-dir` (a directory).
`sft` and `rl` are generated in two stages: `ideas` first reads source material from `--input`
(a directory or a single doc file) and writes a JSONL file of ideas, then `sft` and `rl` read
those ideas from `--ideas-path`. The same ideas file can be reused for several `sft` or `rl` runs.

```sh
satyrn-dataset cpt   --input-dir DIR --output FILE.jsonl           # Continued Pretraining (CPT)
satyrn-dataset ideas --input PATH --output IDEAS.jsonl             # Ideas for SFT and RL
satyrn-dataset sft   --ideas-path IDEAS.jsonl --output FILE.jsonl  # Supervised Fine-Tuning (SFT)
satyrn-dataset rl    --ideas-path IDEAS.jsonl --output FILE.jsonl  # Evaluation / Reinforcement Learning (RL)
```

`ideas`, `sft` and `rl` use an LLM via the DeepSeek API. A `DEEPSEEK_API_KEY` is required in your
environment (or a `.env` file - see `.env.example`). Optionally set `LLM_MODEL_NAME` to choose the
model (default `deepseek-v4-flash`) and `DEEPSEEK_BASE_URL` to use a different DeepSeek-compatible
endpoint (default `https://api.deepseek.com`).

### Generate Continued Pretraining (CPT) datasets

`cpt` converts every `.rst` file under `input-dir` to Markdown and writes one row per file to
`--output`.

Usage example:

```sh
satyrn-dataset cpt --input-dir datasets/python3.14/input/docs --output datasets/python3.14/cpt.jsonl
satyrn-dataset cpt --input-dir datasets/python3.15/input/docs --output datasets/python3.15/cpt.jsonl
```

### Generate ideas

`ideas` asks an LLM to propose up to 50 distinct ideas for short code examples demonstrating the
changes described in each source document. Each row of `--output` holds one idea:

- `description`: what the example should demonstrate;
- `python_version`: the target Python version; and
- `doc_path`: the absolute path of the source document the idea came from.

Because `doc_path` is absolute and `sft`/`rl` re-read the source document, keep the source documents
in place, and generate ideas on the same machine (or with the same paths) that will run `sft`/`rl`.

If the LLM repeatedly fails to return output matching the expected schema for a document, that
document is skipped with a warning and the run continues.

Pass `--preview` to print each document's ideas as they are saved, and `--workers` to process
several source documents in parallel (defaults to 1). Docker is not needed for this step.

Generate ideas for a single PEP with preview:

```sh
satyrn-dataset ideas -i datasets/python3.14/input/docs/PEP750.rst -o datasets/python3.14/ideas.jsonl --python-version 3.14 --preview
```

Generate ideas across all downloaded documents, 8 documents at a time:

```sh
satyrn-dataset ideas -i datasets/python3.15/input/docs -o datasets/python3.15/ideas.jsonl --python-version 3.15 --workers 8
```

### Generate Supervised Fine-Tuning (SFT) datasets

`sft` uses an LLM to turn each idea from an [ideas file](#generate-ideas) into a prompt-response
pair demonstrating a new Python feature, using the idea's source document as context.

Each idea is verified by running in sandboxed Docker containers for the target Python version and
its immediate predecessor. Examples are rejected if both versions produce equivalent output.

Docker is _required_, gVisor (`runsc` runtime) is _recommended_ for sandbox safety. You can verify if both are set 
up in your environment:

```sh
docker run --rm hello-world
docker run --rm --runtime=runsc hello-world
```

Output is written one row per conversation to `--output`, appending as it goes so partial progress
survives an interruption.

#### Usage

Pass `--preview` to print each generated conversation to the terminal as it's saved, useful for
sanity-checking output quality while a run is in progress.

Generating examples from an ideas file with preview:

```sh
satyrn-dataset sft -i datasets/python3.14/ideas.jsonl -o datasets/python3.14/sft.jsonl --python-version 3.14 --preview
```

Pass `--workers` to generate several conversations (one per idea) in parallel. Defaults to 1.

Generating examples 24 at a time:

```sh
satyrn-dataset sft -i datasets/python3.15/ideas.jsonl -o datasets/python3.15/sft.jsonl --python-version 3.15 --workers 24
```

### Distill SFT datasets

`distill` implements Self-Distillation Fine-Tuning ([Yang et al.](https://arxiv.org/abs/2402.13669)).
A seed model rewrites each SFT response, given the instruction and the original response as a
reference. A rewrite that passes the checks below replaces the original; otherwise the original is
kept. Training on the result stays closer to the model's own distribution than training on the
originals.

The seed is the model after CPT, because that is the model SFT starts from. The full workflow is in
[Self-distillation](trainer_unsloth.md#self-distillation).

#### Serve the seed model

vLLM has its own environment in `serving/vllm`, which is Linux-only:

```sh
cd serving/vllm
uv sync
uv run vllm serve ../../outputs/py3.15-Qwen3.8-27B/cpt/merged --served-model-name seed --port 8000 \
  --reasoning-parser <per model>
```

Point `distill` at the server root, without `/v1`, using `VLLM_BASE_URL`, which `.env.example` sets
to `http://localhost:8000`.

#### Usage

Smoke test on the first 20 rows, then read them in the [viewer](#viewing-datasets):

```sh
satyrn-dataset distill -i datasets/python3.15/sft.jsonl \
  -o datasets/python3.15/distilled/qwen3.8-27b/sft.jsonl \
  --thinking --workers 16 --limit 20
```

Pass `--thinking` for models trained with `enable_thinking: true`. Without it, `trace` is left as
is. Set `--max-seq-len` to the experiment's `sft.seq_len`; it and `--max-tokens` default to 1280.

Rows are written in input order, so a restart resumes after the last row in the output file, and
the trainer's seeded split picks the same train and eval rows as the undistilled dataset. Delete the
output file to start again.

The code comes from locally tuned models, so the command exits at startup if gVisor isn't registered
with Docker, unless you pass `--allow-unsandboxed`.

#### Checks

The first check that fails keeps the original row, and is recorded in `distill_rejection`:

| Rejection | The rewrite is rejected if |
| --- | --- |
| `truncated` | `finish_reason` is not `stop` |
| `empty` | Its content is empty, or its reasoning is with `--thinking` |
| `too_long` | Instruction tokens plus completion tokens exceed `--max-seq-len` |
| `leak` | Its content or reasoning mentions a `reference answer` or `reference response` |
| `code_blocks` | It doesn't have exactly one ` ```python ` block |
| `syntax` | The code has a syntax error on the target Python |
| `runtime` | The code doesn't exit 0 within the timeout on the target Python |
| `no_new_feature` | The code doesn't exit 1 on the immediately preceding Python |
| `output_mismatch` | With `--match-expected-output`: its output differs from the original's |

`no_new_feature` is stricter than `sft`, which only requires the output to differ. Rewrites of rows
that show changed behaviour without failing on the prior version are rejected, as are rewrites of
rows that deliberately end in an error.

`--max-attempts` resamples a rejected row, with no feedback to the model. It defaults to 1, which
matches the paper.

The end-of-run report gives the number of distilled rows, the paper's mix ratio, and a count per
rejection reason. Few distilled rows means the run is mostly vanilla SFT and says little about
distillation.

#### Output

Rows have the input's schema plus `distilled` and `distill_rejection`. An accepted row replaces
`completion`, `code` and `expected_output`, and with `--thinking` also `trace`. `expected_output`
becomes the rewrite's actual output on the target Python.

### Import NVIDIA OpenCodeReasoning

`opencode-reasoning` streams every config and split from
[`nvidia/OpenCodeReasoning`](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) and writes one
Satyrn SFT JSONL file per split. It maps `input` to the user prompt, separates a leading `<think>`
block into `trace`, and stores the remaining answer as the assistant completion.

We created four deterministic, deduplicated, difficulty-balanced samples in: `opencode_reasoning_split1.jsonl` through `opencode_reasoning_split4.jsonl`, with 2,000 rows per file.

### Generate evaluation and Reinforcement Learning datasets

`rl` turns each idea from an [ideas file](#generate-ideas) into a small callable programming task,
using the idea's source document as context. Each JSONL row contains:

- `prompt`: a self-contained problem and callable signature;
- `test_cases`: 5-12 independent checks, each with a documented input, expected output, and executable assertion;
- `solution`: a reference implementation used to verify the tests; and
- `metadata`: the source document, normalized PEP identifier when applicable, Python version,
  originating idea, entry point, and test count.

The reference solution must pass every test under the target Python version. The same solution and
suite are then run under the immediately preceding version, and the row is rejected if all tests
still pass there. A candidate model response can be scored by running every test independently and
using `passed_tests / total_tests`, producing a score between 0 and 1.

Docker has the same requirements as for `sft`. Generate from an ideas file with preview:

```sh
satyrn-dataset rl -i datasets/python3.15/ideas.jsonl -o datasets/python3.15/rl.jsonl --python-version 3.15 --preview
```

Generate with parallel workers (one task per idea at a time):

```sh
satyrn-dataset rl -i datasets/python3.15/ideas.jsonl -o datasets/python3.15/rl.jsonl --python-version 3.15 --workers 24
```

## Viewing datasets

`viewer` serves a JSONL dataset (CPT, SFT or RL) in the browser, one row at a time. Prompts and
messages are rendered as Markdown with Python syntax highlighting, prompt/completion pairs are shown
as user/assistant turns (with any `trace` shown as the assistant's thinking), and RL rows show their
tests, reference solution and metadata. The sidebar lists every row and can be filtered by text.
Malformed lines are skipped and reported rather than stopping the viewer from loading.

### Build the frontend

The viewer UI is a TypeScript app in `corpus_builder/frontend`. Its built bundle is not checked in,
so build it once (and again after pulling frontend changes). Node.js and npm are required:

```sh
cd corpus_builder/frontend
npm install
npm run build
```

This writes the bundle to `corpus_builder/src/satyrn/dataset/viewer/static/`, where the
`satyrn-dataset` package serves it from. If you skip this step, the viewer responds with a
"Frontend not built" page.

### Run the viewer

```sh
satyrn-dataset viewer datasets/python3.15/rl.jsonl
```

Then open <http://127.0.0.1:5000/>. Options:

- `--port PORT`: port to listen on (default `5000`);
- `--host HOST`: interface to bind to (default `127.0.0.1`; use `0.0.0.0` to reach it from another machine);
- `--debug`: enable the Flask debug reloader.

The dataset is read once at startup, so restart the viewer to see rows added after it started.

For frontend development with hot reloading, see `corpus_builder/frontend/README.md`.
