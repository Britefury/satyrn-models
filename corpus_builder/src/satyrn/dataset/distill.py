"""Self-distillation of SFT datasets (Yang et al., arXiv:2402.13669).

The seed model rewrites each response given the instruction and the original (Eq. 3). A rewrite that
passes the checks replaces the original; otherwise the original is kept (Eq. 4).
"""

import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import click
from tqdm import tqdm

from satyrn.dataset.llm.models import Completion, VllmClient
from satyrn.dataset.utils.generation import append_dataset_line, read_jsonl_file
from satyrn.dataset.utils.sandbox import (
    Sandbox,
    docker_available,
    get_predecessor_python_version,
    gvisor_available,
    remove_leftover_containers,
)

logger = logging.getLogger(__name__)

# The paper's Fig. 3, verbatim.
PROMPT_TEMPLATE = """Below are an instruction that describes a task along with a reference answer. \
Using the reference answer as a guide, write your own response.
### Instruction:
{instruction}
### Reference Answer:
{original_response}
### Response:"""

LEAK_PATTERN = re.compile(r"reference (?:answer|response)", re.IGNORECASE)
PYTHON_BLOCK_PATTERN = re.compile(r"^```python[ \t]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)


@dataclass(frozen=True)
class DistillSettings:
    thinking: bool
    max_tokens: int
    max_seq_len: int
    max_attempts: int
    match_expected_output: bool


@dataclass
class Rewrite:
    """A response that passed every check."""

    completion: Completion
    code: str
    output: str


def build_prompt(row: dict) -> str:
    """Return the rewrite prompt for row. The original trace is left out, so the model reasons for itself."""
    return PROMPT_TEMPLATE.format(
        instruction=row["prompt"][-1]["content"], original_response=row["completion"][0]["content"]
    )


def check_syntax(code: str, sandbox: Sandbox) -> bool:
    """Return True if code parses, under the sandbox's Python because the host's may be older."""
    return sandbox.execute(f'compile({code!r}, "<rewrite>", "exec")').exit_code == 0


def check_completion(
    row: dict,
    completion: Completion,
    client: VllmClient,
    sandbox: Sandbox,
    predecessor_sandbox: Sandbox,
    settings: DistillSettings,
) -> Rewrite | str:
    """Return the verified rewrite, or the reason for the first check that failed."""
    if completion.finish_reason != "stop":
        return "truncated"

    content = completion.content.strip()
    reasoning = completion.reasoning.strip()
    if not content or (settings.thinking and not reasoning):
        return "empty"

    # sft.seq_len limits the whole rendered row, so max_tokens alone doesn't guarantee a fit.
    instruction_tokens = client.count_tokens(row["prompt"][-1]["content"])
    if instruction_tokens + completion.completion_tokens > settings.max_seq_len:
        return "too_long"

    if LEAK_PATTERN.search(content) or LEAK_PATTERN.search(reasoning):
        return "leak"

    blocks = PYTHON_BLOCK_PATTERN.findall(content)
    if len(blocks) != 1:
        return "code_blocks"
    code = blocks[0]

    if not check_syntax(code, sandbox):
        return "syntax"

    result = sandbox.execute(code)
    if result.exit_code != 0:
        return "runtime"

    # Exit code 1 is an uncaught exception. A timeout or an infrastructure exit code (125, 137) is not a pass.
    if predecessor_sandbox.execute(code).exit_code != 1:
        return "no_new_feature"

    if settings.match_expected_output and result.output.strip() != row["expected_output"].strip():
        return "output_mismatch"

    return Rewrite(completion, code, result.output)


def distill_row(row: dict, client: VllmClient, sandboxes: dict[str, tuple], settings: DistillSettings) -> dict:
    """Return row rewritten by the model, or unchanged but for the rejection if no attempt passed."""
    sandbox, predecessor_sandbox = sandboxes[row["python_version"]]
    prompt = build_prompt(row)

    # A retry resamples the same prompt: hints would pull the output away from the model's distribution.
    rejection = None
    for _ in range(settings.max_attempts):
        completion = client.complete(prompt, settings.thinking, settings.max_tokens)
        outcome = check_completion(row, completion, client, sandbox, predecessor_sandbox, settings)
        if isinstance(outcome, str):
            rejection = outcome
            continue

        distilled_row = {
            **row,
            "completion": [{"role": "assistant", "content": outcome.completion.content.strip()}],
            "code": outcome.code,
            "expected_output": outcome.output,
            "distilled": True,
            "distill_rejection": None,
        }
        if settings.thinking:
            # The trainer uses trace as reasoning_content, so a stale trace would describe the old code.
            distilled_row["trace"] = outcome.completion.reasoning.strip()
        return distilled_row

    return {**row, "distilled": False, "distill_rejection": rejection}


def count_written_rows(output_path: Path) -> int:
    """Return how many rows a previous run wrote. A partly written last row fails here, not in training."""
    if not output_path.exists():
        return 0
    return sum(1 for _ in read_jsonl_file(output_path))


def summarize(output_path: Path) -> str:
    """Return the run's report, counted from the output file so resumed runs are included."""
    rows = list(read_jsonl_file(output_path))
    distilled = sum(row["distilled"] for row in rows)
    rejections = Counter(row["distill_rejection"] for row in rows if not row["distilled"])

    lines = [
        f"Distilled {distilled} of {len(rows)} rows",
        f"Kept the original for {len(rows) - distilled} rows",
    ]
    lines += [f"  {reason}: {count}" for reason, count in rejections.most_common()]
    return "\n".join(lines)


@click.command("distill")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="SFT JSONL file to distill.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file to write. If it exists, the run resumes after its last row.",
)
@click.option("--model", "model_name", default="seed", show_default=True, help="Model name served by vLLM.")
@click.option("--thinking", is_flag=True, default=False, help="Enable thinking and replace each row's trace.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=1280, show_default=True, help="Response limit.")
@click.option(
    "--max-seq-len",
    type=click.IntRange(min=1),
    default=1280,
    show_default=True,
    help="Limit on instruction plus response tokens. Match the experiment's sft.seq_len.",
)
@click.option("--workers", type=click.IntRange(min=1), default=1, help="Number of rows to distill in parallel.")
@click.option("--limit", type=click.IntRange(min=1), help="Only distill the first LIMIT input rows.")
@click.option(
    "--max-attempts", type=click.IntRange(min=1), default=1, show_default=True, help="Samples to try for each row."
)
@click.option("--match-expected-output", is_flag=True, default=False, help="Also require the original row's output.")
@click.option("--allow-unsandboxed", is_flag=True, default=False, help="Run model code without gVisor.")
def main(
    input_path: Path,
    output_path: Path,
    model_name: str,
    thinking: bool,
    max_tokens: int,
    max_seq_len: int,
    workers: int,
    limit: int | None,
    max_attempts: int,
    match_expected_output: bool,
    allow_unsandboxed: bool,
) -> None:
    """Rewrite an SFT dataset with the seed model, keeping the original where a rewrite fails its checks."""
    if output_path.suffix != ".jsonl":
        raise click.BadParameter("Output file must end with .jsonl")
    # The code comes from locally tuned models.
    if not allow_unsandboxed and not (docker_available() and gvisor_available()):
        raise click.ClickException("gVisor (runsc) is not registered with Docker. Pass --allow-unsandboxed to skip it.")

    rows = list(read_jsonl_file(input_path))
    if limit is not None:
        rows = rows[:limit]

    written = count_written_rows(output_path)
    if written > len(rows):
        raise click.ClickException(f"{output_path} has {written} rows, more than the {len(rows)} to distill.")
    logger.info("Distilling %d rows, skipping %d already written", len(rows) - written, written)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows[written:]

    client = VllmClient(model_name)
    sandboxes = {
        python_version: (Sandbox(python_version), Sandbox(get_predecessor_python_version(python_version)))
        for python_version in {row["python_version"] for row in rows}
    }
    settings = DistillSettings(thinking, max_tokens, max_seq_len, max_attempts, match_expected_output)

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        # map, not as_completed: rows stay in input order, so the trainer's seeded split picks the same
        # train and eval rows as the undistilled baseline, and a restart can skip by count.
        distill = partial(distill_row, client=client, sandboxes=sandboxes, settings=settings)
        for dataset_line in tqdm(executor.map(distill, rows), total=len(rows), desc="Rows"):
            append_dataset_line(dataset_line, output_path)
    finally:
        # A server failure stops the run here, rather than marking the remaining rows as rejected.
        executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Cleaning up sandbox containers...")
        removed_count = remove_leftover_containers()
        logger.info("Removed %d leftover sandbox containers", removed_count)

    logger.info("%s", summarize(output_path))
