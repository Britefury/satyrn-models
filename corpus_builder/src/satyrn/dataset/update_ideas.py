"""Generate testable Python evaluation and Reinforcement Learning datasets."""

import logging
from pathlib import Path

import click
from tqdm import tqdm

from satyrn.dataset.utils.generation import (
    IdeaVariant,
    append_dataset_line,
    prepare_output_file,
    read_jsonl_file,
)

logger = logging.getLogger(__name__)

PASS_MARKER = "__SATYRN_TEST_PASSED__"


@click.command("update-ideas")
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory of source material to draw from, or a single doc file.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file to write the generated dataset to.",
)
@click.option(
    "--idea-variant",
    "idea_variant",
    type=click.Choice(["code_demo", "use_case", "hard_use_case"]),
    required=True,
    help="Which type of idea; code_demo|use_case",
)
def main(
    input_path: Path,
    output_path: Path,
    idea_variant: IdeaVariant,
) -> None:
    """Generate a testable evaluation and Reinforcement Learning dataset."""

    prepare_output_file(output_path)
    lines = list(read_jsonl_file(input_path))
    for d in tqdm(lines):
        d["variant"] = idea_variant
        append_dataset_line(d, output_path)
