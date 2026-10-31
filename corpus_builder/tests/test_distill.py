"""Tests for the distill command, with a fake vLLM client and fake sandboxes."""

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from satyrn.dataset import distill
from satyrn.dataset.distill import DistillSettings, build_prompt, count_written_rows, distill_row, summarize
from satyrn.dataset.llm import models
from satyrn.dataset.llm.models import Completion, VllmClient
from satyrn.dataset.utils.sandbox import RunResult, Sandbox

GOOD_CODE = "print('new')"
GOOD_CONTENT = f"Here is the new way.\n\n```python\n{GOOD_CODE}\n```"
GOOD_REASONING = "I should use the new feature."

EXPECTED_PROMPT = """Below are an instruction that describes a task along with a reference answer. \
Using the reference answer as a guide, write your own response.
### Instruction:
How do I do it?
### Reference Answer:
Like this.

```python
print('old')
```
### Response:"""

PASSED = RunResult("", exit_code=0, timed_out=False)


def make_row(index: int = 0, python_version: str = "3.15") -> dict:
    return {
        "prompt": [{"role": "user", "content": "How do I do it?"}],
        "completion": [{"role": "assistant", "content": "Like this.\n\n```python\nprint('old')\n```"}],
        "filename": f"doc-{index}.rst",
        "pep": None,
        "python_version": python_version,
        "idea": "An idea",
        "code": "print('old')",
        "trace": "The old trace.",
        "expected_output": "old\n",
    }


def make_completion(**overrides: Any) -> Completion:
    fields = {
        "content": GOOD_CONTENT,
        "reasoning": GOOD_REASONING,
        "finish_reason": "stop",
        "completion_tokens": 100,
    }
    return Completion(**{**fields, **overrides})


class FakeClient:
    """Serve canned completions in order, repeating the last one."""

    def __init__(self, *completions: Completion, instruction_tokens: int = 10) -> None:
        self.completions = completions or (make_completion(),)
        self.instruction_tokens = instruction_tokens
        self.prompts: list[str] = []

    def complete(self, prompt: str, thinking: bool, max_tokens: int) -> Completion:
        self.prompts.append(prompt)
        return self.completions[min(len(self.prompts) - 1, len(self.completions) - 1)]

    def count_tokens(self, text: str) -> int:
        return self.instruction_tokens


class FakeSandbox:
    """Pass the syntax check, then return the canned result of running the code."""

    def __init__(self, run: RunResult, syntax: RunResult = PASSED) -> None:
        self.run_result = run
        self.syntax = syntax

    def execute(self, code: str) -> RunResult:
        return self.syntax if code.startswith("compile(") else self.run_result


def target(output: str = "new\n", exit_code: int | None = 0, syntax: RunResult = PASSED) -> FakeSandbox:
    return FakeSandbox(RunResult(output, exit_code, timed_out=exit_code is None), syntax)


def predecessor(exit_code: int | None = 1) -> FakeSandbox:
    return FakeSandbox(RunResult("Traceback", exit_code, timed_out=exit_code is None))


def settings(**overrides: Any) -> DistillSettings:
    fields = {
        "thinking": True,
        "max_tokens": 1280,
        "max_seq_len": 1280,
        "max_attempts": 1,
        "match_expected_output": False,
    }
    return DistillSettings(**{**fields, **overrides})


def run_row(
    client: FakeClient | None = None,
    sandbox: FakeSandbox | None = None,
    predecessor_sandbox: FakeSandbox | None = None,
    **setting_overrides: Any,
) -> dict:
    sandboxes = {"3.15": (sandbox or target(), predecessor_sandbox or predecessor())}
    return distill_row(make_row(), client or FakeClient(), sandboxes, settings(**setting_overrides))


def test_prompt_matches_the_template() -> None:
    assert build_prompt(make_row()) == EXPECTED_PROMPT


def test_prompt_handles_braces_in_the_row() -> None:
    row = make_row()
    row["prompt"][0]["content"] = "Why {does} this {0} fail?"

    assert "Why {does} this {0} fail?" in build_prompt(row)


def test_accepted_row_replaces_its_fields_together() -> None:
    row = run_row(sandbox=target(output="actual\n"))

    assert row["completion"] == [{"role": "assistant", "content": GOOD_CONTENT}]
    assert row["trace"] == GOOD_REASONING
    assert row["code"] == GOOD_CODE
    assert row["expected_output"] == "actual\n"
    assert row["distilled"] is True
    assert row["distill_rejection"] is None
    assert row["idea"] == "An idea"


def test_trace_is_left_as_is_without_thinking() -> None:
    row = run_row(thinking=False, client=FakeClient(make_completion(reasoning="")))

    assert row["distilled"] is True
    assert row["trace"] == "The old trace."


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"client": FakeClient(make_completion(finish_reason="length"))}, "truncated"),
        ({"client": FakeClient(make_completion(content="  \n"))}, "empty"),
        ({"client": FakeClient(make_completion(reasoning=""))}, "empty"),
        ({"client": FakeClient(make_completion(completion_tokens=1271))}, "too_long"),
        ({"client": FakeClient(instruction_tokens=1181)}, "too_long"),
        ({"client": FakeClient(make_completion(content=f"The Reference Answer says:\n{GOOD_CONTENT}"))}, "leak"),
        ({"client": FakeClient(make_completion(content=f"The reference response says:\n{GOOD_CONTENT}"))}, "leak"),
        ({"client": FakeClient(make_completion(reasoning="Per the reference answer..."))}, "leak"),
        ({"client": FakeClient(make_completion(content="No code at all."))}, "code_blocks"),
        ({"client": FakeClient(make_completion(content=f"{GOOD_CONTENT}\n\n{GOOD_CONTENT}"))}, "code_blocks"),
        ({"sandbox": target(syntax=RunResult("SyntaxError", 1, False))}, "syntax"),
        ({"sandbox": target(exit_code=1)}, "runtime"),
        ({"sandbox": target(exit_code=None)}, "runtime"),
        ({"predecessor_sandbox": predecessor(exit_code=0)}, "no_new_feature"),
        ({"predecessor_sandbox": predecessor(exit_code=None)}, "no_new_feature"),
        ({"predecessor_sandbox": predecessor(exit_code=125)}, "no_new_feature"),
        ({"predecessor_sandbox": predecessor(exit_code=137)}, "no_new_feature"),
        ({"sandbox": target(output="different\n"), "match_expected_output": True}, "output_mismatch"),
    ],
)
def test_rejected_row_is_otherwise_unchanged(kwargs: dict, reason: str) -> None:
    row = run_row(**kwargs)

    assert row == {**make_row(), "distilled": False, "distill_rejection": reason}


def test_row_exactly_at_the_length_limit_is_accepted() -> None:
    assert run_row(client=FakeClient(make_completion(completion_tokens=1270)))["distilled"] is True


def test_other_fenced_blocks_are_allowed_beside_the_python_block() -> None:
    content = f"{GOOD_CONTENT}\n\nIt prints:\n\n```text\nnew\n```"

    row = run_row(client=FakeClient(make_completion(content=content)))

    assert row["distilled"] is True
    assert row["code"] == GOOD_CODE


def test_multiline_code_is_extracted_whole() -> None:
    code = "def f():\n    return 1\n\n\nprint(f())"

    row = run_row(client=FakeClient(make_completion(content=f"Intro\n\n```python\n{code}\n```\n")))

    assert row["code"] == code


def test_output_may_differ_from_the_original_by_default() -> None:
    assert run_row(sandbox=target(output="different\n"))["distilled"] is True


def test_match_expected_output_accepts_the_same_output() -> None:
    assert run_row(sandbox=target(output="old\n"), match_expected_output=True)["distilled"] is True


def test_syntax_is_checked_by_compiling_the_code_in_the_sandbox() -> None:
    """The real check, on the host's interpreter in place of a sandbox's."""
    calls: list[str] = []

    class CompilingSandbox:
        def execute(self, code: str) -> RunResult:
            calls.append(code)
            try:
                exec(code, {})
            except SyntaxError as error:
                return RunResult(str(error), 1, False)
            return PASSED

    assert distill.check_syntax("print('a\"b')", CompilingSandbox())
    assert not distill.check_syntax("def broken(:", CompilingSandbox())
    assert len(calls) == 2


def test_max_attempts_resamples_the_same_prompt_until_accepted() -> None:
    client = FakeClient(make_completion(finish_reason="length"), make_completion())

    row = run_row(client=client, max_attempts=3)

    assert row["distilled"] is True
    assert len(client.prompts) == 2
    assert client.prompts[0] == client.prompts[1]


def test_max_attempts_stops_at_the_limit_with_the_last_reason() -> None:
    client = FakeClient(
        make_completion(finish_reason="length"), make_completion(content="No code."), make_completion(content="")
    )

    row = run_row(client=client, max_attempts=2)

    assert row["distill_rejection"] == "code_blocks"
    assert len(client.prompts) == 2


def test_each_row_uses_the_sandboxes_of_its_python_version() -> None:
    sandboxes = {
        "3.14": (target(output="from 3.14\n"), predecessor()),
        "3.15": (target(output="from 3.15\n"), predecessor()),
    }

    row = distill_row(make_row(python_version="3.14"), FakeClient(), sandboxes, settings())

    assert row["expected_output"] == "from 3.14\n"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def invoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[Any, list[dict]]]:
    """Run the command against fake backends, with gVisor reported as available."""
    monkeypatch.setattr(distill, "docker_available", lambda: True)
    monkeypatch.setattr(distill, "gvisor_available", lambda: True)
    monkeypatch.setattr(distill, "remove_leftover_containers", lambda: 0)
    monkeypatch.setattr(distill, "Sandbox", lambda version: target() if version == "3.15" else predecessor())
    client = FakeClient()
    monkeypatch.setattr(distill, "VllmClient", lambda model_name: client)
    output_path = tmp_path / "distilled" / "out.jsonl"

    def run(rows: list[dict], *args: str, output: Path = output_path) -> tuple[Any, list[dict]]:
        input_path = tmp_path / "in.jsonl"
        write_rows(input_path, rows)
        result = CliRunner().invoke(
            distill.main, ["-i", str(input_path), "-o", str(output), "--thinking", *args], catch_exceptions=False
        )
        return result, read_rows(output) if output.exists() else []

    run.client = client  # type: ignore[attr-defined]
    run.output_path = output_path  # type: ignore[attr-defined]
    return run


def test_output_order_matches_the_input(invoke: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_first_row(row: dict, *args: Any, **kwargs: Any) -> dict:
        if row["filename"] == "doc-0.rst":
            time.sleep(0.2)
        return {**row, "distilled": True, "distill_rejection": None}

    monkeypatch.setattr(distill, "distill_row", slow_first_row)

    _, written = invoke([make_row(index) for index in range(4)], "--workers", "4")

    assert [row["filename"] for row in written] == [f"doc-{index}.rst" for index in range(4)]


def test_all_rows_have_equal_keys(invoke: Any) -> None:
    invoke.client.completions = (make_completion(), make_completion(finish_reason="length"))

    _, written = invoke([make_row(0), make_row(1)])

    assert [row["distilled"] for row in written] == [True, False]
    assert list(written[0]) == list(written[1])


def test_limit_only_distills_the_first_rows(invoke: Any) -> None:
    _, written = invoke([make_row(index) for index in range(5)], "--limit", "2")

    assert len(written) == 2


def test_restart_skips_rows_already_written(invoke: Any) -> None:
    rows = [make_row(index) for index in range(3)]
    invoke(rows, "--limit", "1")
    assert len(invoke.client.prompts) == 1

    _, written = invoke(rows)

    assert len(invoke.client.prompts) == 3
    assert [row["filename"] for row in written] == [row["filename"] for row in rows]


def test_restart_of_a_finished_run_does_nothing(invoke: Any) -> None:
    invoke([make_row()])

    result, written = invoke([make_row()])

    assert result.exit_code == 0
    assert len(invoke.client.prompts) == 1
    assert len(written) == 1


def test_output_with_more_rows_than_the_input_is_refused(invoke: Any) -> None:
    invoke([make_row(0), make_row(1)])

    result, _ = invoke([make_row(0)])

    assert result.exit_code == 1
    assert "more than the 1 to distill" in result.output


def test_partly_written_output_fails_rather_than_resuming(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"distilled": true}\n{"disti')

    with pytest.raises(json.JSONDecodeError):
        count_written_rows(output_path)


def test_server_failure_stops_the_run_and_keeps_written_rows(invoke: Any) -> None:
    def fail_on_second_call(prompt: str, thinking: bool, max_tokens: int) -> Completion:
        invoke.client.prompts.append(prompt)
        if len(invoke.client.prompts) > 1:
            raise ConnectionError("server gone")
        return make_completion()

    invoke.client.complete = fail_on_second_call

    with pytest.raises(ConnectionError):
        invoke([make_row(0), make_row(1), make_row(2)])

    assert len(read_rows(invoke.output_path)) == 1


def test_output_must_be_jsonl(invoke: Any, tmp_path: Path) -> None:
    result, _ = invoke([make_row()], output=tmp_path / "out.json")

    assert result.exit_code == 2
    assert ".jsonl" in result.output


def test_startup_fails_without_gvisor(invoke: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distill, "gvisor_available", lambda: False)

    result, written = invoke([make_row()])

    assert result.exit_code == 1
    assert "gVisor" in result.output
    assert written == []
    assert invoke.client.prompts == []


def test_allow_unsandboxed_runs_without_gvisor(invoke: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(distill, "gvisor_available", lambda: False)

    result, written = invoke([make_row()], "--allow-unsandboxed")

    assert result.exit_code == 0
    assert len(written) == 1


def test_report_counts_rows_from_earlier_runs(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    rejected = {"distilled": False}
    write_rows(
        output_path,
        [
            {"distilled": True, "distill_rejection": None},
            {"distilled": True, "distill_rejection": None},
            {**rejected, "distill_rejection": "leak"},
            {**rejected, "distill_rejection": "syntax"},
            {**rejected, "distill_rejection": "syntax"},
        ],
    )

    assert summarize(output_path).splitlines() == [
        "Distilled 2 of 5 rows",
        "Kept the original for 3 rows",
        "  syntax: 2",
        "  leak: 1",
    ]


def test_report_includes_a_resumed_run(invoke: Any, caplog: pytest.LogCaptureFixture) -> None:
    rows = [make_row(index) for index in range(3)]
    invoke(rows, "--limit", "2")

    with caplog.at_level("INFO"):
        invoke(rows)

    assert "Distilled 3 of 3 rows" in caplog.text


def stub_response(message: SimpleNamespace, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=42),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> VllmClient:
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.test:8000/")
    return VllmClient("seed")


def stub_completions(client: VllmClient, message: SimpleNamespace) -> list[dict]:
    calls: list[dict] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return stub_response(message)

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))  # type: ignore[assignment]
    return calls


def test_vllm_client_points_at_the_openai_routes(client: VllmClient) -> None:
    assert str(client.client.base_url) == "http://vllm.test:8000/v1/"


@pytest.mark.parametrize("thinking", [True, False])
def test_complete_sends_only_a_user_message(client: VllmClient, thinking: bool) -> None:
    calls = stub_completions(client, SimpleNamespace(content="hello", reasoning=None))

    client.complete("prompt", thinking=thinking, max_tokens=7)

    assert calls == [
        {
            "model": "seed",
            "messages": [{"role": "user", "content": "prompt"}],
            "max_tokens": 7,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        }
    ]


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_complete_reads_either_reasoning_field(client: VllmClient, field: str) -> None:
    stub_completions(client, SimpleNamespace(content="hello", **{field: "why"}))

    completion = client.complete("prompt", thinking=True, max_tokens=7)

    assert completion == Completion("hello", "why", "stop", 42)


def test_complete_turns_missing_fields_into_empty_strings(client: VllmClient) -> None:
    stub_completions(client, SimpleNamespace(content=None))

    completion = client.complete("prompt", thinking=True, max_tokens=7)

    assert (completion.content, completion.reasoning) == ("", "")


def test_vllm_base_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    with pytest.raises(KeyError):
        models.VllmClient("seed")


def make_sandbox(monkeypatch: pytest.MonkeyPatch, result: Any) -> Sandbox:
    """Build a Sandbox whose docker call returns result, or raises it if it is an exception."""

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("satyrn.dataset.utils.sandbox.subprocess.run", fake_run)
    sandbox = Sandbox.__new__(Sandbox)
    sandbox.python_version = "3.15"
    sandbox.image = "image"
    sandbox.use_gvisor = False
    return sandbox


def test_execute_reports_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = make_sandbox(monkeypatch, SimpleNamespace(stdout="boom\n", returncode=1))

    assert sandbox.execute("code") == RunResult("boom\n", exit_code=1, timed_out=False)


def test_execute_reports_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = make_sandbox(monkeypatch, subprocess.TimeoutExpired("docker", 20))

    result = sandbox.execute("code")

    assert result.timed_out
    assert result.exit_code is None


def test_run_still_returns_the_output(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = make_sandbox(monkeypatch, SimpleNamespace(stdout="hello\n", returncode=0))

    assert sandbox.run("code") == "hello\n"


def test_run_still_returns_the_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox = make_sandbox(monkeypatch, subprocess.TimeoutExpired("docker", 20))

    assert sandbox.run("code") == "[did not terminate within 20 seconds]"
