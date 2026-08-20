"""Sandboxed Python execution.

"Sandboxed" here means: a separate subprocess, cwd jailed to WORKDIR, a wall
clock timeout, and (on POSIX) CPU-time/memory rlimits applied before exec.
It does NOT mean network-isolated or filesystem-isolated the way a
container/gVisor sandbox would be -- a determined script could still open
a socket or read absolute paths. For a portfolio MVP this is a reasonable
tradeoff; call it out explicitly rather than overclaiming security. A real
deployment should run this in a container with no network egress.
"""
from __future__ import annotations

import subprocess
import sys

from pydantic import BaseModel, Field

from orchestration import config
from orchestration.tools.registry import ToolRegistry

TIMEOUT_SECONDS = 15
CPU_SECONDS_LIMIT = 10
MEMORY_BYTES_LIMIT = 512 * 1024 * 1024


def _limit_resources():  # pragma: no cover - exercised only in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS_LIMIT, CPU_SECONDS_LIMIT))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES_LIMIT, MEMORY_BYTES_LIMIT))
    except Exception:
        pass  # best-effort; not available on all platforms (e.g. Windows)


class CodeExecInput(BaseModel):
    code: str = Field(description="Python source to execute")


class CodeExecOutput(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def run(input_data: CodeExecInput) -> CodeExecOutput:
    sandbox_dir = config.settings.resolved_workdir()
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if sys.platform != "win32":
        kwargs["preexec_fn"] = _limit_resources

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", input_data.code],
            cwd=str(sandbox_dir),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            **kwargs,
        )
        stderr = proc.stderr
        if proc.returncode == 0 and not proc.stdout.strip() and not stderr.strip():
            stderr = (
                "[sandbox note] The script exited successfully but printed nothing. If you expected "
                "to see a result, add an explicit print(...) call -- this runs as a plain script, not "
                "a REPL, so a bare expression on the last line is silently discarded."
            )
        return CodeExecOutput(stdout=proc.stdout, stderr=stderr, exit_code=proc.returncode, timed_out=False)
    except subprocess.TimeoutExpired as exc:
        return CodeExecOutput(
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=f"Execution timed out after {TIMEOUT_SECONDS}s",
            exit_code=-1,
            timed_out=True,
        )


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="code_exec",
        description=(
            "Execute a short Python snippet in a sandboxed subprocess and return stdout/stderr. "
            "This runs as a plain script (`python -c`), NOT a REPL or notebook: a bare expression "
            "or variable name on its own line produces NO output. You MUST call print(...) on any "
            "value you want to see in the result -- e.g. print(compound_interest), not just "
            "compound_interest on its own line."
        ),
        input_schema=CodeExecInput,
        output_schema=CodeExecOutput,
        allowed_agents=["data_analysis", "code_exec"],
        rate_limit_per_minute=15,
        func=run,
    )
