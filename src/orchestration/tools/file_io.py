"""Sandboxed file read/write. All paths are resolved relative to WORKDIR and
verified to stay inside it, closing the obvious path-traversal hole
(`../../etc/passwd`) before the tool ever touches the filesystem.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from orchestration import config
from orchestration.tools.registry import ToolRegistry


def _resolve_in_sandbox(relative_path: str) -> Path:
    root = config.settings.resolved_workdir()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise PermissionError(f"Path '{relative_path}' escapes the sandboxed workdir")
    return candidate


class FileReadInput(BaseModel):
    path: str = Field(description="Path relative to the sandboxed working directory")


class FileReadOutput(BaseModel):
    content: str
    bytes_read: int


def read(input_data: FileReadInput) -> FileReadOutput:
    path = _resolve_in_sandbox(input_data.path)
    if not path.exists():
        raise FileNotFoundError(f"'{input_data.path}' does not exist in the sandbox")
    content = path.read_text(errors="replace")
    return FileReadOutput(content=content, bytes_read=len(content.encode()))


class FileWriteInput(BaseModel):
    path: str = Field(description="Path relative to the sandboxed working directory")
    content: str


class FileWriteOutput(BaseModel):
    path: str
    bytes_written: int


def write(input_data: FileWriteInput) -> FileWriteOutput:
    path = _resolve_in_sandbox(input_data.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(input_data.content)
    return FileWriteOutput(path=input_data.path, bytes_written=len(input_data.content.encode()))


def register(registry: ToolRegistry) -> None:
    registry.register(
        name="file_read",
        description="Read a text file from the sandboxed working directory.",
        input_schema=FileReadInput,
        output_schema=FileReadOutput,
        allowed_agents=["research", "data_analysis", "writer", "code_exec"],
        rate_limit_per_minute=60,
        func=read,
    )
    registry.register(
        name="file_write",
        description="Write a text file into the sandboxed working directory.",
        input_schema=FileWriteInput,
        output_schema=FileWriteOutput,
        allowed_agents=["research", "data_analysis", "writer", "code_exec"],
        rate_limit_per_minute=60,
        func=write,
    )
