"""Central config, loaded once from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return default if val is None else val.lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("PROVIDER", "openai"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    supervisor_model: str = field(default_factory=lambda: os.getenv("SUPERVISOR_MODEL", "gpt-5-mini"))
    specialist_model: str = field(default_factory=lambda: os.getenv("SPECIALIST_MODEL", "gpt-5-nano"))
    reviewer_model: str = field(default_factory=lambda: os.getenv("REVIEWER_MODEL", "gpt-5-nano"))

    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6")))
    review_score_threshold: float = field(default_factory=lambda: float(os.getenv("REVIEW_SCORE_THRESHOLD", "0.6")))
    max_specialist_retries: int = field(default_factory=lambda: int(os.getenv("MAX_SPECIALIST_RETRIES", "2")))

    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "data/orchestration.db"))
    checkpoint_path: str = field(default_factory=lambda: os.getenv("CHECKPOINT_PATH", "data/checkpoints.db"))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "data/chroma"))
    workdir: str = field(default_factory=lambda: os.getenv("WORKDIR", "workdir"))

    def resolved_sqlite_path(self) -> Path:
        p = Path(self.sqlite_path)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_checkpoint_path(self) -> Path:
        p = Path(self.checkpoint_path)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_chroma_path(self) -> Path:
        p = Path(self.chroma_path)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_workdir(self) -> Path:
        p = Path(self.workdir)
        return p if p.is_absolute() else REPO_ROOT / p


settings = Settings()
