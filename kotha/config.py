"""Runtime settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://kotha:kotha@localhost:5432/kotha"
        )
    )
    # LLM. Without a key Kotha runs a deterministic offline agent.
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    model: str = field(default_factory=lambda: os.getenv("KOTHA_MODEL", "claude-sonnet-5-5"))
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("KOTHA_MAX_TOOL_ROUNDS", "5")))

    # Embeddings: "hash" (local, no network) or "voyage" (multilingual, cross-lingual retrieval).
    embedder: str = field(default_factory=lambda: os.getenv("KOTHA_EMBEDDER", "hash"))
    voyage_api_key: str | None = field(default_factory=lambda: os.getenv("VOYAGE_API_KEY") or None)
    voyage_model: str = field(default_factory=lambda: os.getenv("VOYAGE_MODEL", "voyage-multilingual-2"))

    # Courier: leave unset to use the built-in mock courier backed by Postgres.
    courier_api_url: str | None = field(default_factory=lambda: os.getenv("COURIER_API_URL") or None)
    courier_api_key: str | None = field(default_factory=lambda: os.getenv("COURIER_API_KEY") or None)

    default_shop: str = field(default_factory=lambda: os.getenv("KOTHA_DEFAULT_SHOP", "demo"))
    timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo(os.getenv("KOTHA_TZ", "Asia/Dhaka")))


settings = Settings()
