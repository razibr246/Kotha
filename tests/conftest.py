"""Tests run against a real Postgres + pgvector.

Set TEST_DATABASE_URL (or DATABASE_URL) to a throwaway database; every session drops and
re-seeds it. The clock is pinned to Thu 15 Oct 2026, 10:00 in Dhaka.
"""
import os
from datetime import datetime

import pytest

os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or os.environ.get(
    "DATABASE_URL", "postgresql://kotha:kotha@localhost:5432/kotha_test")
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["COURIER_API_URL"] = ""
os.environ["KOTHA_EMBEDDER"] = "hash"

from kotha import timefmt  # noqa: E402
from kotha.config import settings  # noqa: E402

NOW = datetime(2026, 10, 15, 10, 0, tzinfo=settings.timezone)


@pytest.fixture(scope="session", autouse=True)
def seeded():
    from kotha.db import drop_all, init_schema
    from kotha.embeddings import get_embedder
    from kotha.seed import seed_demo

    drop_all()
    init_schema(get_embedder().dim)
    seed_demo("demo", now=NOW)
    yield


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    monkeypatch.setattr(timefmt, "_now", lambda: NOW)
    return NOW
