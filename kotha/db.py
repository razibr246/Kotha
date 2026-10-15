"""Postgres connection pool and schema."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row, "autocommit": False},
            configure=_configure,
            open=True,
        )
    return _pool


@atexit.register
def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def schema_sql(dim: int) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS shops (
    id            text PRIMARY KEY,
    name          text NOT NULL,
    support_phone text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS orders (
    shop_id        text NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    order_no       integer NOT NULL,
    customer_name  text NOT NULL,
    customer_phone text NOT NULL,
    delivery_area  text NOT NULL,
    items          jsonb NOT NULL,
    total_bdt      numeric(12,2) NOT NULL,
    payment_method text NOT NULL,      -- cod | bkash | nagad | card
    payment_status text NOT NULL,      -- unpaid | paid | refunded
    status         text NOT NULL,      -- pending | confirmed | packed | shipped | delivered | cancelled | returned
    courier        text,
    tracking_id    text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (shop_id, order_no)
);

CREATE TABLE IF NOT EXISTS help_chunks (
    id         bigserial PRIMARY KEY,
    shop_id    text NOT NULL REFERENCES shops(id) ON DELETE CASCADE,
    doc        text NOT NULL,
    section    text NOT NULL,
    lang       text NOT NULL,          -- bn | en
    pos        integer NOT NULL,       -- chunk order within the doc; parallel across languages
    content    text NOT NULL,
    embedding  vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS help_chunks_shop_idx ON help_chunks (shop_id);
CREATE INDEX IF NOT EXISTS help_chunks_hnsw ON help_chunks USING hnsw (embedding vector_cosine_ops);

-- Backing store for the built-in mock courier. A real courier replaces this table.
CREATE TABLE IF NOT EXISTS courier_shipments (
    tracking_id      text PRIMARY KEY,
    courier          text NOT NULL,
    destination_area text NOT NULL,
    shipped_at       timestamptz NOT NULL,
    eta              timestamptz NOT NULL,
    hub              text NOT NULL,
    exception        text            -- e.g. 'customer_unreachable'
);

CREATE TABLE IF NOT EXISTS chat_log (
    id         bigserial PRIMARY KEY,
    shop_id    text NOT NULL,
    lang       text NOT NULL,
    question   text NOT NULL,
    answer     text NOT NULL,
    sources    jsonb NOT NULL,
    engine     text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
"""


def init_schema(dim: int) -> None:
    with connection() as conn:
        existing = conn.execute(
            """SELECT atttypmod FROM pg_attribute
               WHERE attrelid = to_regclass('help_chunks') AND attname = 'embedding'"""
        ).fetchone()
        if existing and existing["atttypmod"] != dim:
            raise RuntimeError(
                f"help_chunks.embedding has dimension {existing['atttypmod']} but the "
                f"configured embedder produces {dim}. Re-seed with --reset."
            )
        conn.execute(schema_sql(dim))
        conn.commit()


def drop_all() -> None:
    with connection() as conn:
        conn.execute(
            "DROP TABLE IF EXISTS chat_log, courier_shipments, help_chunks, orders, shops CASCADE"
        )
        conn.commit()
