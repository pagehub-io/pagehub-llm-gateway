"""Asyncpg pool + schema bootstrap. Mirrors the pagehub-evals pattern."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_ADVISORY_LOCK_KEY = 0xE5_4A_16_00  # pagehub-llm-gateway namespace; distinct from other services


async def init_pool(database_url: str) -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool

    async def _set_session_params(conn: asyncpg.Connection) -> None:
        await conn.execute("SET statement_timeout = '30s'")
        await conn.execute("SET lock_timeout = '5s'")

    _pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=10,
        command_timeout=30,
        statement_cache_size=0,
        init=_set_session_params,
    )
    logger.info("asyncpg pool initialised")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("asyncpg pool not initialised — did lifespan run?")
    return _pool


async def apply_schema() -> None:
    if not _SCHEMA_PATH.exists():
        logger.warning("schema.sql not found at %s — skipping", _SCHEMA_PATH)
        return
    sql = _SCHEMA_PATH.read_text()
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", _ADVISORY_LOCK_KEY)
            await conn.execute(sql)
    logger.info("schema applied")
