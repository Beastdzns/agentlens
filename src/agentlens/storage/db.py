"""
AgentLens — async SQLite database layer.

Architecture
------------
- One **write** connection, serialised behind an ``asyncio.Lock``.
- A small **read pool** (``asyncio.Queue``, N=4 connections).
- Per-connection PRAGMA configuration: WAL mode, NORMAL sync,
  5-second busy timeout, foreign-key enforcement.
- Migration runner: scans ``migrations/*.sql`` sorted alphabetically,
  applies unapplied ones in transactions, records in ``schema_migrations``.

Environment
-----------
``AGENTLENS_DB_PATH`` — path to the SQLite file (default: ``./agentlens.db``).

Usage::

    from agentlens.storage.db import init_db, get_database

    async def startup():
        await init_db()          # open connections + migrate
        db = get_database()      # singleton
        await db.execute_write("INSERT INTO ...", (values,))
        row = await db.fetch_one("SELECT ...", (values,))
        rows = await db.fetch_all("SELECT ...")
"""

from __future__ import annotations

import asyncio
import importlib.resources
import logging
import os
import pathlib
import re
from typing import Any, Optional, Sequence

import aiosqlite

logger = logging.getLogger(__name__)

_READ_POOL_SIZE = 4

# ── per-connection PRAGMA setup ───────────────────────────────────────────────


async def _configure_connection(conn: aiosqlite.Connection) -> None:
    """Apply recommended PRAGMA settings to a freshly opened connection."""
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = aiosqlite.Row


# ── Database class ─────────────────────────────────────────────────────────────


class Database:
    """
    Async SQLite wrapper with one serialised write connection
    and a pool of read connections.

    Do **not** instantiate directly; use :func:`init_db` and
    :func:`get_database`.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        self._write_conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()
        self._read_pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(
            maxsize=_READ_POOL_SIZE
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def _open(self) -> None:
        """Open write + read pool connections (called by :func:`init_db`)."""
        # Write connection
        self._write_conn = await aiosqlite.connect(self._path)
        await _configure_connection(self._write_conn)

        # Read pool
        for _ in range(_READ_POOL_SIZE):
            conn = await aiosqlite.connect(self._path)
            await _configure_connection(conn)
            await self._read_pool.put(conn)

        logger.info("Database opened at %s (WAL, pool=%d)", self._path, _READ_POOL_SIZE)

    async def close(self) -> None:
        """Close all connections.  Call during application shutdown."""
        if self._write_conn is not None:
            await self._write_conn.close()
            self._write_conn = None

        while not self._read_pool.empty():
            conn = self._read_pool.get_nowait()
            await conn.close()

        logger.info("Database closed.")

    # ── write API ─────────────────────────────────────────────────────────────

    async def execute_write(
        self, sql: str, params: Sequence[Any] = ()
    ) -> aiosqlite.Cursor:
        """
        Execute a single DML statement on the write connection.

        The write lock is held for the duration of the call; commits
        after every statement.

        Args:
            sql:    SQL statement string.
            params: Positional parameters for ``?`` placeholders.

        Returns:
            The :class:`aiosqlite.Cursor` (useful for ``lastrowid``).
        """
        assert self._write_conn is not None, "Database not initialised; call init_db() first"
        async with self._write_lock:
            cursor = await self._write_conn.execute(sql, params)
            await self._write_conn.commit()
            return cursor

    async def executemany_write(
        self, sql: str, params_seq: Sequence[Sequence[Any]]
    ) -> None:
        """
        Execute a DML statement for each row in *params_seq* inside one
        transaction (single commit).

        Args:
            sql:        SQL statement string.
            params_seq: Iterable of parameter tuples.
        """
        assert self._write_conn is not None, "Database not initialised; call init_db() first"
        async with self._write_lock:
            await self._write_conn.executemany(sql, params_seq)
            await self._write_conn.commit()

    # ── read API ─────────────────────────────────────────────────────────────

    async def fetch_one(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Optional[aiosqlite.Row]:
        """
        Fetch at most one row from the read pool.

        Args:
            sql:    SELECT statement.
            params: Positional parameters.

        Returns:
            An :class:`aiosqlite.Row` (dict-like), or ``None`` if no row.
        """
        conn = await self._read_pool.get()
        try:
            async with conn.execute(sql, params) as cursor:
                return await cursor.fetchone()
        finally:
            await self._read_pool.put(conn)

    async def fetch_all(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[aiosqlite.Row]:
        """
        Fetch all matching rows from the read pool.

        Args:
            sql:    SELECT statement.
            params: Positional parameters.

        Returns:
            List of :class:`aiosqlite.Row` objects (may be empty).
        """
        conn = await self._read_pool.get()
        try:
            async with conn.execute(sql, params) as cursor:
                return await cursor.fetchall()
        finally:
            await self._read_pool.put(conn)

    # ── internal write helpers (migration runner uses these) ─────────────────

    async def _execute_script(self, script: str) -> None:
        """Execute a raw SQL script (DDL) on the write connection."""
        assert self._write_conn is not None
        async with self._write_lock:
            await self._write_conn.executescript(script)
            await self._write_conn.commit()


# ── migration runner ──────────────────────────────────────────────────────────

def _migrations_dir() -> pathlib.Path:
    """Return the path to the bundled *migrations/* directory."""
    try:
        # Installed-package path via importlib.resources
        ref = importlib.resources.files("agentlens") / "migrations"
        return pathlib.Path(str(ref))
    except Exception:
        # Fall back to the path relative to this file (editable installs)
        return pathlib.Path(__file__).parent.parent / "migrations"


async def _ensure_schema_migrations(db: Database) -> None:
    """Create the schema_migrations bookkeeping table if it doesn't exist."""
    await db.execute_write(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )


async def _applied_versions(db: Database) -> set[int]:
    """Return the set of already-applied migration version numbers."""
    try:
        rows = await db.fetch_all("SELECT version FROM schema_migrations")
        return {row["version"] for row in rows}
    except Exception:
        return set()


async def _run_migrations(db: Database) -> None:
    """
    Scan the migrations directory, apply any unapplied ``.sql`` files
    in sorted (version) order, and record each in ``schema_migrations``.
    """
    await _ensure_schema_migrations(db)
    applied = await _applied_versions(db)

    mdir = _migrations_dir()
    if not mdir.exists():
        logger.warning("Migrations directory not found at %s; skipping.", mdir)
        return

    sql_files = sorted(mdir.glob("*.sql"))
    if not sql_files:
        logger.info("No migration files found in %s.", mdir)
        return

    version_re = re.compile(r"^(\d+)")

    for sql_file in sql_files:
        match = version_re.match(sql_file.name)
        if not match:
            logger.warning("Skipping migration file with no leading version number: %s", sql_file)
            continue

        version = int(match.group(1))
        if version in applied:
            logger.debug("Migration %04d already applied — skip.", version)
            continue

        logger.info("Applying migration %04d: %s", version, sql_file.name)
        script = sql_file.read_text(encoding="utf-8")

        # Run script + record in a single logical operation under the write lock
        async with db._write_lock:
            assert db._write_conn is not None
            await db._write_conn.executescript(script)
            from datetime import datetime, timezone

            now_iso = datetime.now(timezone.utc).isoformat()
            await db._write_conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, now_iso),
            )
            await db._write_conn.commit()

        logger.info("Migration %04d applied successfully.", version)


# ── singleton ──────────────────────────────────────────────────────────────────

_db_instance: Optional[Database] = None


async def init_db(path: Optional[str | os.PathLike[str]] = None) -> Database:
    """
    Initialise the global database singleton.

    Resolves the database path in the following order:

    1. The ``path`` argument (if provided).
    2. The ``AGENTLENS_DB_PATH`` environment variable.
    3. ``./agentlens.db`` (current-working-directory default).

    Opens the write + read-pool connections, then runs any pending migrations.

    Args:
        path: Optional explicit file path to the SQLite database.

    Returns:
        The initialised :class:`Database` instance.

    Example::

        await init_db()
        # or
        await init_db("/var/data/agentlens.db")
    """
    global _db_instance

    if _db_instance is not None:
        return _db_instance

    resolved_path = (
        path
        or os.environ.get("AGENTLENS_DB_PATH")
        or "./agentlens.db"
    )

    db = Database(resolved_path)
    await db._open()
    await _run_migrations(db)

    _db_instance = db
    return db


def get_database() -> Database:
    """
    Return the global :class:`Database` singleton.

    Raises:
        RuntimeError: If :func:`init_db` has not been called yet.
    """
    if _db_instance is None:
        raise RuntimeError(
            "Database not initialised. Call `await init_db()` before using get_database()."
        )
    return _db_instance
