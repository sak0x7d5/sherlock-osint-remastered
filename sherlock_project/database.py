from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite
from platformdirs import user_data_path

from sherlock_project.result import QueryStatus

DB_FILENAME = "sherlock.db"

MEMORY_DATABASE = ":memory:"


def default_database_path(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve where results are stored.

    The database follows the user, not the working directory. Extractions and
    synthesised profiles are cached in it and keyed by username, so a relative
    path would start a fresh, empty database every time the tool was invoked
    from a different directory -- silently discarding that cache and scattering
    scan data across the filesystem.

    `SHERLOCK_DB` overrides the location, mirroring `SHERLOCK_CONFIG` in
    ai_config.
    """
    environment = os.environ if environ is None else environ
    override = environment.get("SHERLOCK_DB")
    if override:
        return Path(override).expanduser()
    return user_data_path("sherlock", appauthor=False) / DB_FILENAME


@dataclass(frozen=True, slots=True)
class AIExtractionJob:
    site_id: int
    username: str
    site_name: str
    response_text: str


@dataclass(frozen=True, slots=True)
class AIProfileEvidenceRecord:
    site_id: int
    site_name: str
    site_url: str | None
    scanned_at: str | None
    ai_extraction: str | None
    ai_extraction_contract_hash: str | None


@dataclass(frozen=True, slots=True)
class ProfileSummaryCache:
    profile_summary: str | None
    input_hash: str | None
    updated_at: str | None


@dataclass(frozen=True, slots=True)
class StoredUsernameOverview:
    """What the database already holds for one username."""

    total_sites: int
    claimed_sites: int
    last_scanned_at: str | None


class SherlockDB:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self.db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @classmethod
    async def create(cls, database_path: str) -> SherlockDB:
        self = cls(database_path)
        await self.connect()
        return self

    async def connect(self) -> None:
        if self.db is not None:
            return

        # A user data directory does not exist until something writes to it,
        # and sqlite will not create intermediate directories itself.
        if self.database_path != MEMORY_DATABASE:
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

        connection = aiosqlite.connect(self.database_path)
        self.db = connection

        try:
            await connection
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            await self._initialize_tables()
        except BaseException as exc:
            try:
                await connection.rollback()
            except BaseException as cleanup_error:
                exc.add_note(
                    "Database rollback during connection cleanup also failed: "
                    f"{cleanup_error!r}"
                )

            try:
                await connection.close()
            except BaseException as cleanup_error:
                exc.add_note(
                    "Database close during connection cleanup also failed: "
                    f"{cleanup_error!r}"
                )
            finally:
                if self.db is connection:
                    self.db = None

            raise

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self.db is None:
            raise RuntimeError("Database is not connected")
        return self.db

    async def _initialize_tables(self) -> None:
        db = self._require_db()

        await db.execute("""
            CREATE TABLE IF NOT EXISTS usernames (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                profile_summary TEXT,
                profile_summary_input_hash TEXT,
                profile_summary_updated_at TIMESTAMP,
                last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                username_id INTEGER NOT NULL,
                site_name TEXT NOT NULL,
                site_url TEXT,
                status TEXT NOT NULL,
                status_code INTEGER,
                query_time_ms REAL,
                error_context TEXT,
                response_text TEXT,
                ai_extraction TEXT,
                ai_extraction_contract_hash TEXT,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username_id) REFERENCES usernames(id),
                UNIQUE(username_id, site_name)
            )
        """)

        await self._ensure_column(
            table_name="usernames",
            column_name="profile_summary_input_hash",
            definition="TEXT",
        )
        await self._ensure_column(
            table_name="usernames",
            column_name="profile_summary_updated_at",
            definition="TIMESTAMP",
        )
        await self._ensure_column(
            table_name="results",
            column_name="ai_extraction_contract_hash",
            definition="TEXT",
        )
        # How much of the site rule actually matched. Orthogonal to status:
        # status is what was decided, confidence is how much agreed. Persisted
        # so cross-site synthesis can weight a confirmed hit above a probable
        # one instead of treating every claimed result as equally true.
        await self._ensure_column(
            table_name="results",
            column_name="confidence",
            definition="TEXT",
        )

        await db.commit()

    async def _ensure_column(
        self,
        *,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        db = self._require_db()
        async with db.execute(f"PRAGMA table_info({table_name})") as cur:
            columns = await cur.fetchall()
        if any(row["name"] == column_name for row in columns):
            return
        await db.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
        )

    async def clear_tables(self) -> None:
        db = self._require_db()

        async with self._write_lock:
            try:
                await db.execute("DELETE FROM results")
                await db.execute("DELETE FROM usernames")
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def _ensure_username_id(self, username: str) -> int:
        db = self._require_db()

        await db.execute(
            "INSERT OR IGNORE INTO usernames (username) VALUES (?)",
            (username,),
        )

        async with db.execute(
            "SELECT id FROM usernames WHERE username = ?",
            (username,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            raise RuntimeError(f"Failed to get username id for {username!r}")

        return int(row["id"])

    async def get_or_create_username_id(self, username: str) -> int:
        db = self._require_db()

        async with self._write_lock:
            try:
                username_id = await self._ensure_username_id(username)
                await db.commit()
                return username_id

            except BaseException:
                await db.rollback()
                raise

    async def save_result(
        self,
        username: str,
        site_name: str,
        site_url: str | None = None,
        status: str = "UNKNOWN",
        status_code: int | None = None,
        query_time_ms: float | None = None,
        error_context: str | None = None,
        response_text: str | None = None,
        ai_extraction: str | None = None,
        ai_extraction_contract_hash: str | None = None,
        force_ai_extraction: bool = False,
        confidence: str | None = None,
    ) -> int:
        db = self._require_db()

        async with self._write_lock:
            try:
                username_id = await self._ensure_username_id(username)

                async with db.execute(
                    """
                    INSERT INTO results (
                        username_id,
                        site_name,
                        site_url,
                        status,
                        status_code,
                        query_time_ms,
                        error_context,
                        response_text,
                        ai_extraction,
                        ai_extraction_contract_hash,
                        confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(username_id, site_name) DO UPDATE SET
                        site_url = excluded.site_url,
                        status = excluded.status,
                        status_code = excluded.status_code,
                        query_time_ms = excluded.query_time_ms,
                        error_context = excluded.error_context,
                        response_text = excluded.response_text,
                        confidence = excluded.confidence,
                        ai_extraction = CASE
                            WHEN excluded.ai_extraction IS NOT NULL
                                THEN excluded.ai_extraction
                            WHEN ?
                                THEN NULL
                            WHEN excluded.status IS NOT results.status
                                OR excluded.response_text IS NOT results.response_text
                                THEN NULL
                            ELSE results.ai_extraction
                        END,
                        ai_extraction_contract_hash = CASE
                            WHEN excluded.ai_extraction IS NOT NULL
                                THEN excluded.ai_extraction_contract_hash
                            WHEN ?
                                THEN NULL
                            WHEN excluded.status IS NOT results.status
                                OR excluded.response_text IS NOT results.response_text
                                THEN NULL
                            ELSE results.ai_extraction_contract_hash
                        END,
                        scanned_at = CURRENT_TIMESTAMP
                        RETURNING id, ai_extraction
                    """,
                    (
                        username_id,
                        site_name,
                        site_url,
                        status,
                        status_code,
                        query_time_ms,
                        error_context,
                        response_text,
                        ai_extraction,
                        (
                            ai_extraction_contract_hash
                            if ai_extraction is not None
                            else None
                        ),
                        confidence,
                        force_ai_extraction,
                        force_ai_extraction,
                    ),
                ) as cur:
                    row = await cur.fetchone()

                if row is None:
                    raise RuntimeError(f"Failed to get site id for {site_name!r}")

                result_id = int(row["id"])
                if row["ai_extraction"] is None or ai_extraction is not None:
                    await self._invalidate_profile_summary(username_id)
                await db.commit()
                return result_id

            except BaseException:
                await db.rollback()
                raise

    async def get_pending_ai_extraction_ids(
        self,
        username: str,
        *,
        contract_hash: str,
    ) -> list[int]:
        db = self._require_db()

        async with db.execute(
            """
            SELECT r.id
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
                AND (
                    r.ai_extraction IS NULL
                    OR r.ai_extraction_contract_hash IS NOT ?
                )
                AND r.status = ?
                AND NULLIF(TRIM(r.response_text), '') IS NOT NULL
            ORDER BY r.id
            """,
            (username, contract_hash, str(QueryStatus.CLAIMED)),
        ) as cur:
            rows = await cur.fetchall()

        return [int(row["id"]) for row in rows]

    async def get_ai_extraction_job(
        self,
        site_id: int,
        *,
        contract_hash: str,
    ) -> AIExtractionJob | None:
        db = self._require_db()

        async with self._write_lock:
            try:
                async with db.execute(
                    """
                    SELECT
                        r.id,
                        r.username_id,
                        u.username,
                        r.site_name,
                        r.response_text,
                        r.ai_extraction_contract_hash
                    FROM results r
                    JOIN usernames u
                        ON u.id = r.username_id
                    WHERE r.id = ?
                        AND (
                            r.ai_extraction IS NULL
                            OR r.ai_extraction_contract_hash IS NOT ?
                        )
                        AND r.status = ?
                        AND NULLIF(TRIM(r.response_text), '') IS NOT NULL
                    """,
                    (site_id, contract_hash, str(QueryStatus.CLAIMED)),
                ) as cur:
                    row = await cur.fetchone()

                if row is None:
                    return None

                if row["ai_extraction_contract_hash"] != contract_hash:
                    await db.execute(
                        """
                        UPDATE results
                        SET
                            ai_extraction = NULL,
                            ai_extraction_contract_hash = NULL
                        WHERE id = ?
                        """,
                        (site_id,),
                    )
                    await self._invalidate_profile_summary(int(row["username_id"]))
                    await db.commit()

            except BaseException:
                await db.rollback()
                raise

        return AIExtractionJob(
            site_id=int(row["id"]),
            username=str(row["username"]),
            site_name=str(row["site_name"]),
            response_text=str(row["response_text"]),
        )

    async def update_result_ai_extraction(
        self,
        site_id: int,
        ai_extraction: str,
        *,
        contract_hash: str,
    ) -> None:
        db = self._require_db()

        async with self._write_lock:
            try:
                async with db.execute(
                    """
                    UPDATE results
                    SET
                        ai_extraction = ?,
                        ai_extraction_contract_hash = ?
                    WHERE id = ?
                    """,
                    (ai_extraction, contract_hash, site_id),
                ) as cur:
                    if cur.rowcount == 0:
                        raise RuntimeError(f"Failed to update AI extraction for site id {site_id!r}")

                await db.execute(
                    """
                    UPDATE usernames
                    SET
                        profile_summary_input_hash = NULL
                    WHERE id = (
                        SELECT username_id
                        FROM results
                        WHERE id = ?
                    )
                    """,
                    (site_id,),
                )
                await db.commit()

            except BaseException:
                await db.rollback()
                raise

    async def update_username_profile_summary(
        self,
        username: str,
        profile_summary: str,
        input_hash: str | None = None,
    ) -> None:
        db = self._require_db()

        async with self._write_lock:
            try:
                await db.execute(
                    """
                    INSERT INTO usernames (
                        username,
                        profile_summary,
                        profile_summary_input_hash,
                        profile_summary_updated_at
                    )
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(username) DO UPDATE SET
                        profile_summary = excluded.profile_summary,
                        profile_summary_input_hash = excluded.profile_summary_input_hash,
                        profile_summary_updated_at = CURRENT_TIMESTAMP
                    """,
                    (username, profile_summary, input_hash),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def get_ai_profile_evidence(
        self,
        username: str,
    ) -> list[AIProfileEvidenceRecord]:
        db = self._require_db()
        async with db.execute(
            """
            SELECT
                r.id,
                r.site_name,
                r.site_url,
                r.scanned_at,
                r.ai_extraction,
                r.ai_extraction_contract_hash
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
                AND r.status = ?
                AND NULLIF(TRIM(r.response_text), '') IS NOT NULL
            ORDER BY r.id
            """,
            (username, str(QueryStatus.CLAIMED)),
        ) as cur:
            rows = await cur.fetchall()

        return [
            AIProfileEvidenceRecord(
                site_id=int(row["id"]),
                site_name=str(row["site_name"]),
                site_url=str(row["site_url"]) if row["site_url"] is not None else None,
                scanned_at=(
                    str(row["scanned_at"])
                    if row["scanned_at"] is not None
                    else None
                ),
                ai_extraction=(
                    str(row["ai_extraction"])
                    if row["ai_extraction"] is not None
                    else None
                ),
                ai_extraction_contract_hash=(
                    str(row["ai_extraction_contract_hash"])
                    if row["ai_extraction_contract_hash"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    async def get_profile_summary_cache(
        self,
        username: str,
    ) -> ProfileSummaryCache | None:
        db = self._require_db()
        async with db.execute(
            """
            SELECT
                profile_summary,
                profile_summary_input_hash,
                profile_summary_updated_at
            FROM usernames
            WHERE username = ?
            """,
            (username,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return None
        return ProfileSummaryCache(
            profile_summary=(
                str(row["profile_summary"])
                if row["profile_summary"] is not None
                else None
            ),
            input_hash=(
                str(row["profile_summary_input_hash"])
                if row["profile_summary_input_hash"] is not None
                else None
            ),
            updated_at=(
                str(row["profile_summary_updated_at"])
                if row["profile_summary_updated_at"] is not None
                else None
            ),
        )

    async def invalidate_username_profile_summary(self, username: str) -> None:
        db = self._require_db()
        async with self._write_lock:
            try:
                await db.execute(
                    """
                    UPDATE usernames
                    SET profile_summary_input_hash = NULL
                    WHERE username = ?
                    """,
                    (username,),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise

    async def _invalidate_profile_summary(self, username_id: int) -> None:
        db = self._require_db()
        await db.execute(
            """
            UPDATE usernames
            SET
                profile_summary_input_hash = NULL
            WHERE id = ?
            """,
            (username_id,),
        )

    async def get_username_by_id(self, username_id: int) -> str | None:
        db = self._require_db()

        async with db.execute(
            "SELECT username FROM usernames WHERE id = ?",
            (username_id,),
        ) as cur:
            row = await cur.fetchone()

        return row["username"] if row else None

    async def get_username_overview(
        self,
        username: str,
    ) -> StoredUsernameOverview | None:
        """Summarise stored results, or None when the username is unknown.

        `last_scanned_at` comes from the results rather than
        `usernames.last_scanned_at`, which is written once on insert and never
        updated — it records when the username was first seen, so reporting it
        as the scan date would be wrong.
        """
        db = self._require_db()

        async with db.execute(
            """
            SELECT
                COUNT(*) AS total_sites,
                SUM(CASE WHEN r.status = ? THEN 1 ELSE 0 END) AS claimed_sites,
                MAX(r.scanned_at) AS last_scanned_at
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
            """,
            (str(QueryStatus.CLAIMED), username),
        ) as cur:
            row = await cur.fetchone()

        if row is None or not row["total_sites"]:
            return None

        return StoredUsernameOverview(
            total_sites=int(row["total_sites"]),
            claimed_sites=int(row["claimed_sites"] or 0),
            last_scanned_at=row["last_scanned_at"],
        )

    async def get_saved_results(self, username: str) -> dict[str, dict[str, Any]]:
        """Return stored results for a username, keyed by site name.

        The scan uses the keys to decide what to skip and the rows to rebuild a
        report covering the skipped sites, so both come from one query.

        Deliberately does not select response_text. The column holds whole
        pages, and the only consumers of this data are the terminal report and
        the exports, neither of which reads it. Loading it would mean pulling
        several hundred documents off disk to display a list of URLs.
        """
        db = self._require_db()

        async with db.execute(
            """
            SELECT
                r.site_name,
                r.site_url,
                r.status,
                r.status_code,
                r.query_time_ms,
                r.error_context,
                r.confidence,
                r.scanned_at
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
            """,
            (username,),
        ) as cur:
            rows = await cur.fetchall()

        return {row["site_name"]: dict(row) for row in rows}
