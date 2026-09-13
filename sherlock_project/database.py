from __future__ import annotations

import asyncio
import os
import sqlite3
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


@dataclass(frozen=True, slots=True)
class StoredUsernameListing:
    """One line of "what is in this database", for a picker.

    Carries `has_profile` rather than the profile itself: the list exists to be
    chosen from, and pass-two summaries are large enough that loading every one
    of them to draw a sidebar would read the whole table off disk to show a
    column of names.
    """

    username: str
    total_sites: int
    claimed_sites: int
    last_scanned_at: str | None
    has_profile: bool


# How hard to wait out a lock sqlite refuses to wait out itself. Two rounds of
# backoff, so ~1.1s in total before giving up rather than ~0.2s.
#
# The first budget was derived on Linux, where 8 connections creating one
# database at once measured 0 failures in 40 rounds. Windows failed it: NTFS
# takes mandatory locks and its file operations are slower, so the losers of
# the race need longer than Linux losers do. Retrying costs nothing when there
# is no contention -- the loop exits on the first attempt -- so the budget is
# sized for the slowest platform rather than the fastest.
_LOCK_RETRY_ATTEMPTS = 8
_LOCK_RETRY_BASE_SECONDS = 0.03


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
            # More than one connection to this file is open at a time -- the
            # results pane loads its listing on one while a delete writes on
            # another, and a scan writes while the pane reads. Under the
            # default rollback journal a reader blocks a writer's COMMIT, and
            # with no busy timeout sqlite does not wait: it raises
            # "database is locked" immediately. That is not theoretical, it is
            # `delete_username` failing mid-transaction with the usernames row
            # already deleted and the results rows not.
            #
            # WAL is the fix rather than a longer timeout, because a timeout
            # only converts the error into a stall -- the UI would freeze for
            # the length of whatever read is in flight. Under WAL readers and
            # one writer proceed concurrently and neither waits.
            #
            # The timeout stays as well, for the case WAL does not cover: two
            # WRITERS still serialise, so a scan saving results while a delete
            # commits needs somewhere to wait.
            #
            # Not applicable to `:memory:`, which has no file to journal --
            # sqlite reports "memory" back and ignores the request, so this is
            # left unguarded rather than special-cased.
            #
            # busy_timeout goes FIRST so that nothing below it can hit a lock
            # with no willingness to wait: switching journal mode takes a brief
            # exclusive lock, and `_initialize_tables()` runs DDL, which is a
            # write, so two connections opening at once do contend. At the
            # default timeout of zero the second one would fail rather than
            # wait. This ordering is a precaution reasoned from the locking
            # rules, not a fix for an observed failure -- both orders passed
            # the suite repeatedly, so do not read it as load-bearing.
            await connection.execute("PRAGMA busy_timeout = 5000")
            await self._enable_wal(connection)
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

    @staticmethod
    async def _enable_wal(connection: aiosqlite.Connection) -> None:
        """Put the database in WAL, tolerating a concurrent opener.

        busy_timeout does not cover this one. Changing journal mode needs a
        brief exclusive lock, and sqlite returns SQLITE_BUSY for it rather than
        waiting, so two connections creating the same database at the same
        instant can collide -- measured at 6 failures in 40 rounds once the DDL
        deadlock below was fixed.

        Retrying is enough because WAL is a property of the FILE, not of the
        connection: it is written into the database header and survives every
        close. So the race exists only at creation, and the loser only has to
        wait for the winner to finish, after which the pragma is a no-op that
        takes no lock at all.

        Ending up without WAL is not fatal and must not stop the app starting.
        The database still works under the rollback journal; readers and
        writers simply contend more, which is what busy_timeout is set for.
        """
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                await connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                # Someone else may have just set it, which is the good case.
                async with connection.execute("PRAGMA journal_mode") as cursor:
                    row = await cursor.fetchone()
                if row is not None and str(row[0]).lower() == "wal":
                    return
                await asyncio.sleep(_LOCK_RETRY_BASE_SECONDS * (attempt + 1))
            else:
                return

    def _require_db(self) -> aiosqlite.Connection:
        if self.db is None:
            raise RuntimeError("Database is not connected")
        return self.db

    async def _initialize_tables(self) -> None:
        db = self._require_db()

        # IMMEDIATE, because busy_timeout cannot save the DDL below. sqlite3
        # runs at isolation_level '', so these statements would otherwise go
        # inside an implicit DEFERRED transaction: the connection takes a
        # SHARED lock on its first statement and asks to upgrade on its first
        # write. Two connections both holding SHARED and both wanting to
        # upgrade is a genuine deadlock, so sqlite returns SQLITE_BUSY
        # *immediately* rather than waiting out the timeout -- waiting could
        # never resolve it. IMMEDIATE takes the write lock up front, leaving
        # no upgrade to deadlock on, and busy_timeout then applies normally.
        #
        # Measured, not reasoned: two connections opening one fresh database
        # concurrently raised "database is locked" out of here 14 times in 40
        # rounds, and 0 in 40 afterwards. The TUI does exactly this -- the
        # results pane loads its listing while a detail load or a delete opens
        # its own connection -- so it was reachable by a user, not only by the
        # suite. It is here rather than in `connect()` so the transaction sits
        # with the statements it protects, and so anything that replaces this
        # method replaces its locking too.
        # Retried for the same reason `_enable_wal` is: on a database that does
        # not exist yet, several connections can be inside CREATE at once and
        # the loser still meets SQLITE_BUSY here. Once any one of them finishes,
        # the file exists in WAL and this stops contending -- so the wait is
        # short and bounded, and measured at 0 failures in 40 rounds for 2 and
        # 3 concurrent creations where 3-way was 8 in 40 without it.
        for attempt in range(_LOCK_RETRY_ATTEMPTS):
            try:
                await db.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                if attempt == _LOCK_RETRY_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_LOCK_RETRY_BASE_SECONDS * (attempt + 1))
            else:
                break

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
        # Which model produced the stored extraction. Deliberately NOT part of
        # the contract hash: a different model's extraction is still valid
        # against the current contract, so it must not be invalidated on sight
        # -- that would discard every cached extraction on disk the moment
        # anyone tried a second model, and discard it again on switching back.
        # Recorded instead so a mixed-model profile is visible rather than
        # silent. NULL means the row predates this column.
        await self._ensure_column(
            table_name="results",
            column_name="ai_extraction_model",
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
        # HOW the site was fetched: "browser", "api", or "http". Evidence
        # provenance, not telemetry -- a hit found without a browser is weaker
        # than one found with it, because a plain request runs no JavaScript and
        # a client-rendered profile arrives without the marker the rule looks
        # for. Months later this column is the only record of which it was.
        # It also stops a cheap answer from silently satisfying the resume
        # filter for a later browser scan. NULL means the row predates it.
        await self._ensure_column(
            table_name="results",
            column_name="transport",
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
        try:
            await db.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )
        except aiosqlite.OperationalError as error:
            # Losing the race is success. The check above and the ALTER below
            # are two statements, so two connections opening the same database
            # at once can both find the column missing and both try to add it --
            # the second one fails with "duplicate column name" even though the
            # column now exists, which is exactly the state this method wanted.
            #
            # Not hypothetical: the UI opens a connection for the results list
            # while a scan holds its own, and on a database created fresh that
            # raced on the first run and took the scan down with it.
            if "duplicate column name" not in str(error).lower():
                raise

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
        transport: str | None = None,
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
                        confidence,
                        transport
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(username_id, site_name) DO UPDATE SET
                        site_url = excluded.site_url,
                        status = excluded.status,
                        status_code = excluded.status_code,
                        query_time_ms = excluded.query_time_ms,
                        error_context = excluded.error_context,
                        response_text = excluded.response_text,
                        confidence = excluded.confidence,
                        -- Overwritten rather than preserved on purpose: the
                        -- column describes the row that is being written, so a
                        -- re-check over a different transport must say so.
                        transport = excluded.transport,
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
                        ai_extraction_model = CASE
                            WHEN excluded.ai_extraction IS NOT NULL
                                THEN NULL
                            WHEN ?
                                THEN NULL
                            WHEN excluded.status IS NOT results.status
                                OR excluded.response_text IS NOT results.response_text
                                THEN NULL
                            ELSE results.ai_extraction_model
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
                        transport,
                        force_ai_extraction,
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
                            ai_extraction_contract_hash = NULL,
                            ai_extraction_model = NULL
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
        model_key: str,
    ) -> None:
        db = self._require_db()

        async with self._write_lock:
            try:
                async with db.execute(
                    """
                    UPDATE results
                    SET
                        ai_extraction = ?,
                        ai_extraction_contract_hash = ?,
                        ai_extraction_model = ?
                    WHERE id = ?
                    """,
                    (ai_extraction, contract_hash, model_key, site_id),
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

    async def get_extraction_model_counts(
        self,
        username: str,
    ) -> dict[str | None, int]:
        """Count stored extractions per model that produced them.

        The key is the model, or None for rows written before the model was
        recorded. Rows with no extraction at all are excluded: they have
        nothing to attribute, and counting them would make every unscanned
        site look like unrecorded provenance.
        """
        db = self._require_db()

        async with db.execute(
            """
            SELECT
                r.ai_extraction_model AS model,
                COUNT(*) AS extraction_count
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
                AND r.ai_extraction IS NOT NULL
            GROUP BY r.ai_extraction_model
            """,
            (username,),
        ) as cur:
            rows = await cur.fetchall()

        return {
            (str(row["model"]) if row["model"] is not None else None): int(
                row["extraction_count"]
            )
            for row in rows
        }

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

    async def list_usernames(self) -> list[StoredUsernameListing]:
        """Every username with stored results, most recently scanned first.

        Scan date comes from `MAX(results.scanned_at)` for the same reason
        `get_username_overview` uses it: `usernames.last_scanned_at` is written
        once on insert and never updated, so it answers "first seen", and a
        list sorted by it would put a username scanned this morning below one
        first seen last year and never touched since.

        A username row with no results is excluded by the join. That is
        deliberate -- one can exist after an interrupted scan, and offering a
        name whose detail view is empty is worse than not offering it.
        """
        db = self._require_db()

        async with db.execute(
            """
            SELECT
                u.username AS username,
                COUNT(r.id) AS total_sites,
                SUM(CASE WHEN r.status = ? THEN 1 ELSE 0 END) AS claimed_sites,
                MAX(r.scanned_at) AS last_scanned_at,
                u.profile_summary IS NOT NULL AS has_profile
            FROM usernames u
            JOIN results r
                ON u.id = r.username_id
            GROUP BY u.id, u.username, u.profile_summary
            ORDER BY MAX(r.scanned_at) DESC, u.username ASC
            """,
            (str(QueryStatus.CLAIMED),),
        ) as cur:
            rows = await cur.fetchall()

        return [
            StoredUsernameListing(
                username=row["username"],
                total_sites=int(row["total_sites"] or 0),
                claimed_sites=int(row["claimed_sites"] or 0),
                last_scanned_at=row["last_scanned_at"],
                has_profile=bool(row["has_profile"]),
            )
            for row in rows
        ]

    async def delete_username(self, username: str) -> int:
        """Erase everything stored for one username. Returns rows removed.

        Both tables, in one transaction. Deleting the results and leaving the
        `usernames` row would keep the name, its first-seen date and its stored
        pass-two profile on disk -- which for a tool whose subject is people is
        not a tidy-up detail: someone asking to remove a person's record means
        the record, not most of it.

        Returns the number of RESULT rows removed, because that is the figure
        the caller showed the user when asking them to confirm.
        """
        db = self._require_db()

        async with db.execute(
            "SELECT id FROM usernames WHERE username = ?", (username,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return 0

        username_id = int(row["id"])
        async with db.execute(
            "SELECT COUNT(*) AS n FROM results WHERE username_id = ?",
            (username_id,),
        ) as cur:
            counted = await cur.fetchone()
        removed = int(counted["n"]) if counted is not None else 0

        await db.execute("DELETE FROM results WHERE username_id = ?", (username_id,))
        await db.execute("DELETE FROM usernames WHERE id = ?", (username_id,))
        await db.commit()
        return removed

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
                r.transport,
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
