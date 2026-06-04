from __future__ import annotations
import aiosqlite


class SherlockDB:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self.db: aiosqlite.Connection | None = None

    @classmethod
    async def create(cls, database_path: str) -> "SherlockDB":
        self = cls(database_path)
        await self.connect()
        return self

    async def connect(self) -> None:
        if self.db is not None:
            return

        self.db = await aiosqlite.connect(self.database_path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA foreign_keys = ON")
        await self._initialize_tables()

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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                profile_summary TEXT,
                last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username_id INTEGER NOT NULL,
                site_name TEXT NOT NULL,
                site_url TEXT,
                status TEXT NOT NULL,
                status_code INTEGER,
                query_time_ms REAL,
                error_context TEXT,
                response_text TEXT,
                ai_extraction TEXT,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username_id) REFERENCES usernames(id),
                UNIQUE(username_id, site_name)
            )
        """)

        await db.commit()

    async def clear_tables(self) -> None:
        db = self._require_db()

        await db.execute("DELETE FROM results")
        await db.execute("DELETE FROM usernames")
        await db.commit()

    async def get_or_create_username_id(self, username: str) -> int:
        db = self._require_db()

        try:
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

            await db.commit()
            return int(row["id"])

        except Exception:
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
    ) -> None:
        db = self._require_db()
        username_id = await self.get_or_create_username_id(username)

        try:
            await db.execute(
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
                    ai_extraction
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username_id, site_name) DO UPDATE SET
                    site_url = excluded.site_url,
                    status = excluded.status,
                    status_code = excluded.status_code,
                    query_time_ms = excluded.query_time_ms,
                    error_context = excluded.error_context,
                    response_text = excluded.response_text,
                    ai_extraction = COALESCE(excluded.ai_extraction, results.ai_extraction),
                    scanned_at = CURRENT_TIMESTAMP
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
                ),
            )
            await db.commit()

        except Exception:
            await db.rollback()
            raise

    async def update_result_ai_extraction(
        self,
        username: str,
        site_name: str,
        ai_extraction: str,
    ) -> None:
        db = self._require_db()
        username_id = await self.get_or_create_username_id(username)

        await db.execute(
            """
            UPDATE results
            SET ai_extraction = ?,
                scanned_at = CURRENT_TIMESTAMP
            WHERE username_id = ? AND site_name = ?
            """,
            (ai_extraction, username_id, site_name),
        )
        await db.commit()

    async def update_username_profile_summary(
        self,
        username: str,
        profile_summary: str,
    ) -> None:
        db = self._require_db()

        await db.execute(
            """
            INSERT INTO usernames (username, profile_summary)
            VALUES (?, ?)
            ON CONFLICT(username) DO UPDATE SET
                profile_summary = excluded.profile_summary,
                last_scanned_at = CURRENT_TIMESTAMP
            """,
            (username, profile_summary),
        )
        await db.commit()

    async def get_username_by_id(self, username_id: int) -> str | None:
        db = self._require_db()

        async with db.execute(
            "SELECT username FROM usernames WHERE id = ?",
            (username_id,),
        ) as cur:
            row = await cur.fetchone()

        return row["username"] if row else None
    
    async def get_saved_sites(self, username: str) -> set[str]:
        db = self._require_db()

        async with db.execute(
            """
            SELECT r.site_name
            FROM results r
            JOIN usernames u
                ON u.id = r.username_id
            WHERE u.username = ?
            """,
            (username,),
        ) as cur:
            rows = await cur.fetchall()

        return {row["site_name"] for row in rows}