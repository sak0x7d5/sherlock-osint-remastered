import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.result import QueryStatus

pytestmark = pytest.mark.asyncio

CONTRACT_HASH = "pass-one-contract-v2"
MODEL_KEY = "vendor/test-model"


@pytest.fixture()
def user_data() -> dict[str, Any]:
    return {
        "username": "blue",
        "site_name": "instagram",
        "site_url": "https://instagram.com/blue",
        "status": "AVAILABLE",
        "status_code": 200,
        "query_time_ms": 123.45,
        "error_context": None,
        "response_text": "<html>User Not Found</html>",
        "ai_extraction": None,
        "profile_summary": None,
    }


async def _get_result_row(db: SherlockDB, username: str, site_name: str):
    assert db.db is not None
    async with db.db.execute(
        """
        SELECT r.*
        FROM results r
        JOIN usernames u ON u.id = r.username_id
        WHERE u.username = ? AND r.site_name = ?
        """,
        (username, site_name),
    ) as cur:
        return await cur.fetchone()


async def _get_username_row(db: SherlockDB, username: str):
    assert db.db is not None
    async with db.db.execute(
        "SELECT * FROM usernames WHERE username = ?",
        (username,),
    ) as cur:
        return await cur.fetchone()


async def test_schema_is_created(db: SherlockDB):
    assert db.db is not None

    async with db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ) as cur:
        rows = await cur.fetchall()

    table_names = {row["name"] for row in rows}
    assert "usernames" in table_names
    assert "results" in table_names

    async with db.db.execute("PRAGMA table_info(usernames)") as cur:
        usernames_cols = await cur.fetchall()
    async with db.db.execute("PRAGMA table_info(results)") as cur:
        results_cols = await cur.fetchall()

    assert {col["name"] for col in usernames_cols} >= {
        "id",
        "username",
        "profile_summary",
        "profile_summary_input_hash",
        "profile_summary_updated_at",
        "last_scanned_at",
    }
    assert {col["name"] for col in results_cols} >= {
        "id",
        "username_id",
        "site_name",
        "site_url",
        "status",
        "status_code",
        "query_time_ms",
        "error_context",
        "response_text",
        "ai_extraction",
        "ai_extraction_contract_hash",
        "confidence",
        "transport",
        "scanned_at",
    }


async def test_cancelled_schema_initialization_closes_connection_and_can_reopen(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "cancelled-initialization.db"
    initialization_started = asyncio.Event()
    captured_connections = []
    rollback_calls = []
    close_calls = []
    original_rollback = aiosqlite.Connection.rollback
    original_close = aiosqlite.Connection.close

    async def block_during_initialization(database: SherlockDB) -> None:
        connection = database._require_db()
        captured_connections.append(connection)
        await connection.execute("BEGIN IMMEDIATE")
        await connection.execute("CREATE TABLE cancelled_startup (id INTEGER)")
        initialization_started.set()
        await asyncio.Event().wait()

    async def record_rollback(connection) -> None:
        rollback_calls.append(connection)
        await original_rollback(connection)

    async def record_close(connection) -> None:
        close_calls.append(connection)
        await original_close(connection)

    with monkeypatch.context() as patch:
        patch.setattr(SherlockDB, "_initialize_tables", block_during_initialization)
        patch.setattr(aiosqlite.Connection, "rollback", record_rollback)
        patch.setattr(aiosqlite.Connection, "close", record_close)

        create_task = asyncio.create_task(SherlockDB.create(str(database_path)))
        try:
            await asyncio.wait_for(initialization_started.wait(), timeout=1)
            create_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await create_task
        finally:
            if not create_task.done():
                create_task.cancel()
                await asyncio.gather(create_task, return_exceptions=True)

    assert len(captured_connections) == 1
    connection = captured_connections[0]
    assert rollback_calls == [connection]
    assert close_calls == [connection]
    assert connection._connection is None

    reopened = await SherlockDB.create(str(database_path))
    try:
        assert reopened.db is not None
        async with reopened.db.execute(
            "SELECT name FROM sqlite_master WHERE name = 'cancelled_startup'"
        ) as cur:
            assert await cur.fetchone() is None

        await reopened.save_result(
            username="reopened",
            site_name="write-after-cancel",
            status=str(QueryStatus.CLAIMED),
            response_text="database remains writable",
        )
        assert (
            await _get_result_row(reopened, "reopened", "write-after-cancel")
            is not None
        )
    finally:
        await reopened.close()


async def test_schema_initialization_error_survives_cleanup_errors(monkeypatch):
    initialization_error = RuntimeError("schema initialization failed")
    original_close = aiosqlite.Connection.close

    async def fail_initialization(database: SherlockDB) -> None:
        raise initialization_error

    async def fail_rollback(connection) -> None:
        raise OSError("rollback failed")

    async def close_then_fail(connection) -> None:
        await original_close(connection)
        raise OSError("close failed")

    with monkeypatch.context() as patch:
        patch.setattr(SherlockDB, "_initialize_tables", fail_initialization)
        patch.setattr(aiosqlite.Connection, "rollback", fail_rollback)
        patch.setattr(aiosqlite.Connection, "close", close_then_fail)

        with pytest.raises(RuntimeError) as exc_info:
            await SherlockDB.create(":memory:")

    assert exc_info.value is initialization_error
    assert exc_info.value.__notes__ == [
        (
            "Database rollback during connection cleanup also failed: "
            "OSError('rollback failed')"
        ),
        (
            "Database close during connection cleanup also failed: "
            "OSError('close failed')"
        ),
    ]


async def test_existing_database_is_migrated_for_profile_cache(tmp_path):
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE usernames (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            profile_summary TEXT,
            last_scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE results (
            id INTEGER PRIMARY KEY,
            username_id INTEGER NOT NULL,
            site_name TEXT NOT NULL,
            status TEXT NOT NULL,
            response_text TEXT,
            ai_extraction TEXT
        )
        """
    )
    connection.commit()
    connection.close()

    migrated = await SherlockDB.create(str(database_path))
    try:
        assert migrated.db is not None
        async with migrated.db.execute("PRAGMA table_info(usernames)") as cur:
            username_columns = await cur.fetchall()
        async with migrated.db.execute("PRAGMA table_info(results)") as cur:
            result_columns = await cur.fetchall()
    finally:
        await migrated.close()

    username_column_names = {column["name"] for column in username_columns}
    result_column_names = {column["name"] for column in result_columns}
    assert "profile_summary_input_hash" in username_column_names
    assert "profile_summary_updated_at" in username_column_names
    assert "ai_extraction_contract_hash" in result_column_names


async def test_get_or_create_username_id_returns_id(db: SherlockDB, user_data: dict[str, Any]):
    username_id = await db.get_or_create_username_id(user_data["username"])
    assert isinstance(username_id, int)
    assert username_id > 0

    row = await _get_username_row(db, user_data["username"])
    assert row is not None
    assert row["id"] == username_id
    assert row["username"] == user_data["username"]


async def test_get_or_create_username_id_is_stable(db: SherlockDB, user_data: dict[str, Any]):
    first_id = await db.get_or_create_username_id(user_data["username"])
    second_id = await db.get_or_create_username_id(user_data["username"])

    assert first_id == second_id


async def test_save_result_persists_row(db: SherlockDB, user_data: dict[str, Any]):
    result_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status=user_data["status"],
        status_code=user_data["status_code"],
        query_time_ms=user_data["query_time_ms"],
        error_context=user_data["error_context"],
        response_text=user_data["response_text"],
    )

    assert isinstance(result_id, int)
    assert result_id > 0

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["id"] == result_id
    assert row["site_url"] == user_data["site_url"]
    assert row["status"] == user_data["status"]
    assert row["status_code"] == user_data["status_code"]
    assert row["query_time_ms"] == user_data["query_time_ms"]
    assert row["error_context"] == user_data["error_context"]
    assert row["response_text"] == user_data["response_text"]
    assert row["ai_extraction"] is None
    assert row["confidence"] is None


async def test_save_result_persists_confidence(db: SherlockDB, user_data: dict[str, Any]):
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        confidence="Confirmed",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row["confidence"] == "Confirmed"


async def test_save_result_updates_confidence_on_rescan(
    db: SherlockDB, user_data: dict[str, Any]
):
    """A rescan that downgrades the evidence must not leave the old grade behind."""
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        confidence="Confirmed",
    )
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        confidence="Probable",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row["confidence"] == "Probable"


async def test_save_result_persists_transport(db: SherlockDB, user_data: dict[str, Any]):
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        transport="http",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row["transport"] == "http"


async def test_save_result_updates_transport_on_rescan(
    db: SherlockDB, user_data: dict[str, Any]
):
    """A re-check over a different transport must not keep the old label.

    The column describes how THIS row was fetched. Leaving "http" on a row a
    browser has since re-checked would understate evidence that is now good,
    and leaving "browser" on one answered cheaply would overstate it -- the
    direction that actually costs an investigation something.
    """
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        transport="http",
    )
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Claimed",
        transport="browser",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row["transport"] == "browser"


async def test_get_saved_results_reports_transport(
    db: SherlockDB, user_data: dict[str, Any]
):
    """The scan reads this to decide whether a stored row may be resumed."""
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Available",
        transport="http",
    )

    saved = await db.get_saved_results(user_data["username"])
    assert saved[user_data["site_name"]]["transport"] == "http"


async def test_save_result_transport_defaults_to_unrecorded(
    db: SherlockDB, user_data: dict[str, Any]
):
    """NULL means "written before the column existed", not "no browser"."""
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status="Available",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row["transport"] is None


async def test_save_result_upserts_same_username_site(db: SherlockDB, user_data: dict[str, Any]):
    first_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status="AVAILABLE",
        status_code=200,
        query_time_ms=111.11,
        error_context=None,
        response_text="old text",
    )

    second_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status="CLAIMED",
        status_code=404,
        query_time_ms=222.22,
        error_context="updated",
        response_text="new text",
    )

    assert second_id == first_id

    assert db.db is not None
    async with db.db.execute(
        """
        SELECT COUNT(*) AS count
        FROM results r
        JOIN usernames u ON u.id = r.username_id
        WHERE u.username = ? AND r.site_name = ?
        """,
        (user_data["username"], user_data["site_name"]),
    ) as cur:
        count_row = await cur.fetchone()

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])

    assert count_row["count"] == 1
    assert row is not None
    assert row["id"] == first_id
    assert row["status"] == "CLAIMED"
    assert row["status_code"] == 404
    assert row["query_time_ms"] == 222.22
    assert row["error_context"] == "updated"
    assert row["response_text"] == "new text"


async def test_update_result_ai_extraction_updates_field(db: SherlockDB, user_data: dict[str, Any]):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status=user_data["status"],
        status_code=user_data["status_code"],
        query_time_ms=user_data["query_time_ms"],
        error_context=user_data["error_context"],
        response_text=user_data["response_text"],
    )

    ai_extraction = '{"name": "Blue", "bio": "Developer"}'
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction=ai_extraction,
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] == ai_extraction
    assert row["ai_extraction_contract_hash"] == CONTRACT_HASH


async def test_update_result_ai_extraction_preserves_scan_timestamp(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    assert db.db is not None
    scanned_at = "2000-01-01 00:00:00"
    await db.db.execute(
        "UPDATE results SET scanned_at = ? WHERE id = ?",
        (scanned_at, site_id),
    )
    await db.db.commit()

    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"name": "Blue"}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["scanned_at"] == scanned_at


async def test_update_result_ai_extraction_raises_for_missing_site(db: SherlockDB):
    with pytest.raises(RuntimeError, match="Failed to update AI extraction"):
        await db.update_result_ai_extraction(
            site_id=999,
            ai_extraction='{"name": "Missing"}',
            contract_hash=CONTRACT_HASH,
            model_key=MODEL_KEY,
        )


async def test_get_pending_ai_extraction_ids_filters_for_eligible_rows(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    pending_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status=str(QueryStatus.CLAIMED),
        response_text=user_data["response_text"],
    )
    await db.save_result(
        username=user_data["username"],
        site_name="github",
        site_url="https://github.com/blue",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction='{"name": "Blue"}',
        ai_extraction_contract_hash=CONTRACT_HASH,
    )
    await db.save_result(
        username=user_data["username"],
        site_name="reddit",
        status=str(QueryStatus.AVAILABLE),
        response_text="not found",
    )
    await db.save_result(
        username=user_data["username"],
        site_name="empty",
        status=str(QueryStatus.CLAIMED),
        response_text="   ",
    )
    await db.save_result(
        username="green",
        site_name=user_data["site_name"],
        site_url="https://instagram.com/green",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    legacy_id = await db.save_result(
        username=user_data["username"],
        site_name="legacy-cache",
        status=str(QueryStatus.CLAIMED),
        response_text="legacy profile",
        ai_extraction='{"full_name": ["Blue Legacy"]}',
    )
    stale_id = await db.save_result(
        username=user_data["username"],
        site_name="stale-cache",
        status=str(QueryStatus.CLAIMED),
        response_text="stale profile",
        ai_extraction='{"full_name": ["Blue Stale"]}',
        ai_extraction_contract_hash="old-contract",
    )

    assert await db.get_pending_ai_extraction_ids(
        user_data["username"],
        contract_hash=CONTRACT_HASH,
    ) == [pending_id, legacy_id, stale_id]


async def test_get_ai_extraction_job_joins_username_and_site_data(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile content",
    )

    job = await db.get_ai_extraction_job(
        site_id,
        contract_hash=CONTRACT_HASH,
    )

    assert job is not None
    assert job.site_id == site_id
    assert job.username == user_data["username"]
    assert job.site_name == user_data["site_name"]
    assert job.response_text == "profile content"


async def test_get_ai_extraction_job_clears_stale_cache_transactionally(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile content",
        ai_extraction='{"full_name": ["Blue Old"]}',
        ai_extraction_contract_hash="old-contract",
    )
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary='{"full_name": ["Blue Old"]}',
        input_hash="cached-profile-hash",
    )

    job = await db.get_ai_extraction_job(
        site_id,
        contract_hash=CONTRACT_HASH,
    )

    assert job is not None
    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] is None
    assert row["ai_extraction_contract_hash"] is None
    cache = await db.get_profile_summary_cache(user_data["username"])
    assert cache is not None
    assert cache.profile_summary == '{"full_name": ["Blue Old"]}'
    assert cache.input_hash is None


async def test_get_ai_extraction_job_skips_current_cached_extraction(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile content",
        ai_extraction='{"full_name": ["Blue"]}',
        ai_extraction_contract_hash=CONTRACT_HASH,
    )

    assert await db.get_ai_extraction_job(
        site_id,
        contract_hash=CONTRACT_HASH,
    ) is None


async def test_save_result_invalidates_ai_only_when_source_changes(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    original_ai = '{"name": "Blue"}'
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
        ai_extraction=original_ai,
        ai_extraction_contract_hash=CONTRACT_HASH,
    )

    same_site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
    )
    unchanged_row = await _get_result_row(
        db,
        user_data["username"],
        user_data["site_name"],
    )

    assert same_site_id == site_id
    assert unchanged_row is not None
    assert unchanged_row["ai_extraction"] == original_ai
    assert unchanged_row["ai_extraction_contract_hash"] == CONTRACT_HASH

    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="changed profile",
    )
    changed_row = await _get_result_row(
        db,
        user_data["username"],
        user_data["site_name"],
    )

    assert changed_row is not None
    assert changed_row["ai_extraction"] is None
    assert changed_row["ai_extraction_contract_hash"] is None


async def test_save_result_can_force_fresh_ai_extraction_for_same_source(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    original_ai = '{"full_name": "Blue Example"}'
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
        ai_extraction=original_ai,
        ai_extraction_contract_hash=CONTRACT_HASH,
    )
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="cached profile",
        input_hash="cached-hash",
    )

    refreshed_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
        force_ai_extraction=True,
    )

    result_row = await _get_result_row(
        db,
        user_data["username"],
        user_data["site_name"],
    )
    username_row = await _get_username_row(db, user_data["username"])
    assert refreshed_id == site_id
    assert result_row is not None
    assert result_row["ai_extraction"] is None
    assert result_row["ai_extraction_contract_hash"] is None
    assert username_row is not None
    assert username_row["profile_summary_input_hash"] is None
    assert await db.get_ai_extraction_job(
        site_id,
        contract_hash=CONTRACT_HASH,
    ) is not None


async def test_concurrent_result_writes_are_serialized(db: SherlockDB):
    site_ids = await asyncio.gather(
        *(
            db.save_result(
                username="blue",
                site_name=f"site-{index}",
                status=str(QueryStatus.CLAIMED),
                response_text=f"profile-{index}",
            )
            for index in range(10)
        )
    )

    assert len(set(site_ids)) == 10
    assert set(
        await db.get_pending_ai_extraction_ids(
            "blue",
            contract_hash=CONTRACT_HASH,
        )
    ) == set(site_ids)


async def test_cancelled_write_rolls_back_and_database_remains_usable(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "cancelled-write.db"
    database = await SherlockDB.create(str(database_path))
    commit_started = asyncio.Event()

    assert database.db is not None

    async def wait_forever_in_commit() -> None:
        commit_started.set()
        await asyncio.Event().wait()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(database.db, "commit", wait_forever_in_commit)
            write_task = asyncio.create_task(
                database.save_result(
                    username="cancelled",
                    site_name="in-flight",
                    status=str(QueryStatus.CLAIMED),
                    response_text="must not be committed",
                )
            )
            await asyncio.wait_for(commit_started.wait(), timeout=1)
            write_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await write_task

        assert await _get_result_row(database, "cancelled", "in-flight") is None

        await database.save_result(
            username="preserved",
            site_name="committed-before-reopen",
            status=str(QueryStatus.CLAIMED),
            response_text="committed",
        )
    finally:
        await database.close()

    reopened = await SherlockDB.create(str(database_path))
    try:
        assert await _get_result_row(reopened, "cancelled", "in-flight") is None
        assert (
            await _get_result_row(
                reopened,
                "preserved",
                "committed-before-reopen",
            )
            is not None
        )

        await reopened.save_result(
            username="preserved",
            site_name="committed-after-reopen",
            status=str(QueryStatus.CLAIMED),
            response_text="still writable",
        )
        assert (
            await _get_result_row(
                reopened,
                "preserved",
                "committed-after-reopen",
            )
            is not None
        )
    finally:
        await reopened.close()


async def test_update_username_profile_summary_inserts_and_updates(db: SherlockDB, user_data: dict[str, Any]):
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="First summary",
        input_hash="hash-one",
    )

    row = await _get_username_row(db, user_data["username"])
    assert row is not None
    assert row["profile_summary"] == "First summary"
    assert row["profile_summary_input_hash"] == "hash-one"
    assert row["profile_summary_updated_at"] is not None
    assert row["last_scanned_at"] is not None

    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="Updated summary",
        input_hash="hash-two",
    )

    row = await _get_username_row(db, user_data["username"])
    assert row is not None
    assert row["profile_summary"] == "Updated summary"
    assert row["profile_summary_input_hash"] == "hash-two"


async def test_ai_update_invalidates_hash_but_preserves_last_good_profile(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary='{"summary": "last good"}',
        input_hash="old-hash",
    )

    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": "Blue"}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    cache = await db.get_profile_summary_cache(user_data["username"])
    assert cache is not None
    assert cache.profile_summary == '{"summary": "last good"}'
    assert cache.input_hash is None


async def test_get_ai_profile_evidence_returns_claimed_profile_rows(
    db: SherlockDB,
):
    claimed_id = await db.save_result(
        username="blue",
        site_name="claimed",
        site_url="https://example.com/blue",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction='{"full_name": "Blue"}',
        ai_extraction_contract_hash=CONTRACT_HASH,
    )
    await db.save_result(
        username="blue",
        site_name="available",
        status=str(QueryStatus.AVAILABLE),
        response_text="not found",
    )

    records = await db.get_ai_profile_evidence("blue")

    assert len(records) == 1
    assert records[0].site_id == claimed_id
    assert records[0].site_name == "claimed"
    assert records[0].ai_extraction == '{"full_name": "Blue"}'
    assert records[0].ai_extraction_contract_hash == CONTRACT_HASH


async def test_get_username_by_id_when_exists(db: SherlockDB, user_data: dict[str, Any]):
    username_id = await db.get_or_create_username_id(user_data["username"])
    username = await db.get_username_by_id(username_id)

    assert username == user_data["username"]


async def test_get_username_by_id_when_not_exists(db: SherlockDB):
    username = await db.get_username_by_id(999)
    assert username is None


async def test_clear_tables(db: SherlockDB, user_data: dict[str, Any]):
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status=user_data["status"],
        status_code=user_data["status_code"],
        query_time_ms=user_data["query_time_ms"],
        error_context=user_data["error_context"],
        response_text=user_data["response_text"],
    )
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="Profile summary",
    )

    await db.clear_tables()

    assert db.db is not None
    async with db.db.execute("SELECT COUNT(*) AS count FROM results") as cur:
        results_row = await cur.fetchone()
    async with db.db.execute("SELECT COUNT(*) AS count FROM usernames") as cur:
        usernames_row = await cur.fetchone()

    assert results_row["count"] == 0
    assert usernames_row["count"] == 0


async def test_query_when_closed(db: SherlockDB):
    await db.close()

    with pytest.raises(RuntimeError):
        await db.get_username_by_id(1)


async def test_default_database_path_does_not_follow_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Results must land in one place regardless of where sherlock is run."""
    before = default_database_path({})

    monkeypatch.chdir(tmp_path)
    after = default_database_path({})

    assert before == after
    assert before.is_absolute()
    assert before.name == "sherlock.db"


async def test_default_database_path_honors_env_override(tmp_path: Path):
    override = tmp_path / "investigation.db"

    assert default_database_path({"SHERLOCK_DB": str(override)}) == override


async def test_default_database_path_expands_user_in_override():
    resolved = default_database_path({"SHERLOCK_DB": "~/sherlock/results.db"})

    assert "~" not in str(resolved)
    assert resolved.is_absolute()


async def test_connect_creates_missing_parent_directories(tmp_path: Path):
    """The user data directory does not exist until we write to it."""
    nested = tmp_path / "does" / "not" / "exist" / "sherlock.db"

    database = await SherlockDB.create(str(nested))
    try:
        assert nested.exists()
    finally:
        await database.close()


async def test_get_saved_results_returns_full_rows(db: SherlockDB):
    """The resume path needs the stored row, not just the site name."""
    await db.save_result(
        username="blue",
        site_name="GitHub",
        site_url="https://github.com/blue",
        status=str(QueryStatus.CLAIMED),
        status_code=200,
        query_time_ms=0.42,
        confidence="Confirmed",
    )

    saved = await db.get_saved_results("blue")

    assert set(saved) == {"GitHub"}
    row = saved["GitHub"]
    assert row["status"] == "Claimed"
    assert row["site_url"] == "https://github.com/blue"
    assert row["status_code"] == 200
    assert row["confidence"] == "Confirmed"
    # response_text is deliberately not selected; it holds whole pages.
    assert "response_text" not in row


async def test_update_result_ai_extraction_records_the_model(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """Provenance for the extraction, so a mixed-model profile is not silent."""
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction_model"] == MODEL_KEY


async def test_the_reasoning_is_stored_beside_the_extraction(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """How the model got there, not just what it decided.

    The extraction says what came out; this says why, and it is the half that
    moves when `resources/pass_one.md` is edited -- which is the only way to see
    a prompt change land on a real page.
    """
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
        reasoning="include Blue as full_name; skip: nav link",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction_reasoning"] == (
        "include Blue as full_name; skip: nav link"
    )


async def test_absent_reasoning_is_null_not_empty(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """NULL means "not recorded", which is not the same as "said nothing".

    Two real cases produce no reasoning and neither is a fault: a page that
    reduced to nothing was never sent to a model at all, and a model whose
    native thinking cannot be disabled is sent the variant prompt, which has no
    reasoning field. The viewer distinguishes those from a model that was asked
    and returned nothing, so an empty string must not be stored for them.
    """
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction="{}",
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction_reasoning"] is None


async def test_forced_extraction_clears_the_recorded_reasoning(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """Reasoning must never outlive the extraction it explains.

    Left behind, it would describe a reading of the page that is no longer on
    the row, and the viewer would show it beside whatever came next -- which is
    worse than showing nothing, because it looks like provenance.
    """
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
        reasoning="include Blue as full_name",
    )

    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
        force_ai_extraction=True,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] is None
    assert row["ai_extraction_reasoning"] is None


async def test_a_contract_change_clears_the_reasoning_too(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """The other route that discards an extraction, and the same rule applies.

    An edited Pass 1 prompt changes the contract hash, so the next run treats
    the row as pending and re-extracts. The reasoning that came with the old
    contract explains a decision made under different instructions.
    """
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
        reasoning="include Blue as full_name",
    )

    job = await db.get_ai_extraction_job(site_id=site_id, contract_hash="different")
    assert job is not None

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] is None
    assert row["ai_extraction_reasoning"] is None


async def test_get_site_extractions_reports_analysed_and_unanalysed_sites(
    db: SherlockDB,
):
    """Sites with no extraction are RETURNED, not filtered out.

    A model that extracted from 2 of 3 found sites must not look like one that
    was only ever asked about 2. `analysed` is what separates "asked and found
    nothing" from "never asked", and the two must not render the same way.
    """
    rich = await db.save_result(
        username="blue",
        site_name="GitHub",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    empty = await db.save_result(
        username="blue",
        site_name="Bandcamp",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    await db.save_result(
        username="blue",
        site_name="Zulip",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    # A site that was never found is not a gap in the analysis; it is not a
    # candidate for one, so it must not appear at all.
    await db.save_result(
        username="blue",
        site_name="Missing",
        status=str(QueryStatus.AVAILABLE),
        response_text="nothing",
    )

    await db.update_result_ai_extraction(
        site_id=rich,
        ai_extraction='{"full_name": ["Blue"], "emails": ["a@b.c", "d@e.f"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
        reasoning="include Blue as full_name",
    )
    await db.update_result_ai_extraction(
        site_id=empty,
        ai_extraction="{}",
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    records = {entry.site_name: entry for entry in await db.get_site_extractions("blue")}
    assert set(records) == {"GitHub", "Bandcamp", "Zulip"}

    # Facts counts VALUES, not keys: one key holding two emails is two facts,
    # and counting keys would flatten a productive model into an unproductive
    # looking one.
    assert records["GitHub"].fact_count == 3
    assert records["GitHub"].analysed is True
    assert records["GitHub"].ai_extraction_reasoning == "include Blue as full_name"

    # Asked, and the page held nothing. A result, not a gap.
    assert records["Bandcamp"].analysed is True
    assert records["Bandcamp"].fact_count == 0

    # Never asked. The distinction the facts column is drawn from.
    assert records["Zulip"].analysed is False
    assert records["Zulip"].fact_count == 0


async def test_a_corrupt_stored_extraction_reads_as_no_facts(db: SherlockDB):
    """Model output that was valid when written is not a promise about disk.

    A viewer that raised on one unparseable row would cost the user every other
    extraction on the screen, which is the failure isolation this codebase
    treats as a design property rather than a backlog item.
    """
    site_id = await db.save_result(
        username="blue",
        site_name="GitHub",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction="{not json at all",
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    (record,) = await db.get_site_extractions("blue")
    assert record.facts == {}
    assert record.fact_count == 0
    # Still analysed: something was written for this site. The row is not a gap.
    assert record.analysed is True


async def test_forced_extraction_clears_the_recorded_model(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """--fresh must not leave the old model attached to a redone extraction."""
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
        force_ai_extraction=True,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] is None
    assert row["ai_extraction_contract_hash"] is None
    assert row["ai_extraction_model"] is None


async def test_unforced_save_keeps_the_recorded_model(
    db: SherlockDB,
    user_data: dict[str, Any],
):
    """A plain re-save of unchanged content keeps both extraction and model.

    The model is recorded alongside the extraction, so it has to survive
    exactly as long as the extraction does -- otherwise every ordinary resume
    would quietly turn a known model into an unrecorded one.
    """
    site_id = await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Blue"]}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        status=str(QueryStatus.CLAIMED),
        response_text="same profile",
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] == '{"full_name": ["Blue"]}'
    assert row["ai_extraction_model"] == MODEL_KEY


async def test_get_extraction_model_counts_groups_by_model(db: SherlockDB):
    """Counts per model, with rows that predate the column reported as None."""
    ids = {}
    for site_name in ("GitHub", "GitLab", "Keybase", "Mastodon"):
        ids[site_name] = await db.save_result(
            username="blue",
            site_name=site_name,
            status=str(QueryStatus.CLAIMED),
            response_text=f"{site_name} profile",
        )

    await db.update_result_ai_extraction(
        site_id=ids["GitHub"],
        ai_extraction="{}",
        contract_hash=CONTRACT_HASH,
        model_key="vendor/small",
    )
    await db.update_result_ai_extraction(
        site_id=ids["GitLab"],
        ai_extraction="{}",
        contract_hash=CONTRACT_HASH,
        model_key="vendor/small",
    )
    await db.update_result_ai_extraction(
        site_id=ids["Keybase"],
        ai_extraction="{}",
        contract_hash=CONTRACT_HASH,
        model_key="vendor/large",
    )
    # A row written before the model was recorded.
    assert db.db is not None
    await db.db.execute(
        "UPDATE results SET ai_extraction = '{}', ai_extraction_model = NULL "
        "WHERE id = ?",
        (ids["Mastodon"],),
    )
    await db.db.commit()

    counts = await db.get_extraction_model_counts("blue")

    assert counts == {"vendor/small": 2, "vendor/large": 1, None: 1}


async def test_get_extraction_model_counts_ignores_sites_without_extractions(
    db: SherlockDB,
):
    """An unextracted site has nothing to attribute and must not read as None."""
    await db.save_result(
        username="blue",
        site_name="GitHub",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    assert await db.get_extraction_model_counts("blue") == {}
    assert await db.get_extraction_model_counts("nobody") == {}


async def test_a_delete_survives_a_read_held_open_on_another_connection(tmp_path):
    """Two connections to one file is this app's normal state, not an edge case.

    The results pane loads its listing on its own connection while a delete
    commits on another, and a scan writes while the pane reads. Under sqlite's
    default rollback journal a reader blocks the writer's COMMIT, and with no
    busy timeout sqlite does not wait for it -- it raises "database is locked"
    at once, which leaves `delete_username` half applied: the transaction that
    was meant to take both tables together fails partway.

    This reproduced deterministically here before the WAL pragma went in, and
    it is what failed CI on macOS and windows while ubuntu stayed green -- the
    race is real on every platform and only the timing of it differs, which is
    exactly the kind of fault that a timing-dependent test lets through.
    """
    path = str(tmp_path / "sherlock.db")

    writer = await SherlockDB.create(path)
    reader = await SherlockDB.create(path)
    try:
        for username in ("marcus", "keeper"):
            await writer.save_result(
                username=username,
                site_name="GitHub",
                status=str(QueryStatus.CLAIMED),
                response_text=None,
            )

        # A read transaction left open, the way an in-flight query holds one.
        await reader.db.execute("BEGIN")
        async with reader.db.execute("SELECT * FROM results") as cur:
            await cur.fetchone()

        assert await writer.delete_username("keeper") == 1
        remaining = [item.username for item in await writer.list_usernames()]
        assert remaining == ["marcus"]
    finally:
        await reader.db.rollback()
        await reader.close()
        await writer.close()


async def test_concurrent_creation_of_a_new_database_does_not_lock_itself(tmp_path):
    """Several connections opening a database that does not exist yet.

    A real path, not a contrived one: on a first run the results pane loads its
    listing while a scan or a detail load opens its own connection, and none of
    them finds a file there yet. Two connections doing this failed 14 times in
    40 rounds before `_initialize_tables` took its write lock with BEGIN
    IMMEDIATE -- `sqlite3.OperationalError: database is locked` raised out of
    the DDL, because sqlite3 opens its implicit transaction DEFERRED and
    busy_timeout does not wait out a lock UPGRADE deadlock.

    Four is above anything the app does -- the results pane's list and detail
    loads plus a scan is three -- and deliberately not higher. Eight was tried
    and passed on Linux and macOS while failing on Windows, where NTFS takes
    mandatory locks and the losers of the race need longer than a bounded retry
    is willing to wait. Asserting a number that only holds on the fastest
    platform is how a suite teaches people to ignore it.

    Opening only, too: several connections racing to WRITE into a database
    being created in the same instant can still contend. No caller does that,
    and a bounded retry cannot honestly promise it away.
    """
    database_path = tmp_path / "raced-into-existence.db"
    assert not database_path.exists()

    async def open_and_close() -> None:
        db = await SherlockDB.create(str(database_path))
        await db.close()

    await asyncio.gather(*(open_and_close() for _ in range(4)))

    # And the schema that survived the race is usable.
    db = await SherlockDB.create(str(database_path))
    try:
        await db.save_result(
            username="racer",
            site_name="Example",
            site_url="https://example.invalid/racer",
            status="Claimed",
            status_code=200,
            query_time_ms=1.0,
            error_context=None,
            response_text=None,
        )
        assert [item.username for item in await db.list_usernames()] == ["racer"]
    finally:
        await db.close()


async def _seed_mixed_manifest_history(db: SherlockDB) -> None:
    """A username scanned under two different site lists.

    The duplicate spelling is the point: the two manifests name the same
    platform differently, so the record ends up holding both.
    """
    for site_name in ("Threads", "GitHub", "threads", "Ask.fm"):
        await db.save_result(
            username="0day",
            site_name=site_name,
            site_url=f"https://example.invalid/{site_name}",
            status=str(QueryStatus.CLAIMED),
            response_text="profile",
        )


async def test_delete_retired_sites_drops_only_what_the_manifest_lost(
    db: SherlockDB,
):
    """The duplicate survives as one row, not two.

    `Threads` and `threads` are one platform under two site lists, and exact
    matching is what tells them apart -- the current list spells it one way, so
    the other spelling is the retired one.
    """
    await _seed_mixed_manifest_history(db)

    removed = await db.delete_retired_sites("0day", {"Threads", "GitHub"})

    assert removed == 2
    assert sorted(await db.get_saved_results("0day")) == ["GitHub", "Threads"]


async def test_delete_retired_sites_keeps_the_profile_and_extractions(
    db: SherlockDB,
):
    """The reason this is not `delete_username`.

    A re-scan rewrites the rows it re-checks, but only an AI run rebuilds an
    extraction and only synthesis rebuilds a profile. Wiping the record would
    charge a scan without analysis for work a model already did.
    """
    await _seed_mixed_manifest_history(db)
    await db.update_username_profile_summary(
        username="0day",
        profile_summary='{"username": "0day"}',
        input_hash="h",
    )
    kept = await _get_result_row(db, "0day", "GitHub")
    await db.update_result_ai_extraction(
        site_id=int(kept["id"]),
        ai_extraction='{"extraction": []}',
        contract_hash=CONTRACT_HASH,
        model_key=MODEL_KEY,
    )

    await db.delete_retired_sites("0day", {"Threads", "GitHub"})

    assert await db.get_profile_summary_cache("0day") is not None
    surviving = await _get_result_row(db, "0day", "GitHub")
    assert surviving["ai_extraction"] == '{"extraction": []}'
    assert surviving["ai_extraction_model"] == MODEL_KEY


async def test_delete_retired_sites_treats_an_empty_manifest_as_unknown(
    db: SherlockDB,
):
    """A manifest that failed to load must not read as "every site retired".

    This is the difference between nothing being known and nothing existing,
    and getting it wrong erases the record it was asked to tidy.
    """
    await _seed_mixed_manifest_history(db)

    assert await db.delete_retired_sites("0day", set()) == 0
    assert len(await db.get_saved_results("0day")) == 4


async def test_delete_retired_sites_is_quiet_when_there_is_nothing_to_do(
    db: SherlockDB,
):
    """Every re-scan after the first, and every username never scanned."""
    await _seed_mixed_manifest_history(db)

    assert await db.delete_retired_sites("0day", {"Threads", "GitHub", "threads", "Ask.fm"}) == 0
    assert await db.delete_retired_sites("nobody", {"GitHub"}) == 0
