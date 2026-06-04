import pytest
import pytest_asyncio
from typing import Any
from sherlock_project.database import SherlockDB

pytestmark = pytest.mark.asyncio




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
        "scanned_at",
    }


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

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["site_url"] == user_data["site_url"]
    assert row["status"] == user_data["status"]
    assert row["status_code"] == user_data["status_code"]
    assert row["query_time_ms"] == user_data["query_time_ms"]
    assert row["error_context"] == user_data["error_context"]
    assert row["response_text"] == user_data["response_text"]
    assert row["ai_extraction"] is None


async def test_save_result_upserts_same_username_site(db: SherlockDB, user_data: dict[str, Any]):
    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status="AVAILABLE",
        status_code=200,
        query_time_ms=111.11,
        error_context=None,
        response_text="old text",
    )

    await db.save_result(
        username=user_data["username"],
        site_name=user_data["site_name"],
        site_url=user_data["site_url"],
        status="CLAIMED",
        status_code=404,
        query_time_ms=222.22,
        error_context="updated",
        response_text="new text",
    )

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
    assert row["status"] == "CLAIMED"
    assert row["status_code"] == 404
    assert row["query_time_ms"] == 222.22
    assert row["error_context"] == "updated"
    assert row["response_text"] == "new text"


async def test_update_result_ai_extraction_updates_field(db: SherlockDB, user_data: dict[str, Any]):
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

    ai_extraction = '{"name": "Blue", "bio": "Developer"}'
    await db.update_result_ai_extraction(
        username=user_data["username"],
        site_name=user_data["site_name"],
        ai_extraction=ai_extraction,
    )

    row = await _get_result_row(db, user_data["username"], user_data["site_name"])
    assert row is not None
    assert row["ai_extraction"] == ai_extraction


async def test_update_username_profile_summary_inserts_and_updates(db: SherlockDB, user_data: dict[str, Any]):
    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="First summary",
    )

    row = await _get_username_row(db, user_data["username"])
    assert row is not None
    assert row["profile_summary"] == "First summary"
    assert row["last_scanned_at"] is not None

    await db.update_username_profile_summary(
        username=user_data["username"],
        profile_summary="Updated summary",
    )

    row = await _get_username_row(db, user_data["username"])
    assert row is not None
    assert row["profile_summary"] == "Updated summary"


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