"""Tests for `sherlock show`, the read-only view of stored results.

The behaviour worth protecting here is negative: showing must never write.
Before this command existed, the only way to display a pass-two profile was
`--ai-synthesize-only`, which rebuilds and overwrites it, so looking at a
profile destroyed it.
"""

import json

import pytest

from sherlock_project.database import SherlockDB
from sherlock_project.profile_synthesis import IdentityAnchor, ProfileSynthesis
from sherlock_project.result import QueryStatus
from sherlock_project.show import _claimed_accounts, _load_profile, run_show


def _profile_json(username: str) -> str:
    profile = ProfileSynthesis(
        username=username,
        input_hash="hash-1",
        mode="aggregate",
        resolution_status="aggregated",
        completeness="partial",
        strong_profile={"display_name": ["Tony"]},
    )
    return json.dumps(profile.model_dump(mode="json"), sort_keys=True)


async def _seed(path: str, *, with_profile: bool = True) -> None:
    db = await SherlockDB.create(path)
    try:
        await db.save_result(
            username="blue",
            site_name="GitHub",
            site_url="https://github.com/blue",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            confidence="Confirmed",
        )
        await db.save_result(
            username="blue",
            site_name="Reddit",
            site_url="https://reddit.com/u/blue",
            status=str(QueryStatus.AVAILABLE),
            status_code=404,
        )
        if with_profile:
            await db.update_username_profile_summary(
                username="blue",
                profile_summary=_profile_json("blue"),
                input_hash="hash-1",
            )
    finally:
        await db.close()


async def _snapshot(path: str) -> list[tuple]:
    """Everything `show` must leave untouched."""
    db = await SherlockDB.create(path)
    try:
        cache = await db.get_profile_summary_cache("blue")
        rows = await db.get_saved_results("blue")
    finally:
        await db.close()
    return [
        (cache.profile_summary, cache.input_hash, cache.updated_at),
        sorted((name, row["status"]) for name, row in rows.items()),
    ]


def test_claimed_accounts_filters_and_sorts():
    rows = {
        "Zulip": {"status": "Claimed", "site_url": "https://z.example/blue"},
        "GitHub": {"status": "Claimed", "site_url": "https://github.com/blue"},
        "Reddit": {"status": "Available", "site_url": "https://r.example/blue"},
    }

    accounts = _claimed_accounts(rows)

    assert [item["site_name"] for item in accounts] == ["GitHub", "Zulip"]


def test_load_profile_returns_none_for_unreadable_summary():
    """A profile from an older schema must not crash the viewer."""
    assert _load_profile(None) is None
    assert _load_profile("") is None
    assert _load_profile('{"schema_version": 1, "gone": true}') is None


async def test_show_does_not_modify_the_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The whole point of the command: looking changes nothing."""
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    before = await _snapshot(str(database))
    exit_code = await run_show(["blue", "--no-color"])
    after = await _snapshot(str(database))

    assert exit_code == 0
    assert before == after

    out = capsys.readouterr().out
    assert "https://github.com/blue" in out
    # A site where the username was not found is not an account.
    assert "reddit.com/u/blue" not in out


async def test_show_reports_unknown_username(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    exit_code = await run_show(["never-scanned", "--no-color"])

    assert exit_code == 1
    assert "Nothing stored for" in capsys.readouterr().out


async def test_show_json_is_parseable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Regression: JSON output must survive the terminal.

    Two separate bugs produced invalid output here. Routing it through the
    reporter word-wrapped long URLs and inserted newlines inside JSON strings,
    and emitting raw non-ASCII crashed on Windows, where redirected stdout
    encodes as cp1252.
    """
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    exit_code = await run_show(["blue", "--json"])

    assert exit_code == 0
    payload = capsys.readouterr().out
    parsed = json.loads(payload)

    assert payload.isascii()
    assert parsed[0]["username"] == "blue"
    assert parsed[0]["accounts_found"] == 1
    assert parsed[0]["sites_checked"] == 2
    assert parsed[0]["profile"]["strong_profile"] == {"display_name": ["Tony"]}


async def test_show_accounts_only_omits_the_profile(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["blue", "--accounts", "--no-color"])

    out = capsys.readouterr().out
    assert "https://github.com/blue" in out
    assert "AI profile" not in out


async def test_show_points_at_the_rebuild_when_no_profile_is_stored(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    database = tmp_path / "sherlock.db"
    await _seed(str(database), with_profile=False)
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["blue", "--profile", "--no-color"])

    assert "--ai-synthesize-only" in capsys.readouterr().out


async def test_username_overview_counts_only_that_username(tmp_path):
    database = tmp_path / "sherlock.db"
    await _seed(str(database))

    db = await SherlockDB.create(str(database))
    try:
        await db.save_result(
            username="green",
            site_name="GitHub",
            site_url="https://github.com/green",
            status=str(QueryStatus.CLAIMED),
        )
        blue = await db.get_username_overview("blue")
        green = await db.get_username_overview("green")
        missing = await db.get_username_overview("absent")
    finally:
        await db.close()

    assert blue.total_sites == 2
    assert blue.claimed_sites == 1
    assert green.total_sites == 1
    assert missing is None


async def test_show_displays_the_anchors_a_profile_was_built_from(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Reading back an anchored profile must include what it was anchored to."""
    profile = ProfileSynthesis(
        username="blue",
        input_hash="hash-1",
        mode="anchored",
        resolution_status="resolved",
        completeness="complete",
        strong_profile={"full_name": ["Avery Stone"]},
        anchors=[
            IdentityAnchor(field="name", value="Avery Stone"),
            IdentityAnchor(field="roles", value="Hacker", trust="context"),
        ],
    )

    database = tmp_path / "sherlock.db"
    await _seed(str(database), with_profile=False)
    db = await SherlockDB.create(str(database))
    try:
        await db.update_username_profile_summary(
            username="blue",
            profile_summary=json.dumps(profile.model_dump(mode="json"), sort_keys=True),
            input_hash="hash-1",
        )
    finally:
        await db.close()
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["blue", "--profile", "--no-color"])

    out = capsys.readouterr().out
    assert "anchors" in out
    assert "name=Avery Stone" in out
    assert "roles=Hacker [context]" in out
