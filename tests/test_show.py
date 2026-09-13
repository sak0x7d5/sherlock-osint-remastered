"""Tests for `sherlock-rm show`, the read-only view of stored results.

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
from sherlock_project.show import (
    _claimed_accounts,
    _describe_extraction_models,
    _load_profile,
    _transport_note,
    _unresolved_sites,
    run_show,
)


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


def test_transport_note_marks_only_the_browserless_case():
    """Marking every result would make the one that matters invisible.

    NULL is a row written before the column existed. "Unknown" must not be
    reported as "no browser" -- that would put a warning on evidence that was
    very likely collected with one.
    """
    assert _transport_note("http") == "no browser"
    assert _transport_note("browser") is None
    assert _transport_note("api") is None
    assert _transport_note(None) is None


def test_claimed_accounts_carry_their_transport():
    rows = {
        "GitHub": {
            "status": "Claimed",
            "site_url": "https://github.com/blue",
            "transport": "http",
        },
    }

    assert _claimed_accounts(rows)[0]["transport"] == "http"


async def test_show_marks_accounts_found_without_a_browser(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """A hit found without a browser is weaker evidence, and must say so.

    Once the row is stored, this is the only surface that can still tell the
    reader how the answer was obtained.
    """
    database = tmp_path / "sherlock.db"
    db = await SherlockDB.create(str(database))
    try:
        await db.save_result(
            username="blue",
            site_name="GitHub",
            site_url="https://github.com/blue",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            transport="http",
        )
        await db.save_result(
            username="blue",
            site_name="Zulip",
            site_url="https://z.example/blue",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            transport="browser",
        )
    finally:
        await db.close()
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    exit_code = await run_show(["blue", "--accounts"])

    assert exit_code == 0
    lines = capsys.readouterr().out.splitlines()
    github = next(line for line in lines if "GitHub" in line)
    zulip = next(line for line in lines if "Zulip" in line)
    assert "no browser" in github
    assert "no browser" not in zulip


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


async def _seed_unresolved(path: str) -> None:
    """A username whose scan left real answers and real gaps."""
    db = await SherlockDB.create(path)
    try:
        await db.save_result(
            username="grey",
            site_name="Found",
            site_url="https://found.example/grey",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
        )
        await db.save_result(
            username="grey",
            site_name="Absent",
            site_url="https://absent.example/grey",
            status=str(QueryStatus.AVAILABLE),
            status_code=404,
        )
        await db.save_result(
            username="grey",
            site_name="Timeouts",
            site_url="https://timeouts.example/grey",
            status=str(QueryStatus.UNKNOWN),
            error_context="Timeout Error",
        )
        await db.save_result(
            username="grey",
            site_name="Guarded",
            site_url="https://guarded.example/grey",
            status=str(QueryStatus.WAF),
        )
    finally:
        await db.close()


def test_unresolved_sites_excludes_real_answers():
    """Only genuine non-answers. Absent IS an answer and must not appear here."""
    rows = {
        "Hit": {"status": "Claimed", "site_url": "https://h.example/x"},
        "Gone": {"status": "Available", "site_url": "https://g.example/x"},
        "Zed": {"status": "WAF", "site_url": "https://z.example/x"},
        "Alpha": {"status": "Unknown", "site_url": "https://a.example/x"},
    }

    unresolved = _unresolved_sites(rows)

    assert [entry["site_name"] for entry in unresolved] == ["Alpha", "Zed"]
    assert [entry["reason"] for entry in unresolved] == [
        "inconclusive",
        "blocked by bot protection",
    ]


async def test_show_warns_about_unresolved_sites_without_listing_them(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The count rides along with the accounts; the names wait to be asked for."""
    database = tmp_path / "sherlock.db"
    await _seed_unresolved(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["grey", "--accounts", "--no-color"])

    out = capsys.readouterr().out
    assert "2 sites of 4 gave no answer" in out
    assert "1 inconclusive" in out
    assert "1 blocked by bot protection" in out
    assert "grey --unresolved" in out
    # Not listed unless asked: a hit list must not be buried by its own caveat.
    assert "https://timeouts.example/grey" not in out


async def test_show_unresolved_lists_each_site_with_its_reason(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    database = tmp_path / "sherlock.db"
    await _seed_unresolved(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["grey", "--unresolved", "--no-color"])

    out = capsys.readouterr().out
    assert "https://timeouts.example/grey" in out
    assert "Timeout Error" in out
    assert "https://guarded.example/grey" in out
    assert "blocked by bot protection" in out
    # Narrowed to the gaps: neither the hits nor the profile come along.
    assert "https://found.example/grey" not in out
    assert "AI profile" not in out


async def test_show_stays_quiet_when_every_site_resolved(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """No gaps, no caveat -- the warning must mean something when it appears."""
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["blue", "--accounts", "--no-color"])

    assert "gave no answer" not in capsys.readouterr().out


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
    # Its own section, headed and shaped like CONFIDENT beside it.
    assert "ANCHORS" in out
    assert "anchored to" not in out
    assert "name=Avery Stone" not in out
    for text in ("name", "Avery Stone", "roles", "Hacker", "context"):
        assert text in out


async def test_show_sources_flag_switches_to_full_urls(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Compact by default; the URLs are one flag away, never lost."""
    profile = ProfileSynthesis.model_validate(
        {
            "username": "blue",
            "input_hash": "hash-1",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "provenance": [
                {
                    "field": "full_name",
                    "value": "Avery Stone",
                    "source_site_ids": [1],
                    "origins": ["extraction"],
                }
            ],
            "source_decisions": [
                {
                    "site_id": 1,
                    "site_name": "Mastodon",
                    "site_url": "https://mastodon.social/@avery",
                    "disposition": "aggregated",
                }
            ],
        }
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
    compact = capsys.readouterr().out
    assert "1 site: Mastodon" in compact
    assert "mastodon.social" not in compact

    await run_show(["blue", "--profile", "--sources", "--no-color"])
    verbose = capsys.readouterr().out.replace("\n", "")
    assert "https://mastodon.social/@avery" in verbose


def test_describe_extraction_models_names_one_and_counts_several():
    """One model reads as a name; several have to read as a mixture."""
    assert _describe_extraction_models([]) is None
    assert (
        _describe_extraction_models([{"model": "vendor/small", "count": 12}])
        == "vendor/small"
    )
    assert _describe_extraction_models(
        [
            {"model": "vendor/small", "count": 12},
            {"model": "vendor/large", "count": 3},
            {"model": None, "count": 1},
        ]
    ) == "12 vendor/small, 3 vendor/large, 1 unrecorded model"


async def test_show_reports_which_models_produced_the_extractions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """A profile assembled by two models must say so on the profile line."""
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    db = await SherlockDB.create(str(database))
    try:
        rows = await db.get_saved_results("blue")
        assert "GitHub" in rows
        site_id = await db.save_result(
            username="blue",
            site_name="Keybase",
            site_url="https://keybase.io/blue",
            status=str(QueryStatus.CLAIMED),
            response_text="profile",
        )
        await db.update_result_ai_extraction(
            site_id=site_id,
            ai_extraction='{"full_name": ["Blue"]}',
            contract_hash="contract-1",
            model_key="vendor/large",
        )
        other_id = await db.save_result(
            username="blue",
            site_name="Mastodon",
            site_url="https://m.example/@blue",
            status=str(QueryStatus.CLAIMED),
            response_text="profile",
        )
        await db.update_result_ai_extraction(
            site_id=other_id,
            ai_extraction="{}",
            contract_hash="contract-1",
            model_key="vendor/small",
        )
    finally:
        await db.close()
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    exit_code = await run_show(["blue", "--profile", "--no-color"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "extracted by" in output
    assert "vendor/large" in output
    assert "vendor/small" in output


async def test_show_json_carries_model_provenance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    database = tmp_path / "sherlock.db"
    await _seed(str(database))
    db = await SherlockDB.create(str(database))
    try:
        site_id = await db.save_result(
            username="blue",
            site_name="Keybase",
            status=str(QueryStatus.CLAIMED),
            response_text="profile",
        )
        await db.update_result_ai_extraction(
            site_id=site_id,
            ai_extraction='{"full_name": ["Blue"]}',
            contract_hash="contract-1",
            model_key="vendor/large",
        )
    finally:
        await db.close()
    monkeypatch.setenv("SHERLOCK_DB", str(database))

    await run_show(["blue", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload[0]["extraction_models"] == [
        {"model": "vendor/large", "count": 1}
    ]
