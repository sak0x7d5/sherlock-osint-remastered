"""Read-only view of what the database already holds for a username.

`sherlock show <username>` answers "what do you already know about this
person" without scanning anything and without writing anything. That second
guarantee is the point of the module: before it existed, the only way to see a
pass-two profile was `--ai-synthesize-only`, which rebuilds the profile and
overwrites the stored one, so looking at a profile destroyed it.

Nothing here opens a browser, contacts a site, or loads a model. It reads
SQLite and prints.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.notify import TerminalReporter
from sherlock_project.profile_synthesis import ProfileSynthesis
from sherlock_project.result import QueryStatus


def build_show_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="sherlock show",
        description=(
            "Show what is already stored for a username. Never scans, never "
            "writes."
        ),
    )
    parser.add_argument(
        "username",
        nargs="+",
        metavar="USERNAMES",
        help="One or more usernames to look up in the local database.",
    )
    parser.add_argument(
        "--accounts",
        action="store_true",
        default=False,
        help="Show only the accounts found, not the AI profile.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Show only the AI profile, not the accounts found.",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        default=False,
        help=(
            "Show the full URL of every site backing each profile value "
            "instead of a count and the first few names."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        default=False,
        help="Emit machine-readable JSON instead of the formatted report.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Don't color terminal output.",
    )
    return parser


def _claimed_accounts(saved_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored rows for sites where the username was found, name-sorted."""
    claimed = [
        {
            "site_name": site_name,
            "url": row.get("site_url") or "",
            "confidence": row.get("confidence"),
            "scanned_at": row.get("scanned_at"),
        }
        for site_name, row in saved_rows.items()
        if row.get("status") == str(QueryStatus.CLAIMED)
    ]
    return sorted(claimed, key=lambda item: item["site_name"].lower())


def _load_profile(raw_summary: str | None) -> ProfileSynthesis | None:
    """Parse a stored profile, or None if absent or no longer readable.

    A profile written by an older version of the response model will not
    validate. That is worth reporting as "unreadable" rather than crashing a
    command whose entire job is to let someone look at their data.
    """
    if not raw_summary:
        return None
    try:
        return ProfileSynthesis.model_validate_json(raw_summary)
    except ValidationError:
        return None


async def _collect(db: SherlockDB, username: str) -> dict[str, Any]:
    overview = await db.get_username_overview(username)
    if overview is None:
        return {"username": username, "known": False}

    saved_rows = await db.get_saved_results(username)
    cache = await db.get_profile_summary_cache(username)
    raw_summary = cache.profile_summary if cache is not None else None

    return {
        "username": username,
        "known": True,
        "last_scanned_at": overview.last_scanned_at,
        "sites_checked": overview.total_sites,
        "accounts_found": overview.claimed_sites,
        "accounts": _claimed_accounts(saved_rows),
        "profile_updated_at": cache.updated_at if cache is not None else None,
        "profile": _load_profile(raw_summary),
        "profile_unreadable": bool(raw_summary) and _load_profile(raw_summary) is None,
    }


def _report(
    reporter: TerminalReporter,
    record: dict[str, Any],
    *,
    want_accounts: bool,
    want_profile: bool,
    show_sources: bool = False,
) -> None:
    username = record["username"]

    if not record["known"]:
        reporter.warning(
            f"Nothing stored for {username!r}. "
            f"Scan it first with: sherlock {username}"
        )
        return

    reporter.info(
        f"Stored results for {username!r}",
        detail=(
            f"last scanned {record['last_scanned_at']} · "
            f"{record['sites_checked']} sites checked · nothing written"
        ),
    )

    if want_accounts:
        accounts = record["accounts"]
        if not accounts:
            reporter.warning(
                f"No accounts found for {username!r} across "
                f"{record['sites_checked']} sites checked"
            )
        for account in accounts:
            qualifier = ""
            if account["confidence"] and account["confidence"] != "Confirmed":
                qualifier = f" [{account['confidence']}]"
            reporter.success(f"{account['site_name']}: {account['url']}{qualifier}")

    if want_profile:
        if record["profile_unreadable"]:
            reporter.warning(
                f"A profile is stored for {username!r} but no longer matches the "
                f"current format. Rebuild it with: "
                f"sherlock {username} --ai-synthesize-only"
            )
        elif record["profile"] is None:
            reporter.info(
                f"No AI profile stored for {username!r}. Build one with: "
                f"sherlock {username} --ai-synthesize-only"
            )
        else:
            reporter.info(
                f"AI profile for {username!r}",
                detail=f"built {record['profile_updated_at']}",
            )
            reporter.render_profile(record["profile"], show_sources=show_sources)


def _as_json(records: list[dict[str, Any]]) -> str:
    payload = []
    for record in records:
        entry = dict(record)
        profile = entry.pop("profile", None)
        entry["profile"] = (
            profile.model_dump(mode="json") if profile is not None else None
        )
        payload.append(entry)
    # ensure_ascii is on deliberately. Redirected stdout on Windows encodes as
    # cp1252, and profiles routinely carry names outside it, so emitting raw
    # non-ASCII makes `show --json > file` die with UnicodeEncodeError. Escapes
    # are lossless and every JSON parser decodes them.
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


async def run_show(argv: Sequence[str]) -> int:
    parser = build_show_parser()
    args = parser.parse_args(list(argv))

    # Neither narrowing flag means show everything; both means the same thing.
    want_accounts = args.accounts or not args.profile
    want_profile = args.profile or not args.accounts

    reporter = TerminalReporter(no_color=args.no_color)
    database_path = default_database_path()
    reporter.debug(f"Using database at {database_path}")

    db = await SherlockDB.create(str(database_path))
    try:
        records = [await _collect(db, username) for username in args.username]
    finally:
        await db.close()

    if args.as_json:
        # Deliberately NOT reporter.raw(): that renders through Rich, which
        # word-wraps to the terminal width and will happily insert a newline
        # into the middle of a long URL, producing invalid JSON. Machine output
        # bypasses the reporter; the reporter stays the surface for progress
        # and errors.
        print(_as_json(records))
        return 0

    for record in records:
        _report(
            reporter,
            record,
            want_accounts=want_accounts,
            want_profile=want_profile,
            show_sources=args.sources,
        )

    # Exit 1 when nothing at all was found, so scripts can branch on it.
    return 0 if any(record["known"] for record in records) else 1
