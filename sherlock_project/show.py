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
        "--unresolved",
        action="store_true",
        default=False,
        help=(
            "List the sites that gave no answer -- inconclusive, blocked by "
            "bot protection, or rejecting the username format. These are not "
            "the same as sites where the username was absent."
        ),
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
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help=(
            "Include the diagnostic notes recorded while the profile was "
            "built. They name internal site ids and model failures, so they "
            "are summarised as a count by default."
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


# The stored transport that did NOT run a browser. "browser" and "api" both
# go through one; "http" is the plain request, which runs no JavaScript.
_BROWSERLESS_TRANSPORT = "http"


def _transport_note(transport: str | None) -> str | None:
    """The words that go beside a result fetched without a browser.

    Only the browserless case is annotated. Labelling every browser result too
    would make the exception invisible, which is the opposite of the point. A
    NULL transport is a row written before the column existed -- unknown, which
    is not the same as "no browser", so it is left unmarked rather than guessed.
    """
    return "no browser" if transport == _BROWSERLESS_TRANSPORT else None


def _claimed_accounts(saved_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored rows for sites where the username was found, name-sorted."""
    claimed = [
        {
            "site_name": site_name,
            "url": row.get("site_url") or "",
            "confidence": row.get("confidence"),
            "transport": row.get("transport"),
            "scanned_at": row.get("scanned_at"),
        }
        for site_name, row in saved_rows.items()
        if row.get("status") == str(QueryStatus.CLAIMED)
    ]
    return sorted(claimed, key=lambda item: item["site_name"].lower())


# Stored statuses that mean "no answer", mapped to the reason a reader needs.
# AVAILABLE is deliberately absent: it is a real answer. CLAIMED likewise.
_UNRESOLVED_REASONS: dict[str, str] = {
    str(QueryStatus.UNKNOWN): "inconclusive",
    str(QueryStatus.WAF): "blocked by bot protection",
    str(QueryStatus.ILLEGAL): "username format rejected",
}


def _unresolved_sites(
    saved_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stored rows where the check produced no usable answer, name-sorted.

    Kept separate from `_claimed_accounts` because the two answer different
    questions: that one is "what did we find", this one is "what did we fail to
    determine". Merging them is what let silence read as absence in the first
    place.
    """
    unresolved = [
        {
            "site_name": site_name,
            "url": row.get("site_url") or "",
            "status": row.get("status"),
            "reason": _UNRESOLVED_REASONS[str(row.get("status"))],
            "transport": row.get("transport"),
            "context": row.get("error_context") or None,
        }
        for site_name, row in saved_rows.items()
        if str(row.get("status")) in _UNRESOLVED_REASONS
    ]
    return sorted(unresolved, key=lambda item: item["site_name"].lower())


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


def _extraction_models(counts: dict[str | None, int]) -> list[dict[str, Any]]:
    """Which models produced the stored extractions, largest share first.

    A null model is a row written before the model was recorded, and is
    reported as such rather than dropped: "we do not know" and "there were
    none" are different answers.
    """
    return [
        {"model": model, "count": count}
        for model, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0] or ""),
        )
    ]


def _describe_extraction_models(entries: list[dict[str, Any]]) -> str | None:
    """One line naming the models behind a profile, or None when there are none.

    A single model is named plainly. Several are listed with counts, because a
    profile assembled by more than one model is exactly the case a reader needs
    to notice -- two claims that disagree may simply be two different models.
    """
    if not entries:
        return None

    def label(model: str | None) -> str:
        return model if model is not None else "unrecorded model"

    if len(entries) == 1:
        return label(entries[0]["model"])
    return ", ".join(
        f"{entry['count']} {label(entry['model'])}" for entry in entries
    )


async def _collect(db: SherlockDB, username: str) -> dict[str, Any]:
    overview = await db.get_username_overview(username)
    if overview is None:
        return {"username": username, "known": False}

    saved_rows = await db.get_saved_results(username)
    cache = await db.get_profile_summary_cache(username)
    raw_summary = cache.profile_summary if cache is not None else None
    model_counts = await db.get_extraction_model_counts(username)

    return {
        "username": username,
        "known": True,
        "last_scanned_at": overview.last_scanned_at,
        "sites_checked": overview.total_sites,
        "accounts_found": overview.claimed_sites,
        "accounts": _claimed_accounts(saved_rows),
        "unresolved": _unresolved_sites(saved_rows),
        "extraction_models": _extraction_models(model_counts),
        "profile_updated_at": cache.updated_at if cache is not None else None,
        "profile": _load_profile(raw_summary),
        "profile_unreadable": bool(raw_summary) and _load_profile(raw_summary) is None,
    }


def _report_unresolved(
    reporter: TerminalReporter,
    record: dict[str, Any],
    *,
    listing: bool,
) -> None:
    """Summarise, and optionally list, the sites that gave no answer.

    Summarised whenever accounts are shown and listed only on request, for the
    same reason the scan does it that way: the count has to be unmissable
    because it changes what the account list means, while several hundred site
    names would bury the accounts the user came to read.
    """
    unresolved = record["unresolved"]
    if not unresolved:
        return

    username = record["username"]
    counts: dict[str, int] = {}
    for entry in unresolved:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    # Fixed order, not alphabetical: it has to match the scan's summary, and
    # sorting by name puts a stray "blocked" ahead of a much larger
    # "inconclusive" for no reason a reader can see.
    breakdown = ", ".join(
        f"{counts[reason]} {reason}"
        for reason in _UNRESOLVED_REASONS.values()
        if counts.get(reason)
    )

    site_word = "site" if len(unresolved) == 1 else "sites"
    reporter.warning(
        f"{len(unresolved)} {site_word} of {record['sites_checked']} gave no "
        f"answer: {breakdown}"
    )
    if not listing:
        reporter.hint(
            f'Not the same as "not found". List them: sherlock show '
            f"{username} --unresolved"
        )
        return

    reporter.hint(
        'These were never determined. Absence here is not evidence of absence.'
    )
    for entry in unresolved:
        detail = entry["reason"]
        # Before the context, not after: "no browser" is often the whole
        # explanation for an inconclusive result, and burying it behind a
        # timeout message hides the cause behind the symptom.
        note = _transport_note(entry["transport"])
        if note:
            detail = f"{detail}; {note}"
        if entry["context"]:
            detail = f"{detail}; {entry['context']}"
        reporter.warning(f"{entry['site_name']}: {entry['url']}", detail=detail)


def _report(
    reporter: TerminalReporter,
    record: dict[str, Any],
    *,
    want_accounts: bool,
    want_profile: bool,
    want_unresolved: bool = False,
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
            # A hit found without a browser is weaker evidence than one found
            # with it, and months later this is the only place that says so.
            reporter.success(
                f"{account['site_name']}: {account['url']}{qualifier}",
                detail=_transport_note(account["transport"]),
            )

    if want_accounts or want_unresolved:
        _report_unresolved(reporter, record, listing=want_unresolved)

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
            detail = f"built {record['profile_updated_at']}"
            models = _describe_extraction_models(record["extraction_models"])
            if models is not None:
                detail = f"{detail} · extracted by {models}"
            reporter.info(f"AI profile for {username!r}", detail=detail)
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

    # No narrowing flag means show everything; several mean the union of them.
    # --unresolved only ever LISTS on request: the count still surfaces with the
    # accounts, because a hit list whose coverage is unstated is the defect this
    # flag exists to fix.
    narrowed = args.accounts or args.profile or args.unresolved
    want_accounts = args.accounts or not narrowed
    want_profile = args.profile or not narrowed

    reporter = TerminalReporter(no_color=args.no_color, verbose=args.verbose)
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
            want_unresolved=args.unresolved,
            show_sources=args.sources,
        )

    # Exit 1 when nothing at all was found, so scripts can branch on it.
    return 0 if any(record["known"] for record in records) else 1
