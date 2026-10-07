#!/usr/bin/env python
"""Measure Gemini on Pass 1 against what the local model already stored.

The gate before building batching (see the cloud plan, phase 0): before any
decision about packing several sites into one request, find out what one site
per request actually looks like on Gemini, on pages this machine has already
scanned. Five questions, each answered with a number:

  1. Quality  -- per site, which values Gemini and the stored local extraction
                 agree on, and which only one of them found.
  2. Size     -- real page sizes after content extraction, in tokens as the
                 provider counts them. Batching arithmetic depends on these.
  3. Blocks   -- how often the content filter refuses a page. NSFW and dating
                 sites are in the scan set; a high rate changes the design.
  4. Schema   -- whether Gemini accepted the enforced JSON schema, or the
                 provider had to fall back to plain JSON mode.
  5. Thinking -- whether `reasoning_effort` was accepted.

Read-only: the database is opened with `mode=ro`, so nothing is written, no
cached extraction is replaced, and the local results stay the comparison.

Cost: one request per site with content, paced under --rpm. A typical
username is 30-80 sites, well inside a day's free quota.

Examples:

    export GEMINI_API_KEY=...
    python devel/gemini_spike.py --username alice
    python devel/gemini_spike.py --username alice --limit 20 --report spike.json
    SHERLOCK_DB=/path/to/sherlock.db python devel/gemini_spike.py -u alice -u bob
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sherlock_project.ai_config import DEFAULT_GEMINI_BASE_URL, AISettings
from sherlock_project.ai_engine import AIRequestTrace, AIService
from sherlock_project.ai_provider import (
    AIQuotaExhaustedError,
    OpenAICompatibleProvider,
)
from sherlock_project.content_extraction import extract_profile_content
from sherlock_project.database import default_database_path
from sherlock_project.profile_synthesis import CANONICAL_PROFILE_FIELDS
from sherlock_project.result import QueryStatus


def _rows(db_path: Path, usernames: list[str], limit: int | None) -> list[sqlite3.Row]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in usernames)
    query = f"""
        SELECT r.id, u.username, r.site_name, r.site_url, r.response_text,
               r.ai_extraction, r.ai_extraction_model
        FROM results r JOIN usernames u ON u.id = r.username_id
        WHERE u.username IN ({placeholders})
          AND r.status = ?
          AND NULLIF(TRIM(r.response_text), '') IS NOT NULL
        ORDER BY u.username, r.id
    """
    try:
        rows = connection.execute(
            query, (*usernames, str(QueryStatus.CLAIMED))
        ).fetchall()
    finally:
        connection.close()
    return rows[:limit] if limit else rows


def _values(extraction: dict[str, list[str]] | None) -> set[tuple[str, str]]:
    """(key, value) pairs, case- and space-folded, for comparison only."""
    pairs: set[tuple[str, str]] = set()
    for key, values in (extraction or {}).items():
        for value in values if isinstance(values, list) else [values]:
            pairs.add((key, " ".join(str(value).split()).casefold()))
    return pairs


def _plain_values(pairs: set[tuple[str, str]]) -> set[str]:
    # Agreement on the VALUE regardless of key: two models filing "Berlin"
    # under `location` and `city` found the same fact and named it differently.
    return {value for _, value in pairs}


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


async def run(arguments: argparse.Namespace) -> int:
    db_path = Path(arguments.db) if arguments.db else default_database_path()
    if not db_path.exists():
        print(f"[x] No database at {db_path}. Scan first, or pass --db.")
        return 2
    rows = _rows(db_path, arguments.username, arguments.limit)
    if not rows:
        print("[x] No claimed sites with stored page text for those usernames.")
        return 2

    settings = AISettings(
        provider="gemini",
        base_url=arguments.base_url,
        model=arguments.model,
        requests_per_minute=arguments.rpm,
        temperature=0.1,
    )
    traces: list[AIRequestTrace] = []
    service = await AIService.create(settings, trace_callback=traces.append)
    provider = service._provider
    assert isinstance(provider, OpenAICompatibleProvider)

    sites: list[dict[str, object]] = []
    errors: Counter[str] = Counter()
    print(f"[*] {len(rows)} sites, {arguments.model}, {arguments.rpm} requests/minute")
    try:
        for index, row in enumerate(rows, start=1):
            content = extract_profile_content(
                row["response_text"],
                searched_username=row["username"],
                site_name=row["site_name"],
                site_url=row["site_url"],
            )
            local = json.loads(row["ai_extraction"]) if row["ai_extraction"] else None
            record: dict[str, object] = {
                "site": row["site_name"],
                "username": row["username"],
                "content_chars": len(content),
                "local_model": row["ai_extraction_model"],
                "local": local,
            }
            if not content:
                record["skipped"] = "no content after extraction"
                sites.append(record)
                continue
            traces.clear()
            try:
                response = await service.extract_profile(
                    username=row["username"],
                    site_name=row["site_name"],
                    site_content=content,
                    known_profile_keys=list(CANONICAL_PROFILE_FIELDS),
                )
            except AIQuotaExhaustedError as error:
                print(f"[!] {error} Stopping after {index - 1} sites.")
                errors["AIQuotaExhaustedError"] += 1
                break
            except Exception as error:  # measured, not handled
                errors[type(error).__name__] += 1
                record["error"] = f"{type(error).__name__}: {error}"
                print(f"    {index:>3} {row['site_name']}: {record['error']}")
            else:
                gemini = response.extraction
                record["gemini"] = gemini
                record["gemini_reasoning"] = getattr(response, "reasoning", None)
                if local is not None:
                    g, l = _plain_values(_values(gemini)), _plain_values(_values(local))
                    record["agree"] = sorted(g & l)
                    record["only_gemini"] = sorted(g - l)
                    record["only_local"] = sorted(l - g)
                print(
                    f"    {index:>3} {row['site_name']}: "
                    f"{sum(len(v) for v in gemini.values())} values"
                    + (
                        f" (agree {len(record['agree'])}, +{len(record['only_gemini'])}"
                        f" gemini, +{len(record['only_local'])} local)"
                        if "agree" in record
                        else ""
                    )
                )
            if traces:
                trace = traces[-1]
                record["prompt_tokens"] = trace.stats.input_tokens
                record["output_tokens"] = trace.stats.output_tokens
                record["reasoning_tokens"] = trace.stats.reasoning_tokens
                record["elapsed_seconds"] = round(trace.elapsed_seconds, 2)
            sites.append(record)
    finally:
        await service.close()

    answered = [site for site in sites if "gemini" in site]
    compared = [site for site in answered if "agree" in site]
    prompt_tokens = [int(s["prompt_tokens"]) for s in sites if s.get("prompt_tokens")]
    summary = {
        "model": arguments.model,
        "sites": len(sites),
        "sent": sum(1 for s in sites if "skipped" not in s),
        "answered": len(answered),
        "errors": dict(errors),
        "content_filter_blocks": errors.get("AIContentBlockedError", 0),
        "structured_output_mode": provider.structured_output_mode,
        "reasoning_effort_accepted": provider.reasoning_effort_supported,
        "prompt_tokens_p50": _percentile(prompt_tokens, 0.5),
        "prompt_tokens_p90": _percentile(prompt_tokens, 0.9),
        "prompt_tokens_max": max(prompt_tokens, default=0),
        "mean_seconds": round(
            statistics.mean(float(s["elapsed_seconds"]) for s in answered), 2
        )
        if answered
        else None,
        "values_agreed": sum(len(s["agree"]) for s in compared),
        "values_only_gemini": sum(len(s["only_gemini"]) for s in compared),
        "values_only_local": sum(len(s["only_local"]) for s in compared),
    }
    print()
    for key, value in summary.items():
        print(f"{key:>28}: {value}")
    print(
        "\nonly_gemini / only_local are disagreements, not errors: read the "
        "per-site lists in the report to judge which side was right."
    )
    if arguments.report:
        Path(arguments.report).write_text(
            json.dumps({"summary": summary, "sites": sites}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[*] Per-site report: {arguments.report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-u", "--username", action="append", required=True)
    parser.add_argument("--db", help="Database path. Defaults to SHERLOCK_DB or the usual location.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--base-url", default=DEFAULT_GEMINI_BASE_URL)
    parser.add_argument("--rpm", type=int, default=10, help="Requests per minute (default 10).")
    parser.add_argument("--limit", type=int, help="Only the first N sites.")
    parser.add_argument("--report", help="Write per-site results here as JSON.")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
