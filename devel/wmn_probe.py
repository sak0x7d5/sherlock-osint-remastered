#!/usr/bin/env python
"""Drive the WMN adapter and two-sided detection against live sites.

This is a validation harness, not the scan path. The migration is not yet wired
into `sherlock`, so this is how the new pipeline gets exercised end to end while
that work is in progress.

Two modes:

  Accuracy (default) -- probe each site twice, once with a username the dataset
  records as real and once with a random one, and report whether detection got
  both right. This is the measurement that matters: a rule that cannot tell its
  own known-good account from a random string is not usable, and the two-sided
  data is what makes the question answerable at all.

  Lookup (`--username`) -- probe each site once for a real username, the way a
  scan would.

Probes run concurrently, bounded by the engine's semaphore, so grading the whole
dataset costs about what one scan costs. --sample exists to shorten a run, not
because a full one is impractical.

Examples:

    # grade every site in the dataset
    python devel/wmn_probe.py

    # shorter run, both transports compared
    python devel/wmn_probe.py --sample 60 --compare

    # look one username up across the social category
    python devel/wmn_probe.py --username alice --category social

    # drill into specific sites
    python devel/wmn_probe.py --site Instagram --site Keybase --compare
"""

import argparse
import asyncio
import random
import secrets
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sherlock_project.detection import evaluate
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.wmn_adapter import (
    load_wmn_manifest,
    normalize_username,
    preferred_transport,
)

MANIFEST_PATH = REPO_ROOT / "sherlock_project" / "resources" / "wmn-data.json"

# Telegram and friends never reach 'load'; the markers live in the initial HTML.
PAGE_WAIT_UNTIL = "domcontentloaded"


def transport_for(record: dict) -> str:
    """Route exactly as the scan does.

    This used to be a second copy of the rule, which is how it came to disagree
    with the scan without anyone noticing: the harness reported Instagram
    working over the browser while the scan was quietly sending every site to
    the API. A validation harness that does not share the code it validates is
    measuring something else.
    """
    return preferred_transport(record)


async def fetch(engine, record, username, transport):
    username = normalize_username(username, record.get("strip_bad_char"))
    url = record["url"].replace("{}", username)
    headers = record.get("headers") or {}

    # page.goto cannot issue a POST, so those sites are API-only.
    if transport == "browser" and record.get("request_method", "GET") == "GET":
        resp = await engine.fetch_with_page(
            url=url, headers=headers, timeout=45000, wait_until=PAGE_WAIT_UNTIL
        )
    else:
        payload = record.get("request_payload")
        if payload:
            payload = payload.replace("{}", username)
        resp = await engine.fetch_with_api(
            request_fn=engine.get_request_fn(record.get("request_method", "GET")),
            url=url,
            headers=headers,
            timeout=45000,
            request_payload=payload,
        )

    if resp is None:
        return None, None
    return resp.status, resp.text


async def verdict_for(engine, record, username, transport):
    status, body = await fetch(engine, record, username, transport)
    return evaluate(record["detection"], status, body), status, len(body or "")


def select_sites(sites: dict, args) -> list[tuple[str, dict]]:
    chosen = sites
    if args.site:
        wanted = {s.casefold() for s in args.site}
        chosen = {n: r for n, r in chosen.items() if n.casefold() in wanted}
        unknown = wanted - {n.casefold() for n in chosen}
        if unknown:
            print(f"not in dataset: {sorted(unknown)}", file=sys.stderr)
    if args.category:
        cats = {c.casefold() for c in args.category}
        chosen = {n: r for n, r in chosen.items() if r["category"].casefold() in cats}
    if not args.nsfw:
        chosen = {n: r for n, r in chosen.items() if not r["isNSFW"]}

    items = sorted(chosen.items())
    if args.sample and args.sample < len(items):
        items = random.Random(args.seed).sample(items, args.sample)
    return items


async def _grade_one(engine, name, record, transport, primary):
    """Probe one site on one transport with a real and a random username."""
    known = record["known"][0]
    ghost = secrets.token_hex(8)

    try:
        # The two probes are independent, and the engine's own semaphore is what
        # bounds real concurrency, so there is no reason to serialize them.
        (hit, hcode, hlen), (miss, mcode, mlen) = await asyncio.gather(
            verdict_for(engine, record, known, transport),
            verdict_for(engine, record, ghost, transport),
        )
    except Exception as error:
        return {"name": name, "transport": transport, "primary": primary, "error": error}

    return {
        "name": name,
        "transport": transport,
        "primary": primary,
        "known": known,
        "hit": hit, "hcode": hcode, "hlen": hlen,
        "miss": miss, "mcode": mcode, "mlen": mlen,
    }


async def run_accuracy(engine, items, compare: bool) -> Counter:
    """Grade every selected site concurrently.

    Grading used to walk the list one site at a time, which made a full-dataset
    run take hours for no reason -- the scan itself has always been concurrent.
    Every probe is now a task, bounded by the engine's existing semaphore, so
    grading all 719 sites costs about what a scan costs.
    """
    jobs = []
    for name, record in items:
        primary = transport_for(record)
        transports = [primary]
        if compare:
            transports = [primary, "api" if primary == "browser" else "browser"]
        for transport in transports:
            jobs.append(_grade_one(engine, name, record, transport, primary))

    tally: Counter = Counter()
    print(f"{'site':26} {'transport':9} {'known':26} {'random':26}")
    print("-" * 92)

    for completed in asyncio.as_completed(jobs):
        outcome = await completed
        name = outcome["name"]
        transport = outcome["transport"]
        is_primary = transport == outcome["primary"]

        if "error" in outcome:
            if is_primary:
                tally["transport failure"] += 1
            error = outcome["error"]
            print(f"{name:26} {transport:9} FAIL {type(error).__name__}: {str(error)[:40]}")
            continue

        hit, miss = outcome["hit"], outcome["miss"]
        correct = hit.exists is True and miss.exists is False

        # The two ways of being confidently wrong are not equally bad, and
        # lumping them together hides which one a change traded for the other.
        # A fabricated account puts a real person in an investigation they have
        # nothing to do with; a missed one loses a lead. Counted apart so the
        # severe failure can never be averaged away by the mild one.
        false_positive = miss.exists is True
        false_negative = hit.exists is False

        if is_primary:
            if correct:
                tally["correct"] += 1
            elif false_positive:
                tally["FALSE POSITIVE"] += 1
            elif false_negative:
                tally["false negative"] += 1
            else:
                tally["undecided"] += 1

        flag = "OK " if correct else ("!!!" if false_positive else ("-fn" if false_negative else " ? "))
        print(
            f"{name:26} {transport:9} "
            f"{flag} {hit.exists!s:5} {hit.confidence.value:9} {outcome['hcode']!s:4} "
            f"| {miss.exists!s:5} {miss.confidence.value:9} {outcome['mcode']!s:4}"
        )
        if not correct:
            if hit.exists is not True:
                print(f"{'':36}known={outcome['known']!r}: {hit.reason} ({outcome['hlen']}b)")
            if miss.exists is not False:
                print(f"{'':36}random: {miss.reason} ({outcome['mlen']}b)")

    return tally


async def _look_up_one(engine, name, record, username):
    try:
        verdict, code, length = await verdict_for(
            engine, record, username, transport_for(record)
        )
    except Exception as error:
        return {"name": name, "error": error}
    return {"name": name, "record": record, "verdict": verdict, "code": code, "length": length}


async def run_lookup(engine, items, username: str) -> Counter:
    jobs = [_look_up_one(engine, name, record, username) for name, record in items]

    tally: Counter = Counter()
    for completed in asyncio.as_completed(jobs):
        outcome = await completed
        name = outcome["name"]

        if "error" in outcome:
            tally["error"] += 1
            error = outcome["error"]
            print(f"[err ] {name:26} {type(error).__name__}: {str(error)[:45]}")
            continue

        verdict = outcome["verdict"]
        label = {True: "FOUND", False: "  -  ", None: " ??? "}[verdict.exists]
        tally[label.strip() or "?"] += 1
        if verdict.exists is not False:
            url = outcome["record"]["urlProfile"].replace("{}", username)
            print(
                f"[{label}] {name:26} {verdict.confidence.value:9} "
                f"{outcome['code']} {outcome['length']:>8}b  {url}"
            )
    return tally


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", help="Look this username up instead of measuring accuracy.")
    parser.add_argument("--site", action="append", help="Restrict to a site (repeatable).")
    parser.add_argument("--category", action="append", help="Restrict to a category (repeatable).")
    parser.add_argument("--sample", type=int, help="Probe a random subset of this size.")
    parser.add_argument("--seed", type=int, default=0, help="Sample seed, for reproducibility.")
    parser.add_argument("--concurrency", type=int, default=30, help="Matches the scan's default.")
    parser.add_argument("--nsfw", action="store_true", help="Include NSFW targets.")
    parser.add_argument("--compare", action="store_true", help="Probe both transports, to measure the routing split.")
    args = parser.parse_args()

    sites, rejected = load_wmn_manifest(str(MANIFEST_PATH))
    print(f"{len(sites)} sites adapted, {len(rejected)} rejected")

    items = select_sites(sites, args)
    if not items:
        print("nothing selected", file=sys.stderr)
        return 2
    print(f"probing {len(items)}\n")

    async with PlaywrightEngine(concurrency=args.concurrency) as engine:
        if args.username:
            tally = await run_lookup(engine, items, args.username)
        else:
            tally = await run_accuracy(engine, items, args.compare)

    print("\n" + "  ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
