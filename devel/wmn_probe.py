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

Examples:

    # accuracy over a random 60-site sample, both transports compared
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
)

MANIFEST_PATH = REPO_ROOT / "sherlock_project" / "resources" / "wmn-data.json"

# Telegram and friends never reach 'load'; the markers live in the initial HTML.
PAGE_WAIT_UNTIL = "domcontentloaded"


def transport_for(record: dict) -> str:
    """Pick a transport the way the planned routing will.

    An API endpoint is cheap and safe to trust. An HTML page is not: a login
    wall or block page renders as a normal 200 and can contain the rule's miss
    marker, which is how Instagram produced a *confident* wrong answer over the
    raw API path. uri_pretty is the dataset's own signal for which is which --
    when it is present, uri_check is an API; when absent, uri_check is the page.
    """
    if record.get("protection"):
        return "browser"
    if record["urlProfile"] != record["url"]:
        return "api"
    return "browser"


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


async def run_accuracy(engine, items, compare: bool) -> Counter:
    tally: Counter = Counter()
    print(f"{'site':26} {'transport':9} {'known':26} {'random':26}")
    print("-" * 92)

    for name, record in items:
        primary = transport_for(record)
        known = record["known"][0]
        ghost = secrets.token_hex(8)

        transports = [primary]
        if compare:
            transports = ["api", "browser"] if primary == "browser" else ["browser", "api"]
            transports = sorted(set(transports), key=lambda t: t != primary)

        for transport in transports:
            try:
                hit, hcode, hlen = await verdict_for(engine, record, known, transport)
                miss, mcode, mlen = await verdict_for(engine, record, ghost, transport)
            except Exception as error:
                tally["transport failure"] += 1
                print(f"{name:26} {transport:9} FAIL {type(error).__name__}: {str(error)[:40]}")
                continue

            correct = hit.exists is True and miss.exists is False
            # A confident wrong answer is the worst outcome -- worse than an
            # honest refusal -- so it is counted separately.
            confident_wrong = (hit.exists is False) or (miss.exists is True)

            if transport == primary:
                if correct:
                    tally["correct"] += 1
                elif confident_wrong:
                    tally["CONFIDENT WRONG"] += 1
                else:
                    tally["undecided"] += 1

            flag = "OK " if correct else ("!!!" if confident_wrong else " ? ")
            print(
                f"{name:26} {transport:9} "
                f"{flag} {hit.exists!s:5} {hit.confidence.value:9} {hcode!s:4} "
                f"| {miss.exists!s:5} {miss.confidence.value:9} {mcode!s:4}"
            )
            if not correct:
                if hit.exists is not True:
                    print(f"{'':36}known={known!r}: {hit.reason} ({hlen}b)")
                if miss.exists is not False:
                    print(f"{'':36}random: {miss.reason} ({mlen}b)")

    return tally


async def run_lookup(engine, items, username: str) -> Counter:
    tally: Counter = Counter()
    for name, record in items:
        transport = transport_for(record)
        try:
            verdict, code, length = await verdict_for(engine, record, username, transport)
        except Exception as error:
            tally["error"] += 1
            print(f"[err ] {name:26} {type(error).__name__}: {str(error)[:45]}")
            continue

        label = {True: "FOUND", False: "  -  ", None: " ??? "}[verdict.exists]
        tally[label.strip() or "?"] += 1
        if verdict.exists is not False:
            url = record["urlProfile"].replace("{}", username)
            print(f"[{label}] {name:26} {verdict.confidence.value:9} {code} {length:>8}b  {url}")
    return tally


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", help="Look this username up instead of measuring accuracy.")
    parser.add_argument("--site", action="append", help="Restrict to a site (repeatable).")
    parser.add_argument("--category", action="append", help="Restrict to a category (repeatable).")
    parser.add_argument("--sample", type=int, help="Probe a random subset of this size.")
    parser.add_argument("--seed", type=int, default=0, help="Sample seed, for reproducibility.")
    parser.add_argument("--concurrency", type=int, default=8)
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
