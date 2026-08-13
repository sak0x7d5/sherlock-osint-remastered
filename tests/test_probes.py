import random
import string

import pytest

from sherlock_project.notify import QueryNotify
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.result import QueryStatus
from sherlock_project.sherlock import sherlock

# These tests used to be grouped by errorType -- message, status_code,
# response_url -- because that was how the legacy manifest described a site.
# A WMN rule has no such split: every rule states both what a hit looks like and
# what a miss looks like, so there is one code path and the only questions worth
# asking live are "is a real account found" and "is a made-up one not".

# Sites picked for being stable and unlikely to change shape. The usernames come
# from the dataset's own `known` list rather than being hard-coded here, so a
# handle going stale is fixed by refreshing the manifest instead of editing
# this file.
TRUSTED_SITES = ["GitLab", "Docker Hub (User)", "Keybase", "devRant"]


async def simple_query(
    sites_info: dict, site: str, username: str, playwright_engine: PlaywrightEngine, db
) -> QueryStatus:
    query_notify = QueryNotify()
    site_data: dict = {site: sites_info[site]}
    results_total = await sherlock(
        username=username,
        site_data=site_data,
        db=db,
        query_notify=query_notify,
        engine=playwright_engine,
    )
    return results_total[site]["status"].status


def random_handle(length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


@pytest.mark.online
class TestLiveTargets:
    """Actively test probes against live and trusted targets"""

    @pytest.mark.parametrize("site", TRUSTED_SITES)
    @pytest.mark.asyncio()
    async def test_known_username_is_found(
        self, wmn_sites_info, site, playwright_engine, db
    ):
        username = wmn_sites_info[site]["known"][0]
        status = await simple_query(
            sites_info=wmn_sites_info,
            site=site,
            username=username,
            playwright_engine=playwright_engine,
            db=db,
        )
        assert status is QueryStatus.CLAIMED, (
            f"{site} did not find {username!r}, which the dataset records as real"
        )

    @pytest.mark.parametrize("site,random_len", [
        ("GitLab", 30),
        ("Docker Hub (User)", 30),
        ("Keybase", 30),
        ("Codecademy", 30),
    ])
    @pytest.mark.asyncio()
    async def test_invented_username_is_not_found(
        self, wmn_sites_info, site, random_len, playwright_engine, db
    ):
        """A made-up handle must not come back claimed.

        Retried because a random string can, very occasionally, be a real
        account. UNKNOWN is tolerated -- refusing to answer when a site blocks
        the request is the designed behaviour, and failing the run for it would
        make this test a network weather report.
        """
        num_attempts = 3
        attempted: list[str] = []
        status = QueryStatus.CLAIMED
        for _ in range(num_attempts):
            handle = random_handle(random_len)
            attempted.append(handle)
            status = await simple_query(
                sites_info=wmn_sites_info,
                site=site,
                username=handle,
                playwright_engine=playwright_engine,
                db=db,
            )
            if status is not QueryStatus.CLAIMED:
                break
        assert status is not QueryStatus.CLAIMED, (
            f"{site} claimed every invented username tried: {attempted}"
        )


# Marked online because every test here fetches https://httpbin.org. Without
# the marker these ran in the default tox env -- the push gate -- and could turn
# a green branch red with no code change, which is the failure mode the pinned
# ruff version exists to prevent. Observed 2026-08-12: all four skipped when the
# host was down, then minutes later the host was up but slow (12s first
# response) and status/404 failed with UNKNOWN instead of AVAILABLE.
#
# The marker fixes gate integrity, NOT that 404 result, which is still
# undiagnosed and now only runs under `tox -e online`. UNKNOWN rather than
# AVAILABLE on a plain 404 is the difference between "no account here" and
# "could not tell", so it is worth diagnosing on merit. Unresolved: whether this
# is httpbin flakiness (the host was up but taking 12s on first response when it
# failed) or whether `page.goto` on a bare 404 returns None. Pointing these at a
# local server instead of httpbin would make them hermetic and return them to
# the gate.
@pytest.mark.online
class TestDetectionAgainstControlledResponses:
    """Drive the two sides of a rule against httpbin rather than a real site."""

    @pytest.mark.parametrize("path,expected", [
        ("status/200", QueryStatus.CLAIMED),
        ("status/404", QueryStatus.AVAILABLE),
    ])
    @pytest.mark.asyncio()
    async def test_status_only_rule(
        self, path, expected, playwright_engine, db, httpbin_available
    ):
        site_data = {
            "Test": {
                "url": "https://httpbin.org/{}",
                "urlProfile": "https://httpbin.org/{}",
                "urlMain": "https://httpbin.org",
                "detection": {
                    "exists": {"code": 200, "string": ""},
                    "missing": {"code": 404, "string": ""},
                },
            }
        }
        status = await simple_query(
            sites_info=site_data,
            db=db,
            site="Test",
            username=path,
            playwright_engine=playwright_engine,
        )
        assert status == expected

    @pytest.mark.asyncio()
    async def test_marker_absent_on_expected_code_is_undecided(
        self, playwright_engine, db, httpbin_available
    ):
        """The soft-404 guard: a 200 alone must not stand in for the marker."""
        site_data = {
            "Test": {
                "url": "https://httpbin.org/{}",
                "urlProfile": "https://httpbin.org/{}",
                "urlMain": "https://httpbin.org",
                "detection": {
                    "exists": {"code": 200, "string": "a-marker-that-is-not-there"},
                    "missing": {"code": 404, "string": ""},
                },
            }
        }
        status = await simple_query(
            sites_info=site_data,
            db=db,
            site="Test",
            username="status/200",
            playwright_engine=playwright_engine,
        )
        assert status is QueryStatus.UNKNOWN


@pytest.mark.asyncio()
async def test_username_illegal_regex(playwright_engine, db):
    """regexCheck still short-circuits a probe when a rule carries one.

    No WMN rule does today -- the dataset has no equivalent field -- so this
    guards the code path rather than any live target, and makes no request.
    """
    site_data = {
        "Test": {
            "url": "https://example.com/{}",
            "urlProfile": "https://example.com/{}",
            "urlMain": "https://example.com",
            "regexCheck": "^[a-zA-Z0-9]+$",
            "detection": {
                "exists": {"code": 200, "string": "profile"},
                "missing": {"code": 404, "string": ""},
            },
        }
    }
    status = await simple_query(
        sites_info=site_data,
        site="Test",
        username="*#$Y&*JRE",
        playwright_engine=playwright_engine,
        db=db,
    )
    assert status is QueryStatus.ILLEGAL
