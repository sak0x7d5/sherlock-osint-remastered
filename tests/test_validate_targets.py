import re

import pytest
import rstr

from sherlock_project.notify import QueryNotify
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.result import QueryResult, QueryStatus
from sherlock_project.sherlock import sherlock

FALSE_POSITIVE_ATTEMPTS: int = 2    # Since the usernames are randomly generated, it's POSSIBLE that a real username can be hit
FALSE_POSITIVE_QUANTIFIER_UPPER_BOUND: int = 15  # If a pattern uses quantifiers such as `+` `*` or `{n,}`, limit the upper bound (0 to disable)
FALSE_POSITIVE_DEFAULT_PATTERN: str = r'^[a-zA-Z0-9]{7,20}$'  # Used in absence of a regexCheck entry


def set_pattern_upper_bound(pattern: str, upper_bound: int = FALSE_POSITIVE_QUANTIFIER_UPPER_BOUND) -> str:
    """Set upper bound for regex patterns that use quantifiers such as `+` `*` or `{n,}`."""
    def replace_upper_bound(match: re.Match) -> str: # type: ignore
        lower_bound: int = int(match.group(1)) if match.group(1) else 0 # type: ignore
        nonlocal upper_bound
        upper_bound = max(lower_bound, upper_bound) # type: ignore
        return f'{{{lower_bound},{upper_bound}}}'

    pattern = re.sub(r'(?<!\\)\{(\d+),\}', replace_upper_bound, pattern) # {n,} # type: ignore
    pattern = re.sub(r'(?<!\\)\+', f'{{1,{upper_bound}}}', pattern) # +
    pattern = re.sub(r'(?<!\\)\*', f'{{0,{upper_bound}}}', pattern) # *

    return pattern

async def false_positive_check(sites_info: dict[str, dict[str, str]], site: str, pattern: str, playwright_engine: PlaywrightEngine, db) -> QueryStatus:
    """Check if a site is likely to produce false positives."""
    status: QueryStatus = QueryStatus.UNKNOWN

    for _ in range(FALSE_POSITIVE_ATTEMPTS):
        query_notify: QueryNotify = QueryNotify()
        username: str = rstr.xeger(pattern)

        result: QueryResult | str = (await sherlock(
            username=username,
            site_data=sites_info,
            query_notify=query_notify,
            engine=playwright_engine,
            db=db
        ))[site]['status']

        if not hasattr(result, 'status'):
            raise TypeError(f"Result for site {site} does not have 'status' attribute. Actual result: {result}")
        if type(result.status) is not QueryStatus: # type: ignore
            raise TypeError(f"Result status for site {site} is not of type QueryStatus. Actual type: {type(result.status)}") # type: ignore
        status = result.status # type: ignore

        if status in (QueryStatus.AVAILABLE, QueryStatus.WAF):
            return status

    return status


async def false_negative_check(sites_info: dict[str, dict[str, str]], site: str, playwright_engine: PlaywrightEngine, db) -> QueryStatus:
    """Check if a site is likely to produce false negatives."""
    status: QueryStatus = QueryStatus.UNKNOWN
    query_notify: QueryNotify = QueryNotify()

    result: QueryResult | str = (await sherlock(
        username=sites_info[site]['username_claimed'],
        db=db,
        site_data=sites_info,
        query_notify=query_notify,
        engine=playwright_engine
    ))[site]['status']

    if not hasattr(result, 'status'):
            raise TypeError(f"Result for site {site} does not have 'status' attribute. Actual result: {result}")
    if type(result.status) is not QueryStatus: # type: ignore
        raise TypeError(f"Result status for site {site} is not of type QueryStatus. Actual type: {type(result.status)}") # type: ignore
    status = result.status # type: ignore

    return status

@pytest.mark.validate_targets
@pytest.mark.online
class Test_All_Targets:

    @pytest.mark.validate_targets_fp
    @pytest.mark.asyncio
    async def test_false_pos(self, chunked_sites: dict[str, dict[str, str]], playwright_engine, db):
        """Iterate through all sites in the manifest to discover possible false-positive inducting targets."""
        pattern: str
        for site in chunked_sites:
            try:
                pattern = chunked_sites[site]['regexCheck']
            except KeyError:
                pattern = FALSE_POSITIVE_DEFAULT_PATTERN

            if FALSE_POSITIVE_QUANTIFIER_UPPER_BOUND > 0:
                pattern = set_pattern_upper_bound(pattern)

            result: QueryStatus = await false_positive_check(chunked_sites, site, pattern, playwright_engine, db)
            # Only a claim is a false positive. Refusing to answer -- a block, or
            # a response matching neither side of the rule -- is the designed
            # behaviour and must not fail the sweep, or this becomes a report on
            # the network rather than on the site rules.
            assert result is not QueryStatus.CLAIMED, f"{site} produced false positive with pattern {pattern}, result was {result}"

    @pytest.mark.validate_targets_fn
    @pytest.mark.asyncio
    async def test_false_neg(self, chunked_sites: dict[str, dict[str, str]], playwright_engine, db):
        """Iterate through all sites in the manifest to discover possible false-negative inducting targets."""
        for site in chunked_sites:
            result: QueryStatus = await false_negative_check(chunked_sites, site, playwright_engine, db)
            assert result is QueryStatus.CLAIMED, f"{site} produced false negative, result was {result}"

