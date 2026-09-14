"""Deterministic Pass 1 sanitation, applied after the model has answered.

Separate from `test_ai_engine.py` because every rule here is synchronous and
that module marks its whole suite `asyncio`.
"""

import pytest

from sherlock_project.ai_engine import sanitize_pass_one_extraction


def _sanitized_name(value: str, site_name: str) -> list[str]:
    return sanitize_pass_one_extraction(
        {"full_name": [value]},
        searched_username="0day",
        site_name=site_name,
        site_content="",
    ).get("full_name", [])


@pytest.mark.parametrize(
    ("value", "site_name", "expected"),
    [
        # The live failure this rule exists for: a page whose only owner
        # evidence was its title, filed whole under `full_name`.
        ("Steam Community :: Ryan", "Steam", ["Ryan"]),
        # The case `pass_one.md` has taught since it was written, and which
        # the model still gets wrong. Repaired here rather than trusted there.
        ("Game Community :: Erik", "Game Community", ["Erik"]),
        ("Ryan - Pinbase", "Pinbase", ["Ryan"]),
        # Repair, never blanket rejection: real names survive untouched.
        ("Ryan", "Steam", ["Ryan"]),
        ("Ryan M. Montgomery", "Mastodon", ["Ryan M. Montgomery"]),
        # Hyphens inside a name are not title separators.
        ("Anne-Marie Okonkwo", "Steam", ["Anne-Marie Okonkwo"]),
        ("Mary Jane Watson-Parker", "Steam", ["Mary Jane Watson-Parker"]),
        # A site token under five characters is a substring of ordinary names
        # -- `me` is inside `James` -- so it is not treated as branding.
        ("James", "Me", ["James"]),
        # Nothing survives repair in these: they are not names at all.
        ("https://steamcommunity.com/id/x", "Steam", []),
        ("Steam Community", "Steam", []),
        (
            "A security researcher who builds defensive tooling for teams",
            "Steam",
            [],
        ),
    ],
)
def test_name_values_are_repaired_by_shape_not_matched_against_a_denylist(
    value: str,
    site_name: str,
    expected: list[str],
) -> None:
    assert _sanitized_name(value, site_name) == expected


def test_name_repair_leaves_other_keys_alone() -> None:
    """Only person-name keys are shape-checked; handles may look like anything."""

    sanitized = sanitize_pass_one_extraction(
        {"usernames": ["steam_community_ryan"], "bio": ["Ryan - Steam player"]},
        searched_username="0day",
        site_name="Steam",
        site_content="",
    )

    assert sanitized["usernames"] == ["steam_community_ryan"]
    assert sanitized["bio"] == ["Ryan - Steam player"]
