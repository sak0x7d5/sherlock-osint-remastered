from argparse import ArgumentTypeError

import pytest

from sherlock_project.investigation_context import (
    build_investigation_context,
    parse_inline_anchor,
)
from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    InvestigationContext,
    SynthesisEvidence,
    compute_synthesis_input_hash,
)


@pytest.mark.parametrize(
    ("argument", "field", "value", "trust"),
    [
        (
            "full_name=Avery Stone",
            "full_name",
            "Avery Stone",
            "context",
        ),
        ("strong:domain=example.com", "domain", "example.com", "strong"),
        ("context:location=Lisbon", "location", "Lisbon", "context"),
        (
            "VERIFIED:email=Avery@Example.com",
            "email",
            "Avery@Example.com",
            "verified",
        ),
        ("roles=Éthical Hacker", "roles", "Éthical Hacker", "context"),
        (
            "description=Founder, researcher = builder",
            "description",
            "Founder, researcher = builder",
            "context",
        ),
    ],
)
def test_parse_inline_anchor(argument: str, field: str, value: str, trust: str):
    anchor = parse_inline_anchor(argument)

    assert anchor.field == field
    assert anchor.value == value
    assert anchor.trust == trust
    assert anchor.source == "command_line"


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ("full_name", "FIELD=VALUE"),
        ("=Avery", "field must not be empty"),
        ("roles=", "value must not be empty"),
        ("trusted:email=avery@example.com", "unsupported anchor trust"),
        ("roles=[Ethical Hacker, Pentester]", "repeat --anchor"),
    ],
)
def test_parse_inline_anchor_rejects_malformed_values(argument: str, message: str):
    with pytest.raises(ArgumentTypeError, match=message):
        parse_inline_anchor(argument)


def test_build_investigation_context_deduplicates_promotes_and_is_deterministic():
    anchors = [
        IdentityAnchor(
            field="email",
            value="Avery@Example.com",
            trust="verified",
            source="command_line",
        ),
        IdentityAnchor(
            field="role",
            value="Pentester",
            trust="context",
            source="command_line",
        ),
        parse_inline_anchor("email=avery@example.COM"),
        parse_inline_anchor("strong:roles=Pentester"),
        parse_inline_anchor("roles=Wildlife Rescue Volunteer"),
    ]

    first = build_investigation_context(anchors)
    second = build_investigation_context(list(reversed(anchors)))

    assert first == second
    anchors = {
        (anchor.field, anchor.value): anchor
        for anchor in first.anchors
    }
    assert len(anchors) == 3
    assert anchors[("email", "Avery@Example.com")].trust == "verified"
    assert anchors[("email", "Avery@Example.com")].source == "command_line"
    assert anchors[("roles", "Pentester")].trust == "strong"
    assert anchors[("roles", "Pentester")].source == "command_line"

    fingerprints = {
        compute_synthesis_input_hash(
            username="fixture_handle",
            model_key="model",
            context=merged,
            evidence=SynthesisEvidence(),
            prompt_fingerprints={"identity": "one"},
        )
        for merged in (first, second)
    }
    assert len(fingerprints) == 1


def test_unprefixed_context_value_still_counts_as_an_anchor():
    context = InvestigationContext(
        anchors=[parse_inline_anchor("roles=Pentester")]
    )

    assert context.has_trusted_anchor is False
    assert context.has_anchors is True


@pytest.mark.parametrize("trust", ["strong", "verified"])
def test_trusted_inline_anchor_can_create_target(trust: str):
    context = InvestigationContext(
        anchors=[parse_inline_anchor(f"{trust}:email=avery@example.com")]
    )

    assert context.has_trusted_anchor is True
    assert context.has_anchors is True
