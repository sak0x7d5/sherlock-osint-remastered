import pytest
from pydantic import ValidationError

from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    InvalidExtraction,
    InvestigationContext,
    SiteExtraction,
    SynthesisEvidence,
    aggregate_synthesis,
    compute_synthesis_input_hash,
    flatten_facts,
    merge_extraction,
    synthesis_warnings,
)


def _extraction(
    site_id: int,
    extraction: dict[str, list[str]],
    *,
    site_name: str | None = None,
    site_url: str | None = None,
    scanned_at: str | None = None,
) -> SiteExtraction:
    return SiteExtraction(
        site_id=site_id,
        site_name=site_name or f"site-{site_id}",
        site_url=site_url,
        scanned_at=scanned_at,
        extraction=extraction,
    )


def test_investigation_context_rejects_removed_known_facts():
    with pytest.raises(ValidationError, match="known_facts"):
        InvestigationContext.model_validate(
            {"known_facts": {"location": ["Lisbon"]}}
        )


def test_flatten_facts_canonicalizes_nested_values_without_guessing():
    facts = flatten_facts(
        {
            "name": "Avery Stone",
            "organizations": [{"name": "Northstar Collective"}],
            "roles": ["Pentester", "Wildlife Rescue Volunteer"],
        }
    )

    assert {(fact.field, fact.value) for fact in facts} == {
        ("full_name", "Avery Stone"),
        ("organizations", "Northstar Collective"),
        ("roles", "Pentester"),
        ("roles", "Wildlife Rescue Volunteer"),
    }


def test_merge_extraction_preserves_keys_and_deduplicates_only_exact_values():
    profile: dict[str, list[str]] = {}

    merge_extraction(
        profile,
        {
            "employer": ["Northstar Collective"],
            "roles": ["Pentester"],
        },
    )
    merge_extraction(
        profile,
        {
            "employer": ["Northstar Collective", "northstar collective"],
            "organizations": ["Northstar Collective"],
            "roles": ["Pentester", "Wildlife Rescue Volunteer"],
        },
    )

    assert profile == {
        "employer": ["Northstar Collective", "northstar collective"],
        "roles": ["Pentester", "Wildlife Rescue Volunteer"],
        "organizations": ["Northstar Collective"],
    }


def test_anchorless_synthesis_uses_v8_profiles_and_preserves_exact_keys():
    evidence = SynthesisEvidence(
        extractions=[
            _extraction(
                1,
                {
                    "full_name": ["Avery Stone"],
                    "roles": ["Pentester"],
                },
                site_name="Instagram",
            ),
            _extraction(
                2,
                {
                    "name": ["avery stone"],
                    "roles": ["Pentester", "Wildlife Rescue Volunteer"],
                },
                site_name="Threads",
            ),
            _extraction(3, {"full_name": ["Avery"]}, site_name="Forum"),
        ]
    )

    result = aggregate_synthesis(
        username="fixture_handle",
        input_hash="hash",
        context=InvestigationContext(),
        evidence=evidence,
    )

    assert result.mode == "aggregate"
    assert result.resolution_status == "aggregated"
    assert result.strong_profile == {
        "full_name": ["Avery Stone", "Avery"],
        "roles": ["Pentester", "Wildlife Rescue Volunteer"],
        "name": ["avery stone"],
    }
    assert result.unsure_profile == {}
    provenance = {
        (item.field, item.value): item.source_site_ids
        for item in result.provenance
    }
    assert provenance == {
        ("full_name", "Avery Stone"): [1],
        ("full_name", "Avery"): [3],
        ("name", "avery stone"): [2],
        ("roles", "Pentester"): [1, 2],
        ("roles", "Wildlife Rescue Volunteer"): [2],
    }
    assert all(item.origins == ["extraction"] for item in result.provenance)
    assert all(
        decision.disposition == "aggregated"
        for decision in result.source_decisions
    )
    assert all(
        "reason" not in decision.model_dump()
        for decision in result.source_decisions
    )
    assert "different people" in result.warnings[0]
    assert set(result.model_dump()) == {
        "schema_version",
        "username",
        "input_hash",
        "mode",
        "resolution_status",
        "completeness",
        "strong_profile",
        "unsure_profile",
        "provenance",
        "source_decisions",
        "anchors",
        "warnings",
    }


def test_anchorless_synthesis_only_emits_extracted_values():
    context = InvestigationContext()
    result = aggregate_synthesis(
        username="blue",
        input_hash="hash",
        context=context,
        evidence=SynthesisEvidence(
            extractions=[_extraction(1, {"roles": ["Researcher"]})]
        ),
    )

    assert result.strong_profile == {"roles": ["Researcher"]}
    assert result.unsure_profile == {}
    assert [
        item.model_dump()
        for item in result.provenance
    ] == [
        {
            "field": "roles",
            "value": "Researcher",
            "source_site_ids": [1],
            "origins": ["extraction"],
        }
    ]


def test_empty_anchorless_evidence_returns_no_profile():
    result = aggregate_synthesis(
        username="blue",
        input_hash="hash",
        context=InvestigationContext(),
        evidence=SynthesisEvidence(extractions=[_extraction(1, {})]),
    )

    assert result.resolution_status == "no_evidence"
    assert result.strong_profile == {}
    assert result.unsure_profile == {}
    assert result.source_decisions[0].disposition == "ignored"


def test_anchorless_hash_tracks_source_url_but_ignores_model_prompt_and_scan_time():
    first = compute_synthesis_input_hash(
        username="blue",
        model_key="model-a",
        context=InvestigationContext(),
        evidence=SynthesisEvidence(
            extractions=[
                _extraction(
                    1,
                    {"name": ["Alice"]},
                    site_url="https://one.example/blue",
                    scanned_at="2026-01-01",
                )
            ]
        ),
        prompt_fingerprints={"identity": "prompt-a"},
    )
    same_source = compute_synthesis_input_hash(
        username="blue",
        model_key="model-b",
        context=InvestigationContext(),
        evidence=SynthesisEvidence(
            extractions=[
                _extraction(
                    1,
                    {"name": ["Alice"]},
                    site_url="https://one.example/blue",
                    scanned_at="2026-02-02",
                )
            ]
        ),
        prompt_fingerprints={"identity": "prompt-b"},
    )
    changed_source = compute_synthesis_input_hash(
        username="blue",
        model_key="model-b",
        context=InvestigationContext(),
        evidence=SynthesisEvidence(
            extractions=[
                _extraction(
                    1,
                    {"name": ["Alice"]},
                    site_url="https://two.example/blue",
                    scanned_at="2026-02-02",
                )
            ]
        ),
        prompt_fingerprints={"identity": "prompt-b"},
    )

    assert first == same_source
    assert first != changed_source


def test_anchored_hash_tracks_model_and_identity_prompt():
    context = InvestigationContext(
        anchors=[IdentityAnchor(field="email", value="blue@example.com")]
    )
    evidence = SynthesisEvidence(
        extractions=[_extraction(1, {"name": ["Alice"]})]
    )
    first = compute_synthesis_input_hash(
        username="blue",
        model_key="model-a",
        context=context,
        evidence=evidence,
        prompt_fingerprints={"identity": "prompt-a"},
    )
    second = compute_synthesis_input_hash(
        username="blue",
        model_key="model-b",
        context=context,
        evidence=evidence,
        prompt_fingerprints={"identity": "prompt-b"},
    )

    assert first != second


def test_pending_and_invalid_sources_become_completeness_warnings():
    evidence = SynthesisEvidence(
        pending_site_ids=[2],
        invalid_extractions=[
            InvalidExtraction(
                site_id=3,
                site_name="broken",
                payload_hash="abc",
                error="not JSON",
            )
        ],
    )

    assert evidence.completeness == "partial"
    warnings = synthesis_warnings(evidence)
    assert "site ids: 2" in warnings[0]
    assert "site id 3" in warnings[1]
