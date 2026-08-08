import json

import pytest

from sherlock_project.database import SherlockDB
from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    InvestigationContext,
    ProfileSynthesis,
    SourceDecision,
)
from sherlock_project.result import QueryStatus
from sherlock_project.synthesis_pipeline import synthesize_username_profile

pytestmark = pytest.mark.asyncio

CURRENT_PASS_ONE_CONTRACT_HASH = "pass-one-contract-v2"


class FakeSynthesisService:
    def __init__(self) -> None:
        self.model_key = "fake-model"
        self.pass_one_contract_hash = CURRENT_PASS_ONE_CONTRACT_HASH
        self.synthesis_prompt_fingerprints = {"identity": "identity-v1"}
        self.calls: list[dict] = []
        self.error: Exception | None = None
        self.failed_decisions = False

    async def synthesize(self, **kwargs) -> ProfileSynthesis:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        anchored = kwargs["context"].has_anchors
        has_evidence = any(item.extraction for item in kwargs["extractions"])
        return ProfileSynthesis(
            username=kwargs["username"],
            input_hash=kwargs["input_hash"],
            mode="anchored" if anchored else "aggregate",
            resolution_status=(
                "resolved"
                if anchored and has_evidence
                else "insufficient_evidence"
                if anchored
                else "aggregated"
                if has_evidence
                else "no_evidence"
            ),
            completeness=(
                "partial"
                if self.failed_decisions
                or kwargs["pending_site_ids"]
                or kwargs["invalid_extractions"]
                else "complete"
            ),
            anchors=kwargs["context"].anchors,
            source_decisions=(
                [
                    SourceDecision(
                        site_id=1,
                        site_name="Example",
                        disposition="failed",
                    )
                ]
                if self.failed_decisions
                else []
            ),
        )


class ContractFilteringService:
    def __init__(self) -> None:
        self.model_key = "fake-model"
        self.pass_one_contract_hash = CURRENT_PASS_ONE_CONTRACT_HASH
        self.synthesis_prompt_fingerprints: dict[str, str] = {}
        self.calls: list[dict] = []

    async def synthesize(self, **kwargs) -> ProfileSynthesis:
        self.calls.append(kwargs)
        has_evidence = any(item.extraction for item in kwargs["extractions"])
        return ProfileSynthesis(
            username=kwargs["username"],
            input_hash=kwargs["input_hash"],
            mode="aggregate",
            resolution_status="aggregated" if has_evidence else "no_evidence",
            completeness=(
                "partial"
                if kwargs["pending_site_ids"] or kwargs["invalid_extractions"]
                else "complete"
            ),
        )


async def _save_extraction(
    db: SherlockDB,
    *,
    username: str = "blue",
    site_name: str = "example",
    payload: str | None = '{"full_name": ["Alice"]}',
    contract_hash: str | None = CURRENT_PASS_ONE_CONTRACT_HASH,
) -> int:
    return await db.save_result(
        username=username,
        site_name=site_name,
        site_url=f"https://{site_name}.com/{username}",
        status=str(QueryStatus.CLAIMED),
        response_text="profile content",
        ai_extraction=payload,
        ai_extraction_contract_hash=contract_hash,
    )


async def test_pipeline_caches_unchanged_synthesis(db: SherlockDB):
    await _save_extraction(db)
    service = FakeSynthesisService()

    first = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    second = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(service.calls) == 1
    cache = await db.get_profile_summary_cache("blue")
    assert cache is not None
    assert cache.input_hash == first.profile.input_hash
    assert cache.updated_at is not None


async def test_pipeline_force_and_changed_context_bypass_cache(db: SherlockDB):
    await _save_extraction(db)
    service = FakeSynthesisService()

    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
        force=True,
    )
    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(
            anchors=[
                IdentityAnchor(
                    field="location",
                    value="London",
                    trust="context",
                )
            ]
        ),
    )

    assert len(service.calls) == 3


async def test_latest_profile_cache_tracks_run_only_context(db: SherlockDB):
    await _save_extraction(db)
    service = FakeSynthesisService()
    no_anchors = InvestigationContext()
    inline_context = InvestigationContext(
        anchors=[
            IdentityAnchor(
                field="full_name",
                value="Avery Stone",
                trust="verified",
                source="command_line",
            )
        ]
    )

    first = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=no_anchors,
    )
    anchored = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=inline_context,
    )
    latest = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=no_anchors,
    )

    assert first.profile.input_hash != anchored.profile.input_hash
    assert latest.cache_hit is False
    assert len(service.calls) == 3
    cache = await db.get_profile_summary_cache("blue")
    assert cache is not None
    assert cache.input_hash == latest.profile.input_hash


async def test_pipeline_reports_pending_and_invalid_extractions(db: SherlockDB):
    pending_id = await _save_extraction(db, site_name="pending", payload=None)
    invalid_id = await _save_extraction(db, site_name="invalid", payload="not-json")
    await _save_extraction(db, site_name="empty", payload="{}")
    service = FakeSynthesisService()

    result = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    assert result.profile.completeness == "partial"
    call = service.calls[0]
    assert call["pending_site_ids"] == [pending_id]
    assert [item.site_id for item in call["invalid_extractions"]] == [invalid_id]
    assert call["extractions"][0].extraction == {}


async def test_pipeline_excludes_missing_and_stale_contract_extractions(
    db: SherlockDB,
):
    current_id = await _save_extraction(
        db,
        site_name="current",
        payload='{"full_name": ["Alice"]}',
    )
    stale_id = await _save_extraction(
        db,
        site_name="stale",
        payload="not-json",
        contract_hash="older-pass-one-contract",
    )
    legacy_id = await _save_extraction(
        db,
        site_name="legacy",
        payload='{"location": ["London"]}',
        contract_hash=None,
    )
    service = ContractFilteringService()

    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    call = service.calls[0]
    assert [item.site_id for item in call["extractions"]] == [current_id]
    assert call["extractions"][0].extraction == {"full_name": ["Alice"]}
    assert call["pending_site_ids"] == [stale_id, legacy_id]
    assert call["invalid_extractions"] == []


async def test_pipeline_rejects_malformed_current_contract_extraction_shape(
    db: SherlockDB,
):
    malformed_id = await _save_extraction(
        db,
        site_name="malformed",
        payload='{"full_name": "Alice"}',
    )
    valid_id = await _save_extraction(
        db,
        site_name="valid",
        payload='{"conference_talks": ["Defensive Python"]}',
    )
    service = ContractFilteringService()

    result = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    call = service.calls[0]
    assert [item.site_id for item in call["extractions"]] == [valid_id]
    assert call["extractions"][0].extraction == {
        "conference_talks": ["Defensive Python"]
    }
    assert call["pending_site_ids"] == []
    assert [
        item.site_id for item in call["invalid_extractions"]
    ] == [malformed_id]
    assert result.profile.completeness == "partial"


async def test_failed_rebuild_preserves_previous_profile_and_remains_retryable(
    db: SherlockDB,
):
    site_id = await _save_extraction(db)
    service = FakeSynthesisService()
    original = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    await db.update_result_ai_extraction(
        site_id=site_id,
        ai_extraction='{"full_name": ["Bob"]}',
        contract_hash=service.pass_one_contract_hash,
    )
    service.error = RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        await synthesize_username_profile(
            db=db,
            ai_service=service,  # type: ignore[arg-type]
            username="blue",
            context=InvestigationContext(),
        )

    cache = await db.get_profile_summary_cache("blue")
    assert cache is not None
    assert cache.input_hash is None
    assert cache.profile_summary is not None
    stored = ProfileSynthesis.model_validate_json(cache.profile_summary)
    assert stored.input_hash == original.profile.input_hash

    service.error = None
    retried = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    assert retried.cache_hit is False


async def test_anchorless_cache_ignores_model_and_identity_prompt_changes(
    db: SherlockDB,
):
    await _save_extraction(db)
    service = FakeSynthesisService()

    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    service.synthesis_prompt_fingerprints["identity"] = "identity-v2"
    service.model_key = "different-model"
    second = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    assert second.cache_hit is True
    assert len(service.calls) == 1


async def test_anchored_cache_tracks_model_and_identity_prompt_changes(
    db: SherlockDB,
):
    await _save_extraction(db)
    service = FakeSynthesisService()
    context = InvestigationContext(
        anchors=[IdentityAnchor(field="email", value="blue@example.com")]
    )

    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=context,
    )
    service.synthesis_prompt_fingerprints["identity"] = "identity-v2"
    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=context,
    )
    service.model_key = "different-model"
    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=context,
    )

    assert len(service.calls) == 3


async def test_cached_profile_uses_current_schema(db: SherlockDB):
    await _save_extraction(db)
    service = FakeSynthesisService()

    await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    cache = await db.get_profile_summary_cache("blue")
    assert cache is not None and cache.profile_summary is not None
    parsed = json.loads(cache.profile_summary)
    assert parsed["schema_version"] == 8


async def test_cached_profile_from_previous_schema_is_rebuilt(db: SherlockDB):
    await _save_extraction(db)
    service = FakeSynthesisService()
    first = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )
    stale_profile = first.profile.model_dump(mode="json")
    stale_profile["schema_version"] = 7
    await db.update_username_profile_summary(
        username="blue",
        profile_summary=json.dumps(stale_profile),
        input_hash=first.profile.input_hash,
    )

    rebuilt = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(),
    )

    assert rebuilt.cache_hit is False
    assert rebuilt.profile.schema_version == 8
    assert len(service.calls) == 2


async def test_partial_profile_with_failed_decision_is_never_a_cache_hit(
    db: SherlockDB,
):
    await _save_extraction(db)
    service = FakeSynthesisService()
    service.failed_decisions = True

    first = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Alice")]
        ),
    )
    second = await synthesize_username_profile(
        db=db,
        ai_service=service,  # type: ignore[arg-type]
        username="blue",
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Alice")]
        ),
    )

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert len(service.calls) == 2
