from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from pydantic import ValidationError

from sherlock_project.ai_engine import (
    AIService,
    validate_pass_one_extraction_payload,
)
from sherlock_project.database import SherlockDB
from sherlock_project.profile_synthesis import (
    InvalidExtraction,
    InvestigationContext,
    ProfileSynthesis,
    SiteExtraction,
    SynthesisEvidence,
    compute_synthesis_input_hash,
)


@dataclass(frozen=True, slots=True)
class SynthesisRunResult:
    profile: ProfileSynthesis
    cache_hit: bool


async def synthesize_username_profile(
    *,
    db: SherlockDB,
    ai_service: AIService,
    username: str,
    context: InvestigationContext,
    force: bool = False,
) -> SynthesisRunResult:
    records = await db.get_ai_profile_evidence(username)
    current_contract_hash = ai_service.pass_one_contract_hash
    extractions: list[SiteExtraction] = []
    pending_site_ids: list[int] = []
    invalid_extractions: list[InvalidExtraction] = []

    for record in records:
        if (
            record.ai_extraction is None
            or record.ai_extraction_contract_hash != current_contract_hash
        ):
            pending_site_ids.append(record.site_id)
            continue
        try:
            parsed = json.loads(record.ai_extraction)
            validated = validate_pass_one_extraction_payload(parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            invalid_extractions.append(
                InvalidExtraction(
                    site_id=record.site_id,
                    site_name=record.site_name,
                    payload_hash=sha256(
                        record.ai_extraction.encode("utf-8")
                    ).hexdigest(),
                    error=str(error),
                )
            )
            continue

        extractions.append(
            SiteExtraction(
                site_id=record.site_id,
                site_name=record.site_name,
                site_url=record.site_url,
                scanned_at=record.scanned_at,
                extraction=validated,
            )
        )

    evidence = SynthesisEvidence(
        extractions=extractions,
        pending_site_ids=sorted(pending_site_ids),
        invalid_extractions=invalid_extractions,
    )
    input_hash = compute_synthesis_input_hash(
        username=username,
        model_key=ai_service.model_key,
        context=context,
        evidence=evidence,
        prompt_fingerprints=ai_service.synthesis_prompt_fingerprints,
    )

    if not force:
        cached = await db.get_profile_summary_cache(username)
        if (
            cached is not None
            and cached.input_hash == input_hash
            and cached.profile_summary is not None
        ):
            try:
                profile = ProfileSynthesis.model_validate_json(cached.profile_summary)
            except ValidationError:
                pass
            else:
                has_failed_decisions = any(
                    decision.disposition == "failed"
                    for decision in profile.source_decisions
                )
                if profile.input_hash == input_hash and not has_failed_decisions:
                    return SynthesisRunResult(profile=profile, cache_hit=True)

    if force:
        await db.invalidate_username_profile_summary(username)

    profile = await ai_service.synthesize(
        username=username,
        extractions=extractions,
        context=context,
        input_hash=input_hash,
        pending_site_ids=pending_site_ids,
        invalid_extractions=invalid_extractions,
    )
    serialized = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
    )
    await db.update_username_profile_summary(
        username=username,
        profile_summary=serialized,
        input_hash=input_hash,
    )
    return SynthesisRunResult(profile=profile, cache_hit=False)
