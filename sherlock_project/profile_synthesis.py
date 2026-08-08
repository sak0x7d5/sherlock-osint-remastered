from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

AnchorTrust = Literal["verified", "strong", "context"]
ProfileMode = Literal["aggregate", "anchored"]
ProfileOrigin = Literal["anchor", "extraction"]
IdentityStatus = Literal["strong_match", "unsure", "reject"]
SourceDisposition = Literal[
    "aggregated",
    "included",
    "excluded",
    "ignored",
    "failed",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdentityAnchor(StrictModel):
    field: str
    value: str
    trust: AnchorTrust = "strong"
    source: str | None = None

    @field_validator("field", "value")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class InvestigationContext(StrictModel):
    anchors: list[IdentityAnchor] = Field(default_factory=list)

    @property
    def has_trusted_anchor(self) -> bool:
        return any(anchor.trust in {"verified", "strong"} for anchor in self.anchors)

    @property
    def has_anchors(self) -> bool:
        return bool(self.anchors)


class SiteExtraction(StrictModel):
    site_id: int
    site_name: str
    site_url: str | None = None
    scanned_at: str | None = None
    extraction: dict[str, Any]


class InvalidExtraction(StrictModel):
    site_id: int
    site_name: str
    payload_hash: str
    error: str


class SynthesisEvidence(StrictModel):
    extractions: list[SiteExtraction] = Field(default_factory=list)
    pending_site_ids: list[int] = Field(default_factory=list)
    invalid_extractions: list[InvalidExtraction] = Field(default_factory=list)

    @property
    def completeness(self) -> Literal["complete", "partial"]:
        if self.pending_site_ids or self.invalid_extractions:
            return "partial"
        return "complete"


class NormalizedFact(StrictModel):
    field: str
    value: str

    @field_validator("field", "value")
    @classmethod
    def validate_fact_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be empty")
        return value


class ProfileValue(StrictModel):
    field: str
    value: str
    source_site_ids: list[int] = Field(default_factory=list)
    origins: list[ProfileOrigin] = Field(default_factory=list)


class SourceDecision(StrictModel):
    site_id: int
    site_name: str
    site_url: str | None = None
    disposition: SourceDisposition
    identity_status: IdentityStatus | None = None


class ProfileSynthesis(StrictModel):
    schema_version: Literal[8] = 8
    username: str
    input_hash: str
    mode: ProfileMode
    resolution_status: Literal[
        "aggregated",
        "resolved",
        "insufficient_evidence",
        "no_evidence",
    ]
    completeness: Literal["complete", "partial"]
    strong_profile: dict[str, Any] = Field(default_factory=dict)
    unsure_profile: dict[str, Any] = Field(default_factory=dict)
    provenance: list[ProfileValue] = Field(default_factory=list)
    source_decisions: list[SourceDecision] = Field(default_factory=list)
    anchors: list[IdentityAnchor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class _ValueState:
    field: str
    value: str
    source_site_ids: set[int] = field(default_factory=set)
    origins: set[ProfileOrigin] = field(default_factory=set)


_FIELD_ALIASES = {
    "name": "full_name",
    "real_name": "full_name",
    "display_name": "full_name",
    "fullname": "full_name",
    "alias": "aliases",
    "handle": "usernames",
    "username": "usernames",
    "other_username": "usernames",
    "other_usernames": "usernames",
    "mail": "email",
    "email_address": "email",
    "telephone": "phone",
    "phone_number": "phone",
    "website": "url",
    "personal_website": "url",
    "link": "url",
    "company": "organizations",
    "organization": "organizations",
    "organisation": "organizations",
    "employer": "organizations",
    "job": "roles",
    "occupation": "roles",
    "role": "roles",
    "city": "location",
    "country": "location",
}

_ORIGIN_ORDER: dict[ProfileOrigin, int] = {
    "anchor": 0,
    "extraction": 1,
}


def flatten_facts(value: Any, prefix: str = "") -> list[NormalizedFact]:
    facts: list[NormalizedFact] = []
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            facts.extend(flatten_facts(value[key], path))
        return facts

    if isinstance(value, list):
        for item in value:
            facts.extend(flatten_facts(item, prefix))
        return facts

    if value is None or not prefix:
        return facts
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (str, int, float)):
        text = str(value).strip()
    else:
        return facts
    if not text:
        return facts

    facts.append(NormalizedFact(field=canonical_field(prefix), value=text))
    return facts


def canonical_field(field_name: str) -> str:
    parts = [
        re.sub(r"[^a-z0-9]+", "_", part.casefold()).strip("_")
        for part in field_name.split(".")
    ]
    parts = [part for part in parts if part]
    if not parts:
        return "fact"

    field_name = _FIELD_ALIASES.get(parts[-1], parts[-1])
    if field_name in {"full_name", "title", "label", "value"}:
        contextual_fields = {
            "organizations",
            "roles",
            "skills",
            "interests",
            "communities",
            "projects",
        }
        for ancestor in reversed(parts[:-1]):
            contextual_field = _FIELD_ALIASES.get(ancestor, ancestor)
            if contextual_field in contextual_fields:
                return contextual_field
    return field_name


def normalize_value(field_name: str, value: str) -> str:
    field_name = canonical_field(field_name)
    value = " ".join(value.split()).strip()

    if field_name in {"email", "usernames"}:
        return value.removeprefix("@").casefold()
    if field_name == "phone":
        return "".join(character for character in value if character.isdigit())
    if field_name in {"url", "domain"} or value.casefold().startswith(
        ("http://", "https://")
    ):
        candidate = value if "://" in value else f"https://{value}"
        parsed = urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold().removeprefix("www.")
        if field_name == "domain":
            return hostname
        return urlunsplit(("", hostname, parsed.path.rstrip("/"), "", ""))
    return value.casefold()


def is_username_variant(value: Any, username: str) -> bool:
    if not isinstance(value, str):
        return False
    norm_val = value.strip().lstrip("@").replace(" ", "").casefold()
    norm_user = username.strip().lstrip("@").replace(" ", "").casefold()
    return norm_val == norm_user


def merge_extraction(
    profile: dict[str, Any],
    extraction: dict[str, Any],
    username: str | None = None,
) -> None:
    for key, new_value in extraction.items():
        if username is not None and key in {"full_name", "aliases", "usernames", "name"}:
            if isinstance(new_value, list):
                new_value = [
                    item for item in new_value
                    if not is_username_variant(item, username)
                ]
                if not new_value:
                    continue
            elif is_username_variant(new_value, username):
                continue

        if key not in profile:
            profile[key] = new_value
            continue

        existing_value = profile[key]

        if existing_value == new_value:
            continue

        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            merge_extraction(existing_value, new_value, username=username)
            continue

        existing_list = (
            existing_value if isinstance(existing_value, list) else [existing_value]
        )
        new_list = new_value if isinstance(new_value, list) else [new_value]

        merged_list = list(existing_list)
        for item in new_list:
            if item not in merged_list:
                merged_list.append(item)

        profile[key] = merged_list


def build_profile_provenance(
    profile: dict[str, Any],
    extractions: list[SiteExtraction],
) -> list[ProfileValue]:
    """Attach accepted source ids to exact dynamic-key profile values."""

    states: dict[tuple[str, str], _ValueState] = {}
    for field_name in sorted(profile):
        raw_values = profile[field_name]
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            states.setdefault(
                (field_name, value),
                _ValueState(field=field_name, value=value),
            )

    for extraction in sorted(extractions, key=lambda item: item.site_id):
        for field_name, raw_values in extraction.extraction.items():
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            for value in values:
                if not isinstance(value, str):
                    continue
                state = states.get((field_name, value))
                if state is None:
                    continue
                state.source_site_ids.add(extraction.site_id)
                state.origins.add("extraction")

    return [
        ProfileValue(
            field=state.field,
            value=state.value,
            source_site_ids=sorted(state.source_site_ids),
            origins=sorted(state.origins, key=_ORIGIN_ORDER.__getitem__),
        )
        for state in states.values()
        if state.source_site_ids
    ]


def synthesis_warnings(evidence: SynthesisEvidence) -> list[str]:
    warnings: list[str] = []
    if evidence.pending_site_ids:
        warnings.append(
            "Pass-one extraction is still pending for site ids: "
            + ", ".join(str(site_id) for site_id in evidence.pending_site_ids)
        )
    for invalid in evidence.invalid_extractions:
        warnings.append(
            f"Ignored invalid pass-one JSON for site id {invalid.site_id}: "
            f"{invalid.error}"
        )
    return warnings


def aggregate_synthesis(
    *,
    username: str,
    input_hash: str,
    context: InvestigationContext,
    evidence: SynthesisEvidence,
) -> ProfileSynthesis:
    strong_profile: dict[str, Any] = {}
    decisions: list[SourceDecision] = []
    for extraction in sorted(evidence.extractions, key=lambda item: item.site_id):
        if not extraction.extraction:
            decisions.append(
                SourceDecision(
                    site_id=extraction.site_id,
                    site_name=extraction.site_name,
                    site_url=extraction.site_url,
                    disposition="ignored",
                )
            )
            continue
        merge_extraction(strong_profile, extraction.extraction, username=username)
        decisions.append(
            SourceDecision(
                site_id=extraction.site_id,
                site_name=extraction.site_name,
                site_url=extraction.site_url,
                disposition="aggregated",
            )
        )

    warnings = synthesis_warnings(evidence)
    if strong_profile:
        warnings.insert(
            0,
            "No anchor was supplied. Values may describe different people "
            "who use the same username.",
        )
    return ProfileSynthesis(
        username=username,
        input_hash=input_hash,
        mode="aggregate",
        resolution_status="aggregated" if strong_profile else "no_evidence",
        completeness=evidence.completeness,
        strong_profile=strong_profile,
        unsure_profile={},
        provenance=build_profile_provenance(
            strong_profile,
            [
                extraction
                for extraction in evidence.extractions
                if extraction.extraction
            ],
        ),
        source_decisions=decisions,
        anchors=context.anchors,
        warnings=warnings,
    )


def compute_synthesis_input_hash(
    *,
    username: str,
    model_key: str,
    context: InvestigationContext,
    evidence: SynthesisEvidence,
    prompt_fingerprints: dict[str, str],
) -> str:
    anchored = context.has_anchors
    payload = {
        "schema_version": 8,
        "username": username,
        "mode": "anchored" if anchored else "aggregate",
        "model_key": model_key if anchored else None,
        "context": context.model_dump(mode="json"),
        "evidence": {
            "extractions": [
                {
                    "site_id": extraction.site_id,
                    "site_name": extraction.site_name,
                    "site_url": extraction.site_url,
                    "extraction": extraction.extraction,
                }
                for extraction in sorted(
                    evidence.extractions,
                    key=lambda item: item.site_id,
                )
            ],
            "pending_site_ids": sorted(evidence.pending_site_ids),
            "invalid_extractions": [
                {
                    "site_id": invalid.site_id,
                    "site_name": invalid.site_name,
                    "payload_hash": invalid.payload_hash,
                }
                for invalid in sorted(
                    evidence.invalid_extractions,
                    key=lambda item: item.site_id,
                )
            ],
        },
        "prompts": prompt_fingerprints if anchored else {},
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()
