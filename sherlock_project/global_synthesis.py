"""Anchored Pass 2 as a few global decisions instead of one per site.

The sequential Pass 2 (`AIService.synthesize`) asks a model about one site at a
time, because a small local model cannot hold many at once. That costs one
request per site plus one per unsure site, depends on site order, and never
revisits a site rejected before the evidence that would have matched it
arrived. On a metered hosted model the request count is the binding cost.

Here the same decision is made in rounds over the whole set:

  Round 0 (code, no model): links code can see for itself. An exact anchor
    match, a profile linked from an already-matched profile, or an email or
    phone shared with the matched profile.
  Round 1 (model): every undecided profile at once, in chunks, against the
    anchors plus whatever round 0 established.
  Round 2 (model, only if round 1 added evidence): every profile not yet a
    strong match -- rejects included -- against the grown profile.

Every model decision must cite the facts it rests on, and code checks those
citations exist. A decision whose citations do not check out is downgraded.
That turns the prompt's rule -- every match rests on a fact of this site -- from
a request into a guarantee.

Everything in this module is pure: no I/O, no model. `AIService` owns the
requests and calls these to decide what to send and what the answers mean.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sherlock_project.profile_synthesis import (
    IdentityStatus,
    InvestigationContext,
    SiteExtraction,
    canonical_field,
    is_username_variant,
    merge_extraction,
    normalize_value,
)

# A chunk is one request. Thirty compact extractions is roughly 6-10K tokens,
# well inside any hosted context and small enough that attention stays on each
# profile; more than that and the middle of the list is read less carefully.
GLOBAL_MAX_SITES_PER_REQUEST = 30
GLOBAL_MAX_PAYLOAD_BYTES = 48_000
# Model rounds, after round 0. Two mirrors the sequential path's two sweeps.
GLOBAL_MAX_MODEL_ROUNDS = 2
# Fields whose values identify one person outright when two profiles share
# them. Names and places do not qualify: thousands of people share both.
_UNIQUE_IDENTIFIER_FIELDS = frozenset({"email", "phone"})
_LINK_FIELDS = frozenset({"url", "domain", "links", "websites", "social_links"})

_DOWNGRADE: dict[IdentityStatus, IdentityStatus] = {
    "strong_match": "unsure",
    "unsure": "reject",
    "reject": "reject",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    site_value: str = Field(description="The value exactly as it appears in this profile's extraction.")
    reference: str = Field(description="The anchor or strong_profile value it matches, exactly as given.")


class GlobalDecision(_Strict):
    ref: str = Field(description="The profile's ref, exactly as given in sites.")
    identity_status: IdentityStatus
    matched: list[Citation] = Field(
        description="Facts this decision rests on; empty for reject."
    )


class GlobalDecisions(_Strict):
    decisions: list[GlobalDecision]


@dataclass(slots=True)
class Verdict:
    status: IdentityStatus
    # "deterministic" for round 0, otherwise the model round that decided it.
    origin: str
    citations: list[Citation] = field(default_factory=list)
    downgraded: bool = False


def loose(value: object) -> str:
    """Comparison form for citations: spacing, case and a leading @ ignored."""
    return " ".join(str(value).split()).casefold().removeprefix("@")


def _values(raw: object) -> list[str]:
    items = raw if isinstance(raw, list) else [raw]
    return [str(item) for item in items if isinstance(item, (str, int, float))]


def normalized_facts(extraction: Mapping[str, Any], username: str) -> set[tuple[str, str]]:
    """(canonical field, normalized value), with the searched username removed."""
    facts: set[tuple[str, str]] = set()
    for key, raw in extraction.items():
        for value in _values(raw):
            if is_username_variant(value, username):
                continue
            facts.add((canonical_field(key), normalize_value(key, value)))
    return facts


def site_refs(extractions: Sequence[SiteExtraction]) -> dict[str, SiteExtraction]:
    """Opaque refs in site-id order. Site names are left out on purpose.

    A model shown `LinkedIn` beside `RandomForum` weighs them by reputation,
    and reputation is not evidence about who owns either account.
    """
    ordered = sorted(extractions, key=lambda item: item.site_id)
    return {f"s{index}": extraction for index, extraction in enumerate(ordered, start=1)}


def _anchor_facts(context: InvestigationContext, username: str) -> set[tuple[str, str]]:
    return {
        (canonical_field(anchor.field), normalize_value(anchor.field, anchor.value))
        for anchor in context.anchors
        if not is_username_variant(anchor.value, username)
    }


def _link_targets(extraction: Mapping[str, Any]) -> set[str]:
    targets: set[str] = set()
    for key, raw in extraction.items():
        for value in _values(raw):
            text = value.strip()
            looks_like_url = "://" in text or (
                "." in text and "/" in text and " " not in text
            )
            if looks_like_url and (canonical_field(key) in _LINK_FIELDS or "://" in text):
                targets.add(normalize_value("url", text))
    return targets


def deterministic_round(
    *,
    username: str,
    context: InvestigationContext,
    sites: Mapping[str, SiteExtraction],
    verdicts: dict[str, Verdict],
) -> bool:
    """Mark what code can prove without a model. True if anything changed.

    Run to a fixpoint, because each rule can feed the next: an anchor match
    makes a profile strong, its links make the profiles it points at strong,
    and their emails can match a fourth.
    """
    anchor_facts = _anchor_facts(context, username)
    changed_any = False
    while True:
        strong = [ref for ref, verdict in verdicts.items() if verdict.status == "strong_match"]
        linked_from_strong: set[str] = set()
        identifiers: set[tuple[str, str]] = set()
        for ref in strong:
            linked_from_strong |= _link_targets(sites[ref].extraction)
            identifiers |= {
                fact
                for fact in normalized_facts(sites[ref].extraction, username)
                if fact[0] in _UNIQUE_IDENTIFIER_FIELDS
            }
        changed = False
        for ref, site in sites.items():
            current = verdicts.get(ref)
            if current is not None and current.status == "strong_match":
                continue
            facts = normalized_facts(site.extraction, username)
            own_url = normalize_value("url", site.site_url) if site.site_url else None
            if (
                facts & anchor_facts
                or (own_url is not None and own_url in linked_from_strong)
                or facts & identifiers
            ):
                verdicts[ref] = Verdict(status="strong_match", origin="deterministic")
                changed = True
        if not changed:
            return changed_any
        changed_any = True


def strong_profile_of(
    *,
    username: str,
    sites: Mapping[str, SiteExtraction],
    verdicts: Mapping[str, Verdict],
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for ref in sorted(sites, key=lambda ref: sites[ref].site_id):
        verdict = verdicts.get(ref)
        if verdict is not None and verdict.status == "strong_match":
            merge_extraction(profile, sites[ref].extraction, username=username)
    return profile


def chunk_refs(
    refs: Sequence[str],
    sites: Mapping[str, SiteExtraction],
    *,
    base_payload_bytes: int,
    max_sites: int = GLOBAL_MAX_SITES_PER_REQUEST,
    max_bytes: int = GLOBAL_MAX_PAYLOAD_BYTES,
) -> list[list[str]]:
    """Split refs into requests, by count and by serialized size.

    A single profile over the byte budget still gets a request of its own:
    refusing it would silently drop a site, and an over-large request is the
    provider's to reject, visibly.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    size = base_payload_bytes
    for ref in refs:
        entry = len(
            json.dumps({"ref": ref, "extraction": sites[ref].extraction}, ensure_ascii=False).encode("utf-8")
        )
        if current and (len(current) >= max_sites or size + entry > max_bytes):
            chunks.append(current)
            current, size = [], base_payload_bytes
        current.append(ref)
        size += entry
    if current:
        chunks.append(current)
    return chunks


def reference_values(
    anchors: Mapping[str, Sequence[str]],
    strong_profile: Mapping[str, Any],
) -> set[str]:
    values = {loose(value) for items in anchors.values() for value in items}
    for raw in strong_profile.values():
        values |= {loose(value) for value in _values(raw)}
    return values


def ground(
    decision: GlobalDecision,
    *,
    site: SiteExtraction,
    references: set[str],
) -> tuple[IdentityStatus, list[Citation], bool]:
    """Keep only citations that exist; downgrade a decision left with none."""
    if decision.identity_status == "reject":
        return "reject", [], False
    own = {loose(value) for raw in site.extraction.values() for value in _values(raw)}
    valid = [
        citation
        for citation in decision.matched
        if loose(citation.site_value) in own and loose(citation.reference) in references
    ]
    if valid:
        return decision.identity_status, valid, False
    return _DOWNGRADE[decision.identity_status], [], True


def apply_decisions(
    response: GlobalDecisions,
    *,
    chunk: Iterable[str],
    sites: Mapping[str, SiteExtraction],
    references: set[str],
    verdicts: dict[str, Verdict],
    origin: str,
) -> list[str]:
    """Record one chunk's answers. Returns the refs the model left out.

    A ref the model invented, or answered twice, is ignored rather than
    trusted: only the first answer for a ref that was actually asked counts.
    """
    asked = set(chunk)
    answered: set[str] = set()
    for decision in response.decisions:
        if decision.ref not in asked or decision.ref in answered:
            continue
        answered.add(decision.ref)
        status, citations, downgraded = ground(
            decision, site=sites[decision.ref], references=references
        )
        verdicts[decision.ref] = Verdict(
            status=status, origin=origin, citations=citations, downgraded=downgraded
        )
    return [ref for ref in chunk if ref not in answered]
