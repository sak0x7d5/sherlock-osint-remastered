from __future__ import annotations

from collections.abc import Collection
from json import loads as json_loads

from sift_osint.ai_engine import (
    PassOneKeyRegistry,
    validate_pass_one_extraction_payload,
)
from sift_osint.database import SiftDB


async def hydrate_pass_one_key_registry(
    *,
    sift_db: SiftDB,
    registry: PassOneKeyRegistry,
    username: str,
    contract_hash: str,
    exclude_site_ids: Collection[int] = (),
) -> None:
    """Seed current-contract key names in deterministic result-id order."""

    excluded = set(exclude_site_ids)
    extractions: list[dict[str, list[str]]] = []
    for record in await sift_db.get_ai_profile_evidence(username):
        if record.site_id in excluded:
            continue
        if (
            record.ai_extraction is None
            or record.ai_extraction_contract_hash != contract_hash
        ):
            continue
        try:
            parsed = json_loads(record.ai_extraction)
            validated = validate_pass_one_extraction_payload(parsed)
        except (TypeError, ValueError):
            continue
        if validated:
            extractions.append(validated)

    registry.seed(username, extractions)
