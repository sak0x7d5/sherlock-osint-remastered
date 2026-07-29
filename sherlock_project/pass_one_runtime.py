from __future__ import annotations

from collections.abc import Collection
from json import loads as json_loads

from sherlock_project.ai_engine import (
    PassOneKeyRegistry,
    validate_pass_one_extraction_payload,
)
from sherlock_project.database import SherlockDB


async def hydrate_pass_one_key_registry(
    *,
    sherlock_db: SherlockDB,
    registry: PassOneKeyRegistry,
    username: str,
    contract_hash: str,
    exclude_site_ids: Collection[int] = (),
) -> None:
    """Seed current-contract key names in deterministic result-id order."""

    excluded = set(exclude_site_ids)
    extractions: list[dict[str, list[str]]] = []
    for record in await sherlock_db.get_ai_profile_evidence(username):
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
