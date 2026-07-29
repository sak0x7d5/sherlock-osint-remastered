from __future__ import annotations

from argparse import ArgumentTypeError
from typing import Sequence, cast

from sherlock_project.profile_synthesis import (
    AnchorTrust,
    IdentityAnchor,
    InvestigationContext,
    canonical_field,
    normalize_value,
)


_TRUST_ORDER: dict[AnchorTrust, int] = {
    "context": 0,
    "strong": 1,
    "verified": 2,
}
_TRUST_LEVELS = frozenset(_TRUST_ORDER)


def parse_inline_anchor(argument: str) -> IdentityAnchor:
    """Parse one repeatable CLI anchor in [TRUST:]FIELD=VALUE form."""
    if "=" not in argument:
        raise ArgumentTypeError(
            "anchor must use [TRUST:]FIELD=VALUE syntax; "
            "for multiple values, repeat --anchor"
        )

    key, value = argument.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ArgumentTypeError("anchor field must not be empty")
    if not value:
        raise ArgumentTypeError("anchor value must not be empty")
    if value.startswith("[") and value.endswith("]"):
        raise ArgumentTypeError(
            "anchor lists are not supported; repeat --anchor for each value"
        )

    trust: AnchorTrust = "context"
    field = key
    if ":" in key:
        trust_text, field = key.split(":", 1)
        trust_text = trust_text.strip().casefold()
        field = field.strip()
        if trust_text not in _TRUST_LEVELS:
            allowed = ", ".join(sorted(_TRUST_LEVELS))
            raise ArgumentTypeError(
                f"unsupported anchor trust {trust_text!r}; choose one of: {allowed}"
            )
        trust = cast(AnchorTrust, trust_text)
    if not field:
        raise ArgumentTypeError("anchor field must not be empty")

    return IdentityAnchor(
        field=field,
        value=value,
        trust=trust,
        source="command_line",
    )


def build_investigation_context(
    inline_anchors: Sequence[IdentityAnchor],
) -> InvestigationContext:
    """Return a deterministic context built from run-only CLI anchors."""
    grouped: dict[tuple[str, str], list[IdentityAnchor]] = {}
    for anchor in inline_anchors:
        key = (
            canonical_field(anchor.field),
            normalize_value(anchor.field, anchor.value),
        )
        grouped.setdefault(key, []).append(anchor)

    selected = {
        key: max(candidates, key=_anchor_preference)
        for key, candidates in grouped.items()
    }
    anchors = [selected[key] for key in sorted(selected)]
    return InvestigationContext(anchors=anchors)


def _anchor_preference(
    anchor: IdentityAnchor,
) -> tuple[int, int, str, str, str, str, str]:
    return (
        _TRUST_ORDER[anchor.trust],
        int(anchor.source == "command_line"),
        anchor.field.casefold(),
        anchor.value.casefold(),
        anchor.source or "",
        anchor.field,
        anchor.value,
    )
