"""Two-Sided Detection

Decides whether a username exists on a site from a two-sided rule: what a hit
looks like (``exists``) and what a miss looks like (``missing``), each a status
code plus an optional response marker.

The legacy manifest could only describe absence, so CLAIMED was the default and
any 2xx counted as a hit. That makes a soft-404 -- a "no such user" page served
with 200 -- indistinguishable from a real profile, and makes a stale rule
indistinguishable from a found account. Both sides being recorded is what lets
this module refuse to guess: a response matching neither side is reported as
UNKNOWN, not quietly upgraded to a hit.

Two ranking decisions are load-bearing:

* **A marker outranks a status code.** Codes are noisy in ways unrelated to the
  account -- CDNs, rate limiters, and WAFs all rewrite them -- while a marker is
  chosen to be specific to the site's profile page.
* **Contradiction is not evidence.** When the two sides disagree, or when
  nothing matches at all, the answer is UNKNOWN. Signals that cancel out must
  not average into a verdict.

One asymmetry follows from those and is worth stating outright, because the two
cases look alike and resolve differently. A missing hit-marker means *absence*
when the rule's two codes are identical -- the author recorded that the code
carries no information, so the marker is the only discriminator. The same
missing marker means *undecided* when the code matched ``exists``, because there
the code is a real second signal and it disagrees. That second case is the
soft-404 the legacy model reported as a hit.

This module is pure: no I/O, no network, no engine imports. Everything it
decides is reproducible from a status code and a response body.
"""

from enum import Enum


class QueryConfidence(Enum):
    """How much of the rule actually matched.

    Orthogonal to QueryStatus: the status says what was decided, the confidence
    says how much agreed. Downstream consumers use it to weight evidence rather
    than treating every CLAIMED as equally true.
    """

    CONFIRMED = "Confirmed"  # code and marker agree
    PROBABLE = "Probable"    # one signal matched, nothing contradicted it
    AMBIGUOUS = "Ambiguous"  # signals disagree, or nothing matched

    def __str__(self) -> str:
        return self.value


class Verdict:
    """The outcome of evaluating one rule against one response."""

    def __init__(self, exists: bool | None, confidence: QueryConfidence, reason: str):
        self.exists = exists  # None means undecided
        self.confidence = confidence
        self.reason = reason

    @property
    def is_decided(self) -> bool:
        return self.exists is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Verdict):
            return NotImplemented
        return (
            self.exists == other.exists
            and self.confidence == other.confidence
            and self.reason == other.reason
        )

    def __repr__(self) -> str:
        state = {True: "exists", False: "missing", None: "undecided"}[self.exists]
        return f"Verdict({state}, {self.confidence.value}, {self.reason!r})"


def _signature_matches(marker: str, body: str | None) -> bool | None:
    """Whether a marker is present in a body.

    Returns None when the question does not apply -- an empty marker (the rule
    is code-only) or a body that could not be read. Callers must distinguish
    "marker absent" from "could not look", because only the first is evidence.
    """
    if not marker:
        return None
    if not body:
        return None
    return marker in body


def evaluate(
    rule: dict,
    status_code: int | None,
    body: str | None,
) -> Verdict:
    """Evaluate a two-sided detection rule against a response.

    ``rule`` is the ``detection`` block produced by the WMN adapter::

        {"exists":  {"code": 200, "string": "profile-header"},
         "missing": {"code": 404, "string": "not found"}}
    """
    exists_rule = rule.get("exists") or {}
    missing_rule = rule.get("missing") or {}

    exists_code = exists_rule.get("code")
    missing_code = missing_rule.get("code")

    code_says_exists = status_code is not None and status_code == exists_code
    code_says_missing = status_code is not None and status_code == missing_code

    marker_says_exists = _signature_matches(exists_rule.get("string", ""), body)
    marker_says_missing = _signature_matches(missing_rule.get("string", ""), body)

    # Both markers present. Usually a stale rule -- but when one marker is a
    # substring of the other, the shorter match is an artifact of the longer
    # one, not independent evidence. Three dataset entries are written this way
    # (Udemy, Taringa, Independent academia), and reading them as contradictory
    # would make those sites permanently undecidable. The longer marker is
    # strictly more specific, so it wins.
    if marker_says_exists and marker_says_missing:
        hit_marker = exists_rule.get("string", "")
        miss_marker = missing_rule.get("string", "")

        if hit_marker in miss_marker:
            return Verdict(
                False,
                QueryConfidence.PROBABLE,
                "miss marker subsumes the hit marker; the longer match is more specific",
            )
        if miss_marker in hit_marker:
            return Verdict(
                True,
                QueryConfidence.PROBABLE,
                "hit marker subsumes the miss marker; the longer match is more specific",
            )

        return Verdict(
            None,
            QueryConfidence.AMBIGUOUS,
            "both the hit and miss markers matched; the rule is stale",
        )

    # Marker evidence first: it is specific to the site, codes are not.
    if marker_says_exists:
        if code_says_exists:
            return Verdict(True, QueryConfidence.CONFIRMED, "marker and status code agree")
        return Verdict(
            True,
            QueryConfidence.PROBABLE,
            f"hit marker matched but status was {status_code}, not {exists_code}",
        )

    if marker_says_missing:
        if code_says_missing:
            return Verdict(False, QueryConfidence.CONFIRMED, "marker and status code agree")
        return Verdict(
            False,
            QueryConfidence.PROBABLE,
            f"miss marker matched but status was {status_code}, not {missing_code}",
        )

    # No marker matched. Either the rule is code-only, the body was unreadable,
    # or the markers are genuinely absent -- and those mean different things.
    hit_marker_checked = marker_says_exists is not None
    miss_marker_checked = marker_says_missing is not None

    if code_says_exists and code_says_missing:
        # The rule's own author recorded that the status code carries no
        # information here (176 dataset entries look like this). The hit marker
        # is therefore the sole discriminator, and its checked absence is
        # evidence of absence -- not the two-signal contradiction handled below.
        if hit_marker_checked:
            return Verdict(
                False,
                QueryConfidence.PROBABLE,
                "status code cannot discriminate; the hit marker is absent",
            )
        return Verdict(
            None,
            QueryConfidence.AMBIGUOUS,
            "hit and miss share a status code with no distinguishing marker",
        )

    if code_says_exists:
        if hit_marker_checked:
            # The exact false positive the legacy model produced: the status
            # says yes, the page does not. A soft-404 looks precisely like this.
            return Verdict(
                None,
                QueryConfidence.AMBIGUOUS,
                f"status {status_code} matched but the hit marker was absent",
            )
        return Verdict(True, QueryConfidence.PROBABLE, "status code matched; rule has no marker to confirm")

    if code_says_missing:
        if miss_marker_checked:
            return Verdict(
                False,
                QueryConfidence.PROBABLE,
                f"status {status_code} matched but the miss marker was absent",
            )
        return Verdict(False, QueryConfidence.PROBABLE, "status code matched; rule has no marker to confirm")

    return Verdict(
        None,
        QueryConfidence.AMBIGUOUS,
        f"status {status_code} matched neither {exists_code} nor {missing_code}, and no marker matched",
    )


def is_rule_stale(verdict: Verdict) -> bool:
    """Whether a verdict indicates the site rule needs updating.

    An undecided verdict means the response looked like neither branch of the
    rule. Counted per site across scans, that is how dataset rot surfaces --
    a distinction the legacy negative-only model could not draw, because a
    broken rule and a found account produced the same output.
    """
    return not verdict.is_decided
