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


# A response carrying one of these did not deliver the resource, so nothing in
# its body describes the account -- a block page, challenge, or auth wall can
# still contain a string the rule treats as a hit marker. Never read a hit out
# of one unless the rule explicitly expects that code.
NON_CONTENT_CODES = frozenset({401, 403, 407, 429, 451, 500, 502, 503, 504})

# 404 and 410 are the web's unambiguous statement that a resource is not there,
# and no entry in the dataset treats either as a hit. They outrank a marker
# match, unlike an ordinary code disagreement: a generic marker such as
# 'username' or 'Timeline' will match somewhere in a large 404 page, which is
# how a site that correctly reports nothing still gets read as a hit.
ABSENCE_CODES = frozenset({404, 410})


def _signature_matches(marker: str, body: str | None) -> bool | None:
    """Whether a marker is present in a body.

    Returns None when the check could not be run -- an empty marker (the rule
    is code-only) or a body that could not be read. Callers must distinguish
    "marker absent" from "could not look", because only the first is evidence,
    and must further distinguish "rule has no marker" from "marker unchecked":
    trusting a status code is right in the first case and wrong in the second.
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

    # Checked before any marker, because the point is that the body is not
    # evidence at all: a response that did not deliver the resource is a block
    # page, challenge or auth wall, and those can contain the very string the
    # rule treats as a hit marker. A rule that expects the code is honoured --
    # some APIs genuinely answer 403 or 400 for a name that is taken.
    if (
        status_code in NON_CONTENT_CODES
        and status_code != exists_code
        and status_code != missing_code
    ):
        return Verdict(
            None,
            QueryConfidence.AMBIGUOUS,
            f"status {status_code} did not deliver the resource; body is not evidence",
        )

    marker_says_exists = _signature_matches(exists_rule.get("string", ""), body)
    marker_says_missing = _signature_matches(missing_rule.get("string", ""), body)

    # The one place a status code outranks a marker. It earns the exception by
    # being unambiguous where other codes are not: a 500 or a 403 says nothing
    # about whether the account exists, but a 404 says exactly that it does not.
    # Unlike the non-content codes above, the body here is the site's real
    # not-found page, so its miss marker is still worth reading -- it is what
    # separates a confirmed absence from a merely probable one.
    if status_code in ABSENCE_CODES and exists_code not in ABSENCE_CODES:
        # CONFIRMED means the same thing here as everywhere else: the code and
        # the marker both said so. An unread body is not agreement.
        confirmed = code_says_missing and marker_says_missing is True
        return Verdict(
            False,
            QueryConfidence.CONFIRMED if confirmed else QueryConfidence.PROBABLE,
            f"status {status_code} means the resource is not there",
        )

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

    # "The rule defines a marker" is not the same as "the marker was checked".
    # When a rule has a hit marker but the body never arrived, the discriminating
    # test did not run, and the status code alone must not stand in for it.
    rule_defines_hit_marker = bool(exists_rule.get("string"))

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
        if rule_defines_hit_marker:
            # The rule has a discriminator and it never got to run. Reddit
            # returns 200 with an empty body for names that do not exist, so
            # trusting the code here manufactures a hit out of nothing.
            return Verdict(
                None,
                QueryConfidence.AMBIGUOUS,
                f"status {status_code} matched but the body was empty, "
                "so the hit marker could not be checked",
            )
        return Verdict(True, QueryConfidence.PROBABLE, "status code matched; rule has no marker to confirm")

    if code_says_missing:
        # Deliberately more permissive than the hit side. A miss code is almost
        # always a 404 or a redirect, which is strong evidence on its own, and
        # the two errors are not equally costly: a fabricated account in an
        # investigation is far worse than a missed one.
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
