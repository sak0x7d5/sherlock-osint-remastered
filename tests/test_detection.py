import os

import pytest

from sherlock_project.detection import (
    QueryConfidence,
    evaluate,
    is_rule_stale,
)
from sherlock_project.wmn_adapter import load_wmn_manifest

HIT = "profile-header"
MISS = "no such user"


def rule(e_code=200, e_string=HIT, m_code=404, m_string=MISS) -> dict:
    return {
        "exists": {"code": e_code, "string": e_string},
        "missing": {"code": m_code, "string": m_string},
    }


class TestAgreement:
    def test_code_and_marker_both_say_exists(self):
        verdict = evaluate(rule(), 200, f"<div class='{HIT}'>")
        assert verdict.exists is True
        assert verdict.confidence is QueryConfidence.CONFIRMED

    def test_code_and_marker_both_say_missing(self):
        verdict = evaluate(rule(), 404, f"<p>{MISS}</p>")
        assert verdict.exists is False
        assert verdict.confidence is QueryConfidence.CONFIRMED

    def test_same_code_both_sides_is_decided_by_marker(self):
        """The 176 sites where the status code carries no information."""
        same = rule(e_code=200, m_code=200)
        assert evaluate(same, 200, f"x{HIT}x").exists is True
        assert evaluate(same, 200, f"x{MISS}x").exists is False


class TestMarkerOutranksCode:
    def test_marker_wins_when_code_disagrees(self):
        verdict = evaluate(rule(), 500, f"<div>{HIT}</div>")
        assert verdict.exists is True
        assert verdict.confidence is QueryConfidence.PROBABLE
        assert "500" in verdict.reason

    def test_miss_marker_wins_when_code_disagrees(self):
        verdict = evaluate(rule(), 200, f"<p>{MISS}</p>")
        assert verdict.exists is False
        assert verdict.confidence is QueryConfidence.PROBABLE


class TestRefusalToGuess:
    def test_soft_404_is_not_a_hit(self):
        """The exact false positive the legacy model produced.

        Status says 200, but the profile marker is absent -- a "no such user"
        page served with 200. The old status_code rule called this CLAIMED.
        """
        verdict = evaluate(rule(), 200, "<html>nothing here</html>")
        assert verdict.exists is None
        assert verdict.confidence is QueryConfidence.AMBIGUOUS
        assert is_rule_stale(verdict)

    def test_nothing_matches_at_all(self):
        verdict = evaluate(rule(), 503, "<html>gateway timeout</html>")
        assert verdict.exists is None
        assert is_rule_stale(verdict)

    def test_contradictory_markers_do_not_average_out(self):
        verdict = evaluate(rule(), 200, f"{HIT} ... {MISS}")
        assert verdict.exists is None
        assert verdict.confidence is QueryConfidence.AMBIGUOUS
        assert "stale" in verdict.reason


class TestNestedMarkers:
    """Three dataset entries write one marker as a substring of the other.

    A shorter marker matching inside a longer one is an artifact of the longer
    match, not a second opinion -- so the more specific marker wins rather than
    the site becoming permanently undecidable.
    """

    def test_miss_marker_subsumes_hit_marker(self):
        """Taringa/Udemy shape: the miss title contains the hit title."""
        nested = {
            "exists": {"code": 200, "string": " en Taringa!</title>"},
            "missing": {"code": 200, "string": "Colectiva en Taringa!</title>"},
        }
        assert evaluate(nested, 200, "<title>Colectiva en Taringa!</title>").exists is False
        assert evaluate(nested, 200, "<title>alice en Taringa!</title>").exists is True

    def test_hit_marker_subsumes_miss_marker(self):
        """Independent academia shape: the hit marker contains the miss marker."""
        nested = {
            "exists": {"code": 200, "string": "- Academia.edu"},
            "missing": {"code": 200, "string": "Academia.edu"},
        }
        assert evaluate(nested, 200, "<title>alice - Academia.edu</title>").exists is True
        assert evaluate(nested, 200, "<title>Academia.edu</title>").exists is False

    def test_unrelated_markers_both_matching_is_still_stale(self):
        verdict = evaluate(rule(), 200, f"{HIT}{MISS}")
        assert verdict.exists is None
        assert "stale" in verdict.reason


class TestNonDiscriminatingCode:
    """When e_code == m_code the marker is the only discriminator.

    Its checked absence is evidence of absence -- unlike the soft-404 case,
    where the code is a genuine second signal that disagrees.
    """

    def test_absent_marker_means_missing(self):
        same = rule(e_code=200, m_code=200, m_string="")
        verdict = evaluate(same, 200, "<html>unrelated content</html>")
        assert verdict.exists is False
        assert verdict.confidence is QueryConfidence.PROBABLE
        assert "cannot discriminate" in verdict.reason

    def test_unreadable_body_stays_undecided(self):
        """Absent is evidence; unread is not."""
        same = rule(e_code=200, m_code=200, m_string="")
        assert evaluate(same, 200, "").exists is None

    def test_asymmetry_with_soft_404(self):
        """The same absent marker resolves differently when codes differ."""
        body = "<html>unrelated content</html>"
        assert evaluate(rule(e_code=200, m_code=200, m_string=""), 200, body).exists is False
        assert evaluate(rule(e_code=200, m_code=404), 200, body).exists is None

    def test_undecidable_rule_reports_rather_than_guesses(self):
        undecidable = rule(e_code=200, e_string="", m_code=200, m_string="")
        verdict = evaluate(undecidable, 200, "<html></html>")
        assert verdict.exists is None
        assert verdict.confidence is QueryConfidence.AMBIGUOUS


class TestCodeOnlyRules:
    """The two dataset entries with no e_string, both with distinct codes."""

    def test_code_only_hit(self):
        code_only = rule(e_code=302, e_string="", m_code=404, m_string="")
        verdict = evaluate(code_only, 302, "")
        assert verdict.exists is True
        assert verdict.confidence is QueryConfidence.PROBABLE

    def test_code_only_miss(self):
        code_only = rule(e_code=302, e_string="", m_code=404, m_string="")
        verdict = evaluate(code_only, 404, "")
        assert verdict.exists is False

    def test_code_only_neither(self):
        code_only = rule(e_code=302, e_string="", m_code=404, m_string="")
        assert evaluate(code_only, 200, "").exists is None


class TestUnreadableBody:
    """An absent marker is evidence; an unread body is not."""

    @pytest.mark.parametrize("body", [None, ""])
    def test_empty_body_does_not_count_as_marker_absent(self, body):
        verdict = evaluate(rule(), 200, body)
        # Cannot confirm, but must not be reported as a confident miss either.
        assert verdict.exists is True
        assert verdict.confidence is QueryConfidence.PROBABLE
        assert "no marker to confirm" in verdict.reason

    def test_empty_body_with_missing_code(self):
        verdict = evaluate(rule(), 404, None)
        assert verdict.exists is False
        assert verdict.confidence is QueryConfidence.PROBABLE

    def test_empty_body_with_unrelated_code_stays_undecided(self):
        assert evaluate(rule(), 500, None).exists is None


class TestEdgeCases:
    def test_no_status_code(self):
        assert evaluate(rule(), None, "<html></html>").exists is None

    def test_marker_match_is_case_sensitive(self):
        verdict = evaluate(rule(), 200, "PROFILE-HEADER")
        assert verdict.exists is None

    def test_marker_matches_as_substring(self):
        assert evaluate(rule(), 200, f'<div class="x {HIT} y">').exists is True

    def test_empty_rule_is_undecided(self):
        assert evaluate({}, 200, "anything").exists is None

    def test_reason_is_always_populated(self):
        for status, body in [(200, HIT), (404, MISS), (500, ""), (200, "")]:
            assert evaluate(rule(), status, body).reason


class TestRealDatasetRules:
    """Rules taken verbatim from the vendored dataset."""

    def test_lichess_weak_marker(self):
        """e_code == m_code == 200 with a 'true'/'false' body."""
        lichess = {
            "exists": {"code": 200, "string": "true"},
            "missing": {"code": 200, "string": "false"},
        }
        assert evaluate(lichess, 200, "true").exists is True
        assert evaluate(lichess, 200, "false").exists is False

    def test_generic_id_marker_still_needs_the_marker(self):
        """GitLab/Keybase style: '\"id\":' with identical codes."""
        gitlab = {
            "exists": {"code": 200, "string": '"id":'},
            "missing": {"code": 200, "string": ""},
        }
        assert evaluate(gitlab, 200, '{"id":123,"username":"x"}').exists is True
        # An error envelope without an id must not read as a hit.
        assert evaluate(gitlab, 200, '{"message":"404 Not Found"}').exists is False


class TestEveryRuleInTheDataset:
    """Dataset/logic coherence over all vendored rules.

    Guards both directions at once: a rule whose own markers do not reproduce
    its own verdict is either a dataset defect or a regression here. This is
    what caught the nested-marker and non-discriminating-code cases.
    """

    FILLER = "<html><body>unrelated page content</body></html>"

    @pytest.fixture(scope="class")
    def sites(self) -> dict:
        path = os.path.join(
            os.path.dirname(__file__), "../sherlock_project/resources/wmn-data.json"
        )
        adapted, _ = load_wmn_manifest(path)
        return adapted

    def test_every_rule_detects_its_own_hit(self, sites):
        failures = []
        for name, record in sites.items():
            exists = record["detection"]["exists"]
            body = self.FILLER + exists["string"]
            verdict = evaluate(record["detection"], exists["code"], body)
            if verdict.exists is not True:
                failures.append(f"{name}: {verdict.reason}")
        assert not failures, f"{len(failures)} rule(s) fail their own hit: {failures[:5]}"

    def test_every_rule_detects_its_own_miss(self, sites):
        failures = []
        for name, record in sites.items():
            missing = record["detection"]["missing"]
            body = self.FILLER + missing["string"]
            verdict = evaluate(record["detection"], missing["code"], body)
            if verdict.exists is not False:
                failures.append(f"{name}: {verdict.reason}")
        assert not failures, f"{len(failures)} rule(s) fail their own miss: {failures[:5]}"
