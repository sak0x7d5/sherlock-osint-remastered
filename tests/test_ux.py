import pytest
from sherlock_interactives import Interactives, InteractivesSubprocessError

from sherlock_project import sherlock
from sherlock_project.result import QueryStatus


def test_remove_nsfw(sites_obj):
    nsfw_target: str = 'Xvideos'
    assert nsfw_target in {site.name: site.information for site in sites_obj}
    sites_obj.remove_nsfw_sites()
    assert nsfw_target not in {site.name: site.information for site in sites_obj}


# Parametrized sites should *not* include Motherless, which is acting as the control
@pytest.mark.parametrize('nsfwsites', [
    ['Xvideos'],
    ['Xvideos', 'Erome'],
])
def test_nsfw_explicit_selection(sites_obj, nsfwsites):
    for site in nsfwsites:
        assert site in {site.name: site.information for site in sites_obj}
    sites_obj.remove_nsfw_sites(do_not_remove=nsfwsites)
    for site in nsfwsites:
        assert site in {site.name: site.information for site in sites_obj}
        assert 'Motherless' not in {site.name: site.information for site in sites_obj}

def test_wildcard_username_expansion():
    assert sherlock.check_for_parameter('test{?}test') is True
    assert sherlock.check_for_parameter('test{.}test') is False
    assert sherlock.check_for_parameter('test{}test') is False
    assert sherlock.check_for_parameter('testtest') is False
    assert sherlock.check_for_parameter('test{?test') is False
    assert sherlock.check_for_parameter('test?}test') is False
    assert sherlock.multiple_usernames('test{?}test') == ["test_test" , "test-test" , "test.test"]


@pytest.mark.parametrize('cliargs', [
    '',
    '--site urghrtuight --egiotr',
    '--',
])
def test_no_usernames_provided(cliargs):
    with pytest.raises(InteractivesSubprocessError, match=r"error: the following arguments are required: USERNAMES"):
        Interactives.run_cli(cliargs)


def test_restore_saved_results_rebuilds_scan_shape():
    """A repeat scan must still report hits that were found the first time.

    The scan only returns sites it actually checked, so without restoration a
    second run reports nothing and writes empty exports while every hit is
    sitting in SQLite.
    """
    saved_rows = {
        "GitHub": {
            "site_name": "GitHub",
            "site_url": "https://github.com/blue",
            "status": "Claimed",
            "status_code": 200,
            "query_time_ms": 0.42,
            "error_context": None,
            "confidence": "Probable",
        },
        "Reddit": {
            "site_name": "Reddit",
            "site_url": "https://reddit.com/u/blue",
            "status": "Available",
            "status_code": 404,
            "query_time_ms": 0.11,
            "error_context": None,
            "confidence": None,
        },
    }
    site_data_all = {
        "GitHub": {"urlMain": "https://github.com"},
        "Reddit": {"urlMain": "https://reddit.com"},
    }

    restored = sherlock.restore_saved_results(
        username="blue",
        saved_rows=saved_rows,
        site_data_all=site_data_all,
    )

    assert set(restored) == {"GitHub", "Reddit"}

    github = restored["GitHub"]
    # Same keys the exporters read off a live scan result.
    assert github["url_main"] == "https://github.com"
    assert github["url_user"] == "https://github.com/blue"
    assert github["http_status"] == 200
    assert github["status"].status is QueryStatus.CLAIMED
    assert str(github["status"].confidence) == "Probable"
    assert github["status"].query_time == 0.42

    assert restored["Reddit"]["status"].status is QueryStatus.AVAILABLE
    assert restored["Reddit"]["status"].confidence is None


def test_restore_saved_results_skips_unreadable_rows():
    """One bad row must not cost the user the rest of their stored results."""
    saved_rows = {
        "Good": {
            "site_url": "https://good.example/blue",
            "status": "Claimed",
            "status_code": 200,
            "query_time_ms": None,
            "error_context": None,
            "confidence": "nonsense-confidence",
        },
        "Bad": {
            "site_url": "https://bad.example/blue",
            "status": "not-a-real-status",
            "status_code": None,
            "query_time_ms": None,
            "error_context": None,
            "confidence": None,
        },
    }

    restored = sherlock.restore_saved_results(
        username="blue",
        saved_rows=saved_rows,
        site_data_all={},
    )

    assert set(restored) == {"Good"}
    # An unparseable confidence degrades to None rather than dropping the hit.
    assert restored["Good"]["status"].confidence is None
    assert restored["Good"]["url_main"] is None
