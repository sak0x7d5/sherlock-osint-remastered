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


def test_browser_run_rechecks_what_the_fast_transport_could_not_confirm():
    """The rule that stops one fast run from degrading a stored record.

    Without it the fast transport answers every site cheaply, the resume filter
    skips them forever, and a later browser scan silently inherits answers a
    browser never gave.
    """
    unconfirmed = {"status": str(QueryStatus.AVAILABLE), "transport": "http"}
    blocked = {"status": str(QueryStatus.UNKNOWN), "transport": "http"}
    found = {"status": str(QueryStatus.CLAIMED), "transport": "http"}

    assert sherlock.is_resumable(unconfirmed, using_browser=True) is False
    assert sherlock.is_resumable(blocked, using_browser=True) is False
    # A marker proving the account exists was actually found. Finding it
    # without JavaScript does not make it less found -- the failure this rule
    # defends against is an empty page read as a confident absence.
    assert sherlock.is_resumable(found, using_browser=True) is True


def test_browser_results_are_never_re_checked_by_a_fast_run():
    """Replacing browser-grade evidence with something weaker is a downgrade."""
    row = {"status": str(QueryStatus.AVAILABLE), "transport": "browser"}

    assert sherlock.is_resumable(row, using_browser=False) is True
    assert sherlock.is_resumable(row, using_browser=True) is True


def test_rows_predating_the_transport_column_are_resumable():
    """NULL means "unknown", and unknown rows are overwhelmingly browser rows.

    Treating them as suspect would make the first run after this change
    re-scan every site every user has ever stored.
    """
    row = {"status": str(QueryStatus.AVAILABLE)}

    assert sherlock.is_resumable(row, using_browser=True) is True


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
