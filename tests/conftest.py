import json
import os
import urllib

import pytest
import pytest_asyncio
import requests

from sherlock_project import ai_setup as ai_setup_module
from sherlock_project import sherlock as sherlock_module
from sherlock_project.database import SherlockDB
from sherlock_project.llama_server import ServerStatus
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.sites import SitesInformation
from sherlock_project.tui import settings_pane as settings_pane_module


@pytest.fixture(autouse=True)
def isolated_user_state(tmp_path, monkeypatch):
    """Point every per-user state location at tmp_path for the whole suite.

    Settings and the database resolve to per-user locations, not to the working
    directory, so without this the suite reads whatever the developer last saved
    with `sherlock-rm settings`. That is not hypothetical: a stored
    `webbrowser = false` made eight tests in test_synthesis_cli.py fail with
    `KeyError: 'concurrency'` minutes after being saved, with no change to the
    tests or the code.

    The failure mode this closes is worse than the noise. CI has no config file,
    so CI stayed green while the same commit failed locally -- meaning a
    contributor who had ever opened the settings editor got eight failures they
    did not cause, and the badge gave them no reason to suspect their own state.

    Resolution already honours these overrides -- default_database_path reads
    SHERLOCK_DB and ai_config_path reads SHERLOCK_CONFIG -- so this is isolation
    rather than new machinery. LLAMA_SERVER_BASE_URL is cleared for the same
    reason: it overrides the configured endpoint, and a developer who exported
    it should not thereby change what the suite tests.
    """
    monkeypatch.setenv("SHERLOCK_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("SHERLOCK_DB", str(tmp_path / "sherlock.db"))
    monkeypatch.delenv("LLAMA_SERVER_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def no_real_llama_server(monkeypatch):
    """Stop the suite starting, or killing, real llama-server processes.

    Same reasoning as `isolated_user_state`, one step further: a scan now
    launches a model server when one is not already listening. Left real, the
    suite would spawn multi-gigabyte processes on the developer's machine, take
    minutes, and behave differently depending on which models happen to be
    installed -- and it would adopt whatever server the developer had running,
    which is worse than slow because it silently passes.

    Patched at each integration point rather than on the class, so
    `tests/test_llama_server.py` -- which imports it directly and is about this
    behaviour -- still exercises the real thing.
    """
    class _StubServer:
        def __init__(self, _settings, **_kwargs) -> None:
            pass

        async def ensure_running(self) -> ServerStatus:
            return ServerStatus(
                running=True,
                started_by_us=False,
                detail="stubbed in tests",
            )

        async def stop(self) -> None:
            return None

    # Every module that starts a server. Missing one means that surface's
    # tests quietly spawn real processes, which is how this was nearly shipped
    # for the TUI picker.
    for module in (sherlock_module, ai_setup_module, settings_pane_module):
        monkeypatch.setattr(module, "ManagedLlamaServer", _StubServer)


def fetch_local_manifest(honor_exclusions: bool = True) -> dict[str, dict[str, str]]:
    sites_obj = SitesInformation(data_file_path=os.path.join(os.path.dirname(__file__), "../sherlock_project/resources/data.json"), honor_exclusions=honor_exclusions)
    sites_iterable: dict[str, dict[str, str]] = {site.name: site.information for site in sites_obj}
    return sites_iterable

@pytest_asyncio.fixture(scope='session', loop_scope='session')
async def playwright_engine():
    async with PlaywrightEngine() as engine:
        yield engine


@pytest_asyncio.fixture()
async def db() -> SherlockDB:
    db = await SherlockDB.create(":memory:")
    yield db
    await db.close()


@pytest.fixture()
def sites_obj():
    sites_obj = SitesInformation(data_file_path=os.path.join(os.path.dirname(__file__), "../sherlock_project/resources/data.json"))
    yield sites_obj

@pytest.fixture(scope="session")
def sites_info():
    yield fetch_local_manifest()


@pytest.fixture(scope="session")
def wmn_sites_info() -> dict[str, dict]:
    """The manifest the scanner actually runs on.

    sites_info still loads the legacy data.json, which the scan path no longer
    reads: those records carry errorType and no detection block, so probing
    them now yields UNKNOWN for every site.
    """
    sites_obj = SitesInformation()
    return {site.name: site.information for site in sites_obj}

@pytest.fixture(scope="session")
def remote_schema():
    schema_url: str = 'https://raw.githubusercontent.com/sherlock-project/sherlock/master/sherlock_project/resources/data.schema.json'
    with urllib.request.urlopen(schema_url) as remoteschema:
        schemadat = json.load(remoteschema)
    yield schemadat

def pytest_addoption(parser):
    parser.addoption(
        "--chunked-sites",
        action="store",
        default=None,
        help="For tests utilizing chunked sites, include only the (comma-separated) site(s) specified.",
    )

def pytest_generate_tests(metafunc):
    if "chunked_sites" in metafunc.fixturenames:
        # The manifest the scanner runs on. Parametrizing over the legacy
        # data.json gave every case a record with no detection block, so the
        # whole sweep could only ever report UNKNOWN.
        sites_info = {site.name: site.information for site in SitesInformation()}

        # Ingest and apply site selections
        site_filter: str | None = metafunc.config.getoption("--chunked-sites")
        if site_filter:
            selected_sites: list[str] = [site.strip() for site in site_filter.split(",")]
            sites_info = {
                site: data for site, data in sites_info.items()
                if site in selected_sites
            }

        params = [{name: data} for name, data in sites_info.items()]
        ids = list(sites_info.keys())
        metafunc.parametrize("chunked_sites", params, ids=ids)

def is_httpbin_up() -> bool:
    try:
        r = requests.get("https://httpbin.org/status/200", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False

@pytest.fixture(scope="session")
def httpbin_available():
    if not is_httpbin_up():
        pytest.skip("httpbin.org is unavailable")