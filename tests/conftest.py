import os
import json
import urllib
import pytest
import pytest_asyncio
import requests
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.sites import SitesInformation
from sherlock_project.database import SherlockDB


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
        sites_info = fetch_local_manifest(honor_exclusions=False)

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