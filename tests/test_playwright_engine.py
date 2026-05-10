import pytest
import pytest_asyncio
from sherlock_project.playwright_engine import PlaywrightEngine
import asyncio

@pytest.mark.parametrize("invalid_method", ['', 'unkonwn_method'])
def test_get_request_fn_raise_error(invalid_method: str, playwright_engine: PlaywrightEngine):
    with pytest.raises(RuntimeError):
        playwright_engine.get_request_fn(invalid_method)

@pytest.mark.parametrize("valid_method", ['GET', 'POST', 'PUT', 'HEAD'])
def test_get_request_fn_returns_callable(valid_method: str, playwright_engine: PlaywrightEngine):
    method_fn = playwright_engine.get_request_fn(valid_method)
    assert callable(method_fn)

@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "status,url", [
        (302, r'https://httpbin.org/redirect-to?url=https%3A%2F%2Fhttpbin.org%2F&status_code=302'),
        (200, r'https://httpbin.org/status/200')
        ])
async def test_handle_route_redirect_site(status: int, url: str, playwright_engine: PlaywrightEngine):
    resp = await playwright_engine.fetch_site(url=url, max_redirects=0)
    assert resp.status == status
