import pytest
from sherlock_project.playwright_engine import PlaywrightEngine

@pytest.mark.parametrize("invalid_method", ['', 'unkonwn_method'])
def test_get_request_fn_raise_error(invalid_method: str, playwright_engine: PlaywrightEngine):
    with pytest.raises(RuntimeError):
        playwright_engine.get_request_fn(invalid_method)

@pytest.mark.parametrize("valid_method", ['GET', 'POST', 'PUT', 'HEAD'])
def test_get_request_fn_returns_callable(valid_method: str, playwright_engine: PlaywrightEngine):
    method_fn = playwright_engine.get_request_fn(valid_method)
    assert callable(method_fn)
