"""The browser-free transport.

Everything here runs against httpx's MockTransport rather than a network, so
the assertions are about the contract the scan depends on -- response shape,
units, payload encoding -- and not about any site's behaviour.
"""

from __future__ import annotations

import httpx
import pytest

from sherlock_project.http_engine import (
    DEFAULT_USER_AGENT,
    HttpEngine,
    HttpFetchResponse,
    ProxyConfigurationError,
)


async def _engine_with(handler) -> HttpEngine:
    """An entered engine whose client answers from `handler`.

    The real client is closed and replaced rather than injected through the
    constructor: httpx's transport plumbing is an implementation detail of this
    engine, and widening the constructor for tests would make it part of the
    interface.
    """
    engine = HttpEngine(concurrency=2)
    await engine.__aenter__()
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(
        headers=engine.headers,
        follow_redirects=True,
        transport=httpx.MockTransport(handler),
    )
    return engine


def test_every_result_from_this_engine_is_labelled_http():
    """The scan reads this to decide what to record against each row.

    A browser-free engine that left the label to the per-site rule would store
    "browser" against results no browser ever saw -- the exact confusion the
    column was added to prevent.
    """
    assert HttpEngine.fixed_transport == "http"


async def test_fetch_with_page_returns_what_detection_reads():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>found</html>")

    engine = await _engine_with(handler)
    try:
        response = await engine.fetch_with_page(url="https://example.test/blue")
    finally:
        await engine.__aexit__(None, None, None)

    assert isinstance(response, HttpFetchResponse)
    assert (response.status, response.text) == (200, "<html>found</html>")
    assert response.elapsed >= 0
    assert response.profile_text is None


async def test_timeout_is_converted_from_milliseconds():
    """The engine interface speaks Playwright's units; httpx speaks seconds.

    Passing 60000 straight through would mean a 60,000-second timeout: a
    site that never answers would hang the scan rather than failing it.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions.get("timeout") or {})
        return httpx.Response(200, text="")

    engine = await _engine_with(handler)
    try:
        await engine.fetch_with_page(url="https://example.test/blue", timeout=15000)
    finally:
        await engine.__aexit__(None, None, None)

    assert seen["read"] == 15


async def test_string_payload_is_sent_verbatim_under_the_site_content_type():
    """The shape every POST site in the manifest actually uses.

    Measured against the adapted manifest: 22 sites POST, and all 22 carry
    request_payload as a pre-formatted `str` with their own Content-Type --
    application/json for 14, x-www-form-urlencoded for 8. Re-encoding the body
    or overriding that header would make the endpoint answer 400, which the
    scan would report as an inconclusive site rather than as a bug.
    """
    sent: list[tuple[bytes, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((request.content, request.headers["content-type"]))
        return httpx.Response(200, text="")

    engine = await _engine_with(handler)
    try:
        request_fn = engine.get_request_fn("POST")
        await engine.fetch_with_api(
            request_fn=request_fn,
            url="https://example.test/api",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            request_payload="username=blue",
        )
    finally:
        await engine.__aexit__(None, None, None)

    assert sent == [(b"username=blue", "application/x-www-form-urlencoded")]


async def test_dict_payload_is_sent_as_json():
    """Defensive: no site uses a mapping today, but Playwright's `data=` accepts
    one, so the engines must not disagree if the manifest ever grows one.

    Sent raw, a dict would arrive as its Python repr and the endpoint would
    answer 400.
    """
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(200, text="")

    engine = await _engine_with(handler)
    try:
        request_fn = engine.get_request_fn("POST")
        await engine.fetch_with_api(
            request_fn=request_fn,
            url="https://example.test/api",
            request_payload={"username": "blue"},
        )
    finally:
        await engine.__aexit__(None, None, None)

    assert bodies == [b'{"username":"blue"}']


async def test_a_schemeless_proxy_is_accepted_like_the_browser_accepts_it():
    """`host:port` must not mean two different things per transport.

    httpx refuses a schemeless proxy by raising ValueError from inside its
    client constructor -- a raw traceback out of `async with`, exit 1 -- while
    the browser transport took the identical string and ran.
    """
    engine = HttpEngine(proxy="127.0.0.1:8080")

    assert engine.proxy == "http://127.0.0.1:8080"


async def test_an_unusable_proxy_scheme_is_reported_not_raised_from_httpx():
    engine = HttpEngine(proxy="ftp://127.0.0.1:8080")

    with pytest.raises(ProxyConfigurationError, match="Unsupported proxy"):
        await engine.__aenter__()


async def test_requests_do_not_announce_themselves_as_python():
    """A default httpx User-Agent earns 403s that have nothing to do with the
    account existing, which would read as a detection failure."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, text="")

    engine = await _engine_with(handler)
    try:
        await engine.fetch_with_page(url="https://example.test/blue")
    finally:
        await engine.__aexit__(None, None, None)

    assert seen == [DEFAULT_USER_AGENT]


async def test_per_site_headers_override_the_defaults():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["accept"])
        return httpx.Response(200, text="")

    engine = await _engine_with(handler)
    try:
        await engine.fetch_with_page(
            url="https://example.test/blue",
            headers={"Accept": "application/json"},
        )
    finally:
        await engine.__aexit__(None, None, None)

    assert seen == ["application/json"]


async def test_redirects_are_followed():
    """A rule's markers describe where the site lands, not the 301 it sends."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/blue":
            return httpx.Response(302, headers={"Location": "/users/blue"})
        return httpx.Response(200, text="profile")

    engine = await _engine_with(handler)
    try:
        response = await engine.fetch_with_page(url="https://example.test/blue")
    finally:
        await engine.__aexit__(None, None, None)

    assert (response.status, response.text) == (200, "profile")


async def test_unsupported_method_is_rejected():
    engine = HttpEngine()
    with pytest.raises(RuntimeError, match="Unsupported request_method"):
        engine.get_request_fn("TRACE")


async def test_use_outside_the_context_manager_is_an_error():
    """Better than a None-dereference two frames down inside a site task."""
    engine = HttpEngine()
    request_fn = engine.get_request_fn("GET")
    with pytest.raises(RuntimeError, match="outside its context manager"):
        await request_fn(url="https://example.test/blue")
