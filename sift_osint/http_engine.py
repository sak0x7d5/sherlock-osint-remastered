"""The browser-free fetch transport: plain HTTPS requests through httpx.

Chosen by turning the ``webbrowser`` setting off. The win is not really
per-request speed -- it is that Chromium never starts at all, so a scan pays no
browser startup and no page rendering.

This is the INACCURATE half of that choice, and it is written down here rather
than buried in a docs page. A plain request runs no JavaScript, so a profile
built in the browser arrives as an empty shell without the marker its rule looks
for. That is a miss. The worse case is a login wall that happens to contain the
rule's *miss* marker: the scan then reports a confident ABSENCE for an account
that exists, which in an OSINT context is the error that gets acted on. An
earlier attempt to route sites here automatically was reverted for exactly this
-- see ``preferred_transport`` in wmn_adapter.py. It is offered as a user's
choice, announced before the scan, and recorded against every row it produces
(``results.transport``), because months later the database is the only thing
that still knows how an answer was obtained.

The surface mirrors PlaywrightEngine on purpose -- same async context manager,
same ``get_request_fn`` / ``fetch_with_page`` / ``fetch_with_api`` -- so the
scan loop stays transport-agnostic and neither engine has to know the other
exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Self

import httpx

CancellationCallback = Callable[[], None]


class ProxyConfigurationError(ValueError):
    """A proxy setting no transport can use. Reported, never raised at a user."""

# A browser-shaped costume, not a disguise. Without it httpx announces itself as
# `python-httpx/x.y`, which plenty of sites answer with a 403 that has nothing
# to do with whether the account exists. It does NOT make this look like a
# browser to anything that inspects the TLS handshake -- the stealth browser's
# fingerprint is precisely what this transport gives up, so expect the 58
# protection-flagged sites in the manifest to go worse here, not better.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}

SUPPORTED_METHODS = ("GET", "HEAD", "POST", "PUT")

# Schemes both transports can actually route through. Playwright accepts the
# first three; httpx adds socks5h. Anything else is rejected with a message
# rather than allowed through to fail deep inside a client constructor.
SUPPORTED_PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")


def normalize_proxy(proxy: str | None) -> str | None:
    """Accept ``host:port`` the way the browser transport already does.

    httpx refuses a proxy URL with no scheme -- and refuses it by raising
    ValueError from inside AsyncClient's constructor, which is not a place any
    caller can report from. Since `127.0.0.1:8080` is one of the commonest ways
    to write a proxy, and Playwright normalises it silently, assume http rather
    than making one transport pickier than the other about the same string.
    """
    if not proxy:
        return None
    proxy = proxy.strip()
    if not proxy:
        return None
    if "://" not in proxy:
        return f"http://{proxy}"
    return proxy


@dataclass(slots=True)
class HttpFetchResponse:
    """What the scan reads off a response, and nothing more.

    httpx.Response cannot be used directly: the scan attaches ``profile_text``
    to a response and the Playwright paths assign ``text`` and ``elapsed`` onto
    theirs, while on httpx.Response all three are read-only properties. A small
    record of exactly the fields detection and storage consume is clearer than
    working around that -- which is also why the final URL is not kept: nothing
    reads it, and carrying it would imply redirect provenance is being stored
    when it is not.
    """

    status: int
    text: str
    elapsed: float
    profile_text: str | None = None


class HttpEngine:
    """Fetch sites with plain HTTPS requests. No browser, no JavaScript."""

    # Every result from this engine carries one label, unlike the browser
    # engine where the transport varies per site rule. The scan reads this to
    # decide what to record against each row.
    fixed_transport = "http"

    def __init__(
        self,
        concurrency: int = 30,
        proxy: str | None = None,
        cancellation_callback: CancellationCallback | None = None,
        user_agent: str | None = None,
    ):
        self.sem = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency
        self.proxy = normalize_proxy(proxy)
        self.cancellation_callback = cancellation_callback
        self.client: httpx.AsyncClient | None = None
        # Built once the client exists. Every site fetch is a GET, so rebuilding
        # this closure per request allocated one throwaway function per site.
        self._get: Callable[..., Any] | None = None
        headers = dict(DEFAULT_HEADERS)
        if user_agent:
            headers["User-Agent"] = user_agent
        self.headers = headers

    async def __aenter__(self) -> Self:
        # Checked here rather than left to httpx, which raises ValueError from
        # inside its constructor -- a raw traceback out of `async with`, with
        # no caller in a position to turn it into a message.
        scheme = (self.proxy or "://").split("://", 1)[0].lower()
        if self.proxy and scheme not in SUPPORTED_PROXY_SCHEMES:
            raise ProxyConfigurationError(
                f"Unsupported proxy {self.proxy!r}: expected one of "
                f"{', '.join(SUPPORTED_PROXY_SCHEMES)}"
            )

        self.client = httpx.AsyncClient(
            headers=self.headers,
            # Browsers follow redirects, and a rule's markers were written
            # against wherever the site finally lands. Stopping at the 301
            # would decide every redirecting site on an empty body.
            follow_redirects=True,
            # Mirrors the browser context's ignore_https_errors. A site with a
            # broken certificate is still a site whose rule can be evaluated,
            # and refusing it here would report "no answer" where the browser
            # transport reports a real one -- a difference in accuracy that has
            # nothing to do with JavaScript.
            verify=False,
            proxy=self.proxy,
            limits=httpx.Limits(
                max_connections=max(self.concurrency, 10),
                max_keepalive_connections=max(self.concurrency, 10),
            ),
        )
        self._get = self.get_request_fn("GET")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            self._notify_cancellation(exc)
        client = self.client
        self.client = None
        self._get = None
        if client is not None:
            try:
                await client.aclose()
            except BaseException as cleanup_error:
                if exc is None:
                    raise
                exc.add_note(f"HTTP client cleanup also failed: {cleanup_error!r}")
        return False

    def _notify_cancellation(self, cancellation: BaseException) -> None:
        if self.cancellation_callback is None:
            return
        try:
            self.cancellation_callback()
        except BaseException as callback_error:
            cancellation.add_note(
                f"HTTP cancellation callback failed: {callback_error!r}"
            )

    def _require_client(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("HttpEngine used outside its context manager")
        return self.client

    def get_request_fn(self, method: str) -> Callable[..., Any]:
        """Bind a method to a callable shaped like the Playwright engine's.

        The scan asks for a request function and then hands it to
        ``fetch_with_api``, so the two engines have to agree on the shape
        rather than on the library underneath.
        """
        normalized = method.upper()
        if normalized not in SUPPORTED_METHODS:
            raise RuntimeError(f"Unsupported request_method: {method}")

        async def request(
            url: str,
            timeout: float = 60000,
            headers: dict | None = None,
            data: Any | bytes | str | None = None,
        ) -> httpx.Response:
            client = self._require_client()
            # A dict payload is JSON, matching Playwright's `data=`; anything
            # else goes out as a raw body. Getting this wrong sends the literal
            # text "{'user': 'x'}" and every POST-checked site answers 400.
            body: dict[str, Any] = {}
            if isinstance(data, dict):
                body["json"] = data
            elif data is not None:
                body["content"] = data
            return await client.request(
                normalized,
                url,
                # The engine interface speaks milliseconds because Playwright
                # does; httpx speaks seconds.
                timeout=timeout / 1000,
                headers=headers or None,
                **body,
            )

        return request

    async def fetch_with_page(
        self,
        url: str,
        headers: dict | None = None,
        timeout: float = 60000,
        wait_until: str | None = None,
    ) -> HttpFetchResponse | None:
        """Fetch what a person would open -- as a plain GET.

        Named for the browser engine's method rather than for what it does, so
        the scan can call either without asking which it holds. ``wait_until``
        is accepted and ignored: there is no page to wait for, and that absence
        is the entire accuracy cost of this transport.
        """
        # Falls back to building one so that calling this outside the context
        # manager still reaches _require_client's error rather than a None call.
        request_fn = self._get or self.get_request_fn("GET")
        return await self._perform(request_fn, url=url, headers=headers, timeout=timeout)

    async def fetch_with_api(
        self,
        request_fn: Callable[..., Any],
        url: str,
        headers: dict | None = None,
        timeout: float = 60000,
        request_payload: Any | bytes | str | None = None,
    ) -> HttpFetchResponse | None:
        return await self._perform(
            request_fn,
            url=url,
            headers=headers,
            timeout=timeout,
            request_payload=request_payload,
        )

    async def _perform(
        self,
        request_fn: Callable[..., Any],
        *,
        url: str,
        headers: dict | None,
        timeout: float,
        request_payload: Any | bytes | str | None = None,
    ) -> HttpFetchResponse | None:
        async with self.sem:
            start = perf_counter()
            response = await request_fn(
                url=url,
                timeout=timeout,
                headers=headers,
                data=request_payload,
            )
            if response is None:
                return None

            # Decoding can fail on a body that is not text at all. An
            # undecodable response is still evidence -- the status code alone
            # decides a good number of rules -- so it arrives with an empty
            # body rather than as an exception, matching what the browser
            # engine does with a page it cannot read.
            try:
                text = response.text
            except Exception:
                text = ""

            return HttpFetchResponse(
                status=response.status_code,
                text=text,
                elapsed=perf_counter() - start,
            )
