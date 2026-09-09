import asyncio
import os
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal, Protocol

from playwright.async_api import (
    APIRequestContext,
    APIResponse,
    Browser,
    BrowserContext,
    Page,
    Response,
)

os.environ.setdefault("CLOAKBROWSER_AUTO_UPDATE", "false")

from cloakbrowser import binary_info, ensure_binary, launch_async

BrowserStatus = Literal["installing", "starting", "ready"]
BrowserStatusCallback = Callable[[BrowserStatus], None]
CancellationCallback = Callable[[], None]

class BrowserUnavailable(RuntimeError):
    """The stealth browser could not be obtained or started.

    Distinct from any other startup failure because it is the one the user can
    do something about: the scan has a browser-free transport, and this is the
    condition under which recommending it is useful rather than noise. It is
    raised only from `__aenter__`, so catching it upstream cannot swallow an
    error from the scan itself.

    Downloading the binary is a network operation against a third-party host,
    which makes this ordinary rather than exotic -- a proxy, an offline
    machine, TLS interception or a 403 on the release asset all land here, and
    all of them used to surface as sixty lines of httpx traceback.

    Cancellation is deliberately NOT converted. `asyncio.CancelledError` and
    `KeyboardInterrupt` derive from BaseException rather than Exception, so
    the conversion below cannot catch them and Ctrl-C keeps unwinding as it
    should.
    """


class RequestMethod(Protocol):
    async def __call__(self, url: str, **kwargs: Any) -> Any: ...

class PlaywrightEngine:
    # None means "the scan decides per site": a rule needing a POST goes out on
    # the API transport, everything else renders in the browser. The
    # browser-free engine pins this to one value instead -- see HttpEngine.
    fixed_transport: str | None = None

    def __init__(
        self,
        concurrency: int = 30,
        headless: bool = True,
        proxy: dict | None = None,
        status_callback: BrowserStatusCallback | None = None,
        cancellation_callback: CancellationCallback | None = None,
    ):
        self.sem = asyncio.Semaphore(concurrency)
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.api: APIRequestContext | None = None
        self._fn_mapping: dict[str, RequestMethod] = {}
        self.headless = headless
        self.proxy = proxy
        self.status_callback = status_callback
        self.cancellation_callback = cancellation_callback

    async def __aenter__(self):
        try:
            self.ensure_browser_binary(self.status_callback)

            self._notify("starting")
            self.browser = await launch_async(
                headless=self.headless,
                humanize=True,
                handle_sigint=False,
            )
            self.context = await self.browser.new_context(
                ignore_https_errors=True,
                proxy=self.proxy,
            )
            self.api = self.context.request
            self._notify("ready")

            # Initialize mappings when context is not None.
            self._fn_mapping = {
                'GET': self.api.get,
                'HEAD': self.api.head,
                'POST': self.api.post,
                'PUT': self.api.put,
            }
        except BaseException as startup_error:
            if isinstance(
                startup_error,
                (asyncio.CancelledError, KeyboardInterrupt),
            ):
                self._notify_cancellation(startup_error)
            try:
                await self._close_resources()
            except BaseException as cleanup_error:
                startup_error.add_note(
                    f"Playwright cleanup also failed: {cleanup_error!r}"
                )
            if isinstance(startup_error, Exception):
                raise BrowserUnavailable(
                    str(startup_error) or type(startup_error).__name__
                ) from startup_error
            raise

        return self
    

    def _notify(self, status: BrowserStatus) -> None:
        if self.status_callback is not None:
            self.status_callback(status)

    @staticmethod
    def ensure_browser_binary(
        status_callback: BrowserStatusCallback | None = None,
    ) -> None:
        if not binary_info()["installed"]:
            if status_callback is not None:
                status_callback("installing")
            ensure_binary()

    async def __aexit__(self, exc_type, exc, tb):
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            self._notify_cancellation(exc)
        try:
            await self._close_resources()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"Playwright cleanup also failed: {cleanup_error!r}")
        return False

    def _notify_cancellation(self, cancellation: BaseException) -> None:
        if self.cancellation_callback is None:
            return
        try:
            self.cancellation_callback()
        except BaseException as callback_error:
            cancellation.add_note(
                f"Playwright cancellation callback failed: {callback_error!r}"
            )

    async def _close_resources(self) -> None:
        """Close partially or fully initialized browser resources once."""
        context = self.context
        browser = self.browser
        self.context = None
        self.api = None
        self._fn_mapping = {}
        self.browser = None

        context_error: BaseException | None = None
        if context is not None:
            try:
                if not context.is_closed():
                    await context.close()
            except BaseException as error:
                if isinstance(
                    error,
                    (asyncio.CancelledError, KeyboardInterrupt),
                ):
                    self._notify_cancellation(error)
                context_error = error

        browser_error: BaseException | None = None
        if browser is not None:
            try:
                await browser.close()
            except BaseException as error:
                if isinstance(
                    error,
                    (asyncio.CancelledError, KeyboardInterrupt),
                ):
                    self._notify_cancellation(error)
                browser_error = error

        if context_error is not None:
            if browser_error is not None:
                context_error.add_note(
                    f"Browser cleanup also failed: {browser_error!r}"
                )
            raise context_error
        if browser_error is not None:
            raise browser_error

    def get_request_fn(self, method: str):
        if method not in self._fn_mapping:
            raise RuntimeError(f"Unsupported request_method: {method}")
        return self._fn_mapping[method]
    
    async def fetch_with_page(
            self, 
            url: str, 
            headers: dict | None = None,
            timeout: float = 60000,
            wait_until: Literal['commit', 'domcontentloaded', 'load', 'networkidle'] | None = 'load'
            ) -> Response | None: 
        
        page: Page | None = None
        resp: Response | None = None
        try:
            async with self.sem:
                page = await self.context.new_page()

                if headers:
                    await page.set_extra_http_headers(headers)

                start = perf_counter()

                resp = await page.goto(
                    url, 
                    wait_until=wait_until, 
                    timeout=timeout
                )

                if resp is None:
                    return None

                resp.elapsed = perf_counter() - start

                try:
                    resp.text = await page.content()
                except Exception:
                    resp.text = ""

                return resp

        finally:
            if page and not page.is_closed() and not asyncio.current_task().cancelling():
                await page.close()

        return resp
    
    async def fetch_with_api(
            self,
            request_fn, 
            url: str, 
            headers: dict | None = None, 
            timeout: float = 60000,
            request_payload: Any | bytes | str | None  = None
            ) -> APIResponse | None:
        
        resp: Response | None = None
        async with self.sem:
            start = perf_counter()
            resp: APIResponse | None = await request_fn(url=url, timeout=timeout, headers=headers, data=request_payload)
            if resp is not None:
                resp.elapsed = (perf_counter() - start)
                try:
                    resp.text = await resp.text()
                except Exception:
                    resp.text = ''

        return resp
