from playwright.async_api import Page, Browser, BrowserContext
from typing import Any, Protocol, Literal
import asyncio
from time import perf_counter
from playwright.async_api import APIResponse, APIRequestContext, Response
from playwright.async_api import Error as PlaywrightError
from cloakbrowser import launch_async, ensure_binary, binary_info

class RequestMethod(Protocol):
    async def __call__(self, url: str, **kwargs: Any) -> Any: ...

class PlaywrightEngine:
    def __init__(self, concurrency: int = 30, headless: bool = True, proxy: dict = None):
        self.sem = asyncio.Semaphore(concurrency)
        self.playwright = None
        self.browser = None
        self.context = None
        self.headless = headless
        self.proxy = proxy

    async def __aenter__(self):
        self.ensure_browser_binary()

        self.browser: Browser = await launch_async(headless=self.headless, humanize=True, handle_sigint=False)
        self.context: BrowserContext = await self.browser.new_context(ignore_https_errors=True, proxy=self.proxy)
        self.api: APIRequestContext = self.context.request
        print("Playwright Started!")

        # intialize mappings when context is not None
        self._fn_mapping = {
        'GET': self.api.get,
        'HEAD': self.api.head,
        'POST': self.api.post,
        'PUT': self.api.put,
        }

        return self
    

    @staticmethod
    def ensure_browser_binary():
        if not binary_info()["installed"]:
            print("[*] Chrome browser binary not found. Installing...")
            ensure_binary()
            print("\033[H\033[J", end="")

    async def __aexit__(self, exc_type, exc, tb):
        if not self.context.is_closed():
            await self.context.close()

        if self.browser:
            await self.browser.close()

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

        except asyncio.CancelledError:
            raise

        except PlaywrightError:
            raise

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
        try:
            async with self.sem:
                start = perf_counter()
                resp: APIResponse | None = await request_fn(url=url, timeout=timeout, headers=headers, data=request_payload)
                if resp is not None:
                    resp.elapsed = (perf_counter() - start)
                    try:
                        resp.text = await resp.text()
                    except Exception:
                        resp.text = ''

        except asyncio.CancelledError:
            raise

        except PlaywrightError:
            raise

        return resp