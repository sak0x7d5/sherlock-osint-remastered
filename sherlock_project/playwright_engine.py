from playwright.async_api import Page, Browser, BrowserContext, Route
from typing import Any, Protocol, Literal
import asyncio
from time import perf_counter
from playwright.async_api import APIResponse, APIRequestContext, Response
from cloakbrowser import launch_async, ensure_binary

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
        ensure_binary()
        self.browser: Browser = await launch_async(headless=self.headless, humanize=True)
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

    async def __aexit__(self, exc_type, exc, tb):
        await self.context.close()
        await self.browser.close()
        print("Playwright Stopped!")

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
        
        async with self.sem:
            page: Page = await self.context.new_page()
            try:
                if headers:
                    await page.set_extra_http_headers(headers)

                start = perf_counter()

                response = await page.goto(
                    url, 
                    wait_until=wait_until, 
                    timeout=timeout
                )

                if response is None:
                    return None

                response.elapsed = perf_counter() - start

                try:
                    response.text = await page.content()
                except Exception as e:
                    response.text = ""
                    print(f"page.content() failed: {e}")

            finally:
                await page.close()
            
            return response
    
    async def fetch_with_api(
            self,
            request_fn, 
            url: str, 
            headers: dict | None = None, 
            timeout: float = 60000,
            request_payload: Any | bytes | str | None  = None
            ) -> APIResponse | None:
        
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