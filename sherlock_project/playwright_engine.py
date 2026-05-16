from playwright.async_api import Page, Browser, BrowserContext, Route
from typing import Any, Protocol, Literal
import asyncio
from time import perf_counter
from dataclasses import dataclass
from playwright.async_api import APIResponse, APIRequestContext
from cloakbrowser import launch_async

class RequestMethod(Protocol):
    async def __call__(self, url: str, **kwargs: Any) -> Any: ...

class PlaywrightEngine:
    def __init__(self, concurrency: int = 30, headless: bool = True, stealth: bool = True, proxy: dict = None):
        self.sem = asyncio.Semaphore(concurrency)
        self.playwright = None
        self.browser = None
        self.context = None
        self.headless = headless
        self.stealth = stealth
        self.proxy = proxy

    async def __aenter__(self):
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
    
    @staticmethod
    async def handle_route(route: Route):
        try:
            if not route.request.is_navigation_request():
                return await route.continue_()
            
            # Fetch without following redirect
            response = await route.fetch(max_redirects=0)
            headers = response.headers

            # Check for Location header instead of status
            location = headers.pop("location", None) or headers.pop("Location", None) 
            
            if location:
                await route.fulfill(
                    status=response.status,
                    headers=headers,
                    body=""
                )
                return
            
            # Not a redirect -> continue normally
            await route.fulfill(response=response)
        except Exception as e:
            pass
            try:
                await route.continue_()
            except Exception:
                pass

    async def _fetch_with_page(self, url, headers, timeout, max_redirects, wait_until : Literal['commit', 'domcontentloaded', 'load', 'networkidle'] | None = 'load'):
        page: Page = await self.context.new_page()
        try:
            if headers:
                await page.set_extra_http_headers(headers)

            start = perf_counter()

            if max_redirects == 0:
                wait_until = 'commit'
                await page.route("**/*", self.handle_route)

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
    
    async def _fetch_with_api(self, request_fn, url, headers, timeout, max_redirects, request_payload) -> APIResponse | None:
        start = perf_counter()
        resp: APIResponse | None = await request_fn(url, timeout=timeout, headers=headers, max_redirects=max_redirects, data=request_payload)

        if resp is not None:
            resp.elapsed = (perf_counter() - start)
            try:
                resp.text = await resp.text()
            except Exception:
                resp.text = ''
        return resp
    
    async def fetch_site(self, url: str, request_fn: None | RequestMethod = None, **kwargs: Any):
        async with self.sem:
            # TODO:make sherlock handle this
            # seperate page and requests kwargs there
            headers: dict = kwargs.get('headers')
            timeout: float = kwargs.get('timeout', 60) * 1000
            request_payload = kwargs.get('request_payload')
            max_redirects: int = kwargs.get('max_redirects', 20)

            # default to page.goto if not specified
            if request_fn is None:
                return await self._fetch_with_page(url, headers, timeout, max_redirects)
            return await self._fetch_with_api(request_fn, url, headers, timeout, max_redirects, request_payload)
           
