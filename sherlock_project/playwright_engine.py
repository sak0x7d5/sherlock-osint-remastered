from playwright.async_api import Playwright, Page, Browser, BrowserContext, async_playwright
from playwright.async_api import APIRequest, APIRequestContext
from playwright_stealth.stealth import Stealth
from typing import Callable, Literal, Any, Awaitable, Protocol
import asyncio

class RequestMethod(Protocol):
    async def __call__(self, url: str, **kwargs: Any) -> Any: ...

class PlaywrightEngine:
    def __init__(self, concurrency: int = 20, headless: bool = True, stealth: bool = True):
        self.sem = asyncio.Semaphore(concurrency)
        self.playwright = None
        self.browser = None
        self.context = None
        self.headless = headless
        self.stealth = stealth

    async def __aenter__(self):
        self.playwright: Playwright = await async_playwright().start()
        self.browser: Browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context: BrowserContext = await self.browser.new_context(ignore_https_errors=True)

        
        print("Playwright Started!")
        
        # Apply stealth
        if self.stealth:
            await Stealth().apply_stealth_async(self.context)
            print("Stealth Applied!")

        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()

        print("Playwright Stopped!")

    async def fetch_site(self, url: str, request_fn: None | RequestMethod = None, timeout: float = 60000,**kwargs: Any):
        async with self.sem:
            resp = None
            # request_function = None -> use page.goto
            if not request_fn:
                page: Page = await self.context.new_page()
                resp = None
                try:
                    resp = await page.goto(url, wait_until='networkidle', timeout=timeout,**kwargs)
                except Exception as e:
                    print(f'Error occurred: {e}')
                finally:
                    await page.close()
            else:
                resp = await request_fn(url, timeout=timeout, **kwargs)
            return resp
        
if __name__ == '__main__':
    async def main():
        # urls = [(i, 'https://example.com/') for i in range(50)]
        urls = ['https://example.com/'] * 5
        async with PlaywrightEngine(headless=True) as engine:
            tasks = {asyncio.create_task(engine.fetch_site(url)): i for i, url in enumerate(urls)}
            # use async for, and map the site_url with the task completion
            # DONE! with python 3.13
            async for task in asyncio.as_completed(tasks):
                id = tasks[task]
                result = await task
                print(id, result)
                
    asyncio.run(main())