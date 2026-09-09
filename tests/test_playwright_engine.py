import asyncio
import os
import subprocess
import sys

import pytest

import sherlock_project.playwright_engine as playwright_module
from sherlock_project.playwright_engine import BrowserUnavailable, PlaywrightEngine


@pytest.mark.parametrize("invalid_method", ['', 'unkonwn_method'])
def test_get_request_fn_raise_error(invalid_method: str, playwright_engine: PlaywrightEngine):
    with pytest.raises(RuntimeError):
        playwright_engine.get_request_fn(invalid_method)

@pytest.mark.parametrize("valid_method", ['GET', 'POST', 'PUT', 'HEAD'])
def test_get_request_fn_returns_callable(valid_method: str, playwright_engine: PlaywrightEngine):
    method_fn = playwright_engine.get_request_fn(valid_method)
    assert callable(method_fn)


@pytest.mark.asyncio
async def test_semaphore_size_follows_requested_concurrency() -> None:
    """The number the CLI asks for must be the number the fetch layer enforces."""
    engine = PlaywrightEngine(concurrency=2)

    await engine.sem.acquire()
    assert not engine.sem.locked()
    await engine.sem.acquire()
    assert engine.sem.locked()


def test_missing_browser_binary_reports_installation_without_printing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    statuses: list[str] = []
    installs: list[bool] = []
    monkeypatch.setattr(
        playwright_module,
        "binary_info",
        lambda: {"installed": False},
    )
    monkeypatch.setattr(
        playwright_module,
        "ensure_binary",
        lambda: installs.append(True),
    )

    PlaywrightEngine.ensure_browser_binary(statuses.append)  # type: ignore[arg-type]

    assert statuses == ["installing"]
    assert installs == [True]
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_browser_lifecycle_reports_starting_and_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses: list[str] = []
    context_close_calls: list[bool] = []
    browser_close_calls: list[bool] = []

    class FakeAPI:
        async def get(self, **_kwargs: object) -> None:
            pass

        head = get
        post = get
        put = get

    class FakeContext:
        request = FakeAPI()

        def is_closed(self) -> bool:
            return False

        async def close(self) -> None:
            context_close_calls.append(True)

    class FakeBrowser:
        async def new_context(self, **_kwargs: object) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            browser_close_calls.append(True)

    async def fake_launch(**_kwargs: object) -> FakeBrowser:
        return FakeBrowser()

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(lambda _callback=None: None),
    )
    monkeypatch.setattr(playwright_module, "launch_async", fake_launch)

    engine = PlaywrightEngine(
        status_callback=statuses.append,  # type: ignore[arg-type]
    )
    async with engine:
        pass
    await engine.__aexit__(None, None, None)

    assert statuses == ["starting", "ready"]
    assert context_close_calls == [True]
    assert browser_close_calls == [True]


@pytest.mark.asyncio
async def test_startup_failure_becomes_browser_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed binary download is reported as a browser problem, not an httpx one.

    `ensure_binary` reaches a third-party host, so the exception escaping it
    is whatever its HTTP client raised. Converting it here is what lets the
    entrypoint recognise the condition and point at --no-webbrowser instead of
    printing a transport library's stack.
    """
    def failed_install(_callback=None) -> None:
        raise OSError("certificate verify failed")

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(failed_install),
    )

    engine = PlaywrightEngine()
    with pytest.raises(BrowserUnavailable, match="certificate verify failed"):
        await engine.__aenter__()


@pytest.mark.asyncio
async def test_startup_cancellation_is_not_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during startup must still unwind as cancellation.

    The conversion catches Exception, and CancelledError derives from
    BaseException precisely so this stays true -- if it were ever widened,
    interrupting a first run would report an unavailable browser instead of
    an interrupted one.
    """
    def cancelled_install(_callback=None) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(cancelled_install),
    )

    engine = PlaywrightEngine()
    with pytest.raises(asyncio.CancelledError):
        await engine.__aenter__()


@pytest.mark.asyncio
async def test_browser_launch_cancellation_notifies_without_started_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_started = asyncio.Event()
    events: list[str] = []

    async def blocked_launch(**_kwargs: object) -> None:
        launch_started.set()
        await asyncio.Future()

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(lambda _callback=None: None),
    )
    monkeypatch.setattr(playwright_module, "launch_async", blocked_launch)

    engine = PlaywrightEngine(
        cancellation_callback=lambda: events.append(
            "cancellation_callback"
        )
    )
    startup = asyncio.create_task(engine.__aenter__())
    await launch_started.wait()
    startup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await startup

    assert events == ["cancellation_callback"]
    assert engine.browser is None
    assert engine.context is None


@pytest.mark.asyncio
async def test_context_creation_cancellation_closes_started_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_creation_started = asyncio.Event()
    events: list[str] = []

    class FakeBrowser:
        async def new_context(self, **_kwargs: object) -> None:
            context_creation_started.set()
            await asyncio.Future()

        async def close(self) -> None:
            events.append("browser_close")

    async def fake_launch(**_kwargs: object) -> FakeBrowser:
        return FakeBrowser()

    def cancellation_callback() -> None:
        events.append("cancellation_callback")
        raise RuntimeError("callback failed")

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(lambda _callback=None: None),
    )
    monkeypatch.setattr(playwright_module, "launch_async", fake_launch)

    engine = PlaywrightEngine(cancellation_callback=cancellation_callback)
    startup = asyncio.create_task(engine.__aenter__())
    await context_creation_started.wait()
    startup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await startup

    assert events == ["cancellation_callback", "browser_close"]
    assert engine.browser is None
    assert engine.context is None


@pytest.mark.asyncio
async def test_body_cancellation_notifies_before_context_and_browser_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeAPI:
        async def get(self, **_kwargs: object) -> None:
            pass

        head = get
        post = get
        put = get

    class FakeContext:
        request = FakeAPI()

        def is_closed(self) -> bool:
            return False

        async def close(self) -> None:
            events.append("context_close")

    class FakeBrowser:
        async def new_context(self, **_kwargs: object) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            events.append("browser_close")

    async def fake_launch(**_kwargs: object) -> FakeBrowser:
        return FakeBrowser()

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(lambda _callback=None: None),
    )
    monkeypatch.setattr(playwright_module, "launch_async", fake_launch)

    engine = PlaywrightEngine(
        cancellation_callback=lambda: events.append(
            "cancellation_callback"
        )
    )

    with pytest.raises(asyncio.CancelledError):
        async with engine:
            raise asyncio.CancelledError

    assert events == [
        "cancellation_callback",
        "context_close",
        "browser_close",
    ]


@pytest.mark.asyncio
async def test_context_close_failure_still_closes_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_close_calls: list[bool] = []

    class FakeAPI:
        async def get(self, **_kwargs: object) -> None:
            pass

        head = get
        post = get
        put = get

    class FakeContext:
        request = FakeAPI()

        def is_closed(self) -> bool:
            return False

        async def close(self) -> None:
            raise RuntimeError("context close failed")

    class FakeBrowser:
        async def new_context(self, **_kwargs: object) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            browser_close_calls.append(True)

    async def fake_launch(**_kwargs: object) -> FakeBrowser:
        return FakeBrowser()

    monkeypatch.setattr(
        PlaywrightEngine,
        "ensure_browser_binary",
        staticmethod(lambda _callback=None: None),
    )
    monkeypatch.setattr(playwright_module, "launch_async", fake_launch)

    engine = PlaywrightEngine()
    await engine.__aenter__()

    with pytest.raises(RuntimeError, match="context close failed"):
        await engine.__aexit__(None, None, None)

    assert browser_close_calls == [True]


@pytest.mark.parametrize("configured", [None, "true"])
def test_cloakbrowser_update_setting_defaults_off_but_preserves_override(
    configured: str | None,
) -> None:
    environment = os.environ.copy()
    if configured is None:
        environment.pop("CLOAKBROWSER_AUTO_UPDATE", None)
        expected = "false"
    else:
        environment["CLOAKBROWSER_AUTO_UPDATE"] = configured
        expected = configured

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import sherlock_project.playwright_engine; "
                "print(os.environ['CLOAKBROWSER_AUTO_UPDATE'])"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip() == expected
