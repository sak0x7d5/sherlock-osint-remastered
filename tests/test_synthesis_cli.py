import asyncio
import sys
from types import SimpleNamespace
from typing import Self

import pytest

from sherlock_project import sherlock as sherlock_module
from sherlock_project.ai_config import AIConfigError, AISettings
from sherlock_project.investigation_context import parse_inline_anchor
from sherlock_project.notify import QueryNotify
from sherlock_project.profile_synthesis import ProfileSynthesis

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def configured_ai(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sherlock_module,
        "load_ai_settings",
        lambda: AISettings(
            base_url="http://localhost:8080",
            model="example/model",
        ),
    )


@pytest.mark.parametrize(
    "pass_two_args",
    [
        ["--anchor", "roles=Researcher"],
        ["--force-ai-synthesis"],
    ],
)
async def test_targeted_ai_mode_rejects_pass_two_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pass_two_args: list[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "--ai",
            "--site",
            "Example",
            *pass_two_args,
            "blue",
        ],
    )

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    error = capsys.readouterr().err
    assert "--site --ai runs pass one only" in error
    assert "--ai-synthesize-only" in error


async def test_removed_known_facts_option_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "--ai",
            "--known-facts",
            "facts.json",
            "blue",
        ],
    )

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    assert "unrecognized arguments: --known-facts" in capsys.readouterr().err


async def test_setup_ai_dispatches_before_scan_parser(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[str] = []

    async def fake_setup(argv):
        captured.extend(argv)
        return 0

    monkeypatch.setattr(sys, "argv", ["sherlock", "setup", "ai", "--no-color"])
    monkeypatch.setattr(sherlock_module, "run_ai_setup", fake_setup)

    exit_code = await sherlock_module.main()

    assert captured == ["--no-color"]
    assert exit_code == 0


@pytest.mark.parametrize(
    "scan_options",
    [
        ["--site", "Plurk"],
        ["--local"],
        ["--fresh"],
        ["--site", "Plurk", "--local"],
    ],
)
async def test_synthesis_only_rejects_scan_only_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    scan_options: list[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "fixture_handle",
            "--ai-synthesize-only",
            "--anchor",
            "name=Avery",
            *scan_options,
        ],
    )

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    error = capsys.readouterr().err
    assert "cannot be used with --ai-synthesize-only" in error
    assert "uses all saved extractions" in error
    assert "--ai --site" in error


@pytest.mark.parametrize(
    ("extra_args", "expected_sites"),
    [
        ([], ["Unseen"]),
        (["--fresh"], ["Saved", "Unseen"]),
    ],
)
async def test_fresh_disables_the_saved_site_resume_filter(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_sites: list[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [
                SimpleNamespace(name="Saved", information={}),
                SimpleNamespace(name="Unseen", information={}),
            ]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {
                "Saved": {
                    "site_name": "Saved",
                    "site_url": "https://saved.example/blue",
                    "status": "Claimed",
                    "status_code": 200,
                    "query_time_ms": 1.0,
                    "error_context": None,
                    "confidence": "Confirmed",
                }
            }

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()
    scanned_sites: list[str] = []

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_scan(**kwargs: object) -> dict:
        scanned_sites.extend(kwargs["site_data"])
        return {}

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", *extra_args, "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)

    await sherlock_module.main()

    assert sorted(scanned_sites) == expected_sites
    assert database.closed is True


async def test_concurrency_option_reaches_the_fetch_engine(
    monkeypatch: pytest.MonkeyPatch,
):
    """The flag is the easy half; the plumbing is what this guards.

    PlaywrightEngine has accepted a concurrency argument all along -- main()
    simply never passed one, so every run was pinned to the built-in default no
    matter what the user asked for.
    """

    engine_kwargs: dict = {}

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **kwargs: object) -> None:
            engine_kwargs.update(kwargs)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_scan(**_kwargs: object) -> dict:
        return {}

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", "-c", "7", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)

    await sherlock_module.main()

    assert engine_kwargs["concurrency"] == 7


async def test_browser_run_rescans_sites_the_fast_transport_did_not_confirm(
    monkeypatch: pytest.MonkeyPatch,
):
    """The contamination guard, end to end.

    Measured live on 2026-08-12: the fast transport reported Instagram as NOT
    FOUND for a username the browser found in the same minute. Without this
    filter that false absence is stored, satisfies the resume rule forever, and
    a later browser scan never revisits it -- the user would need --fresh, with
    nothing on screen suggesting why.
    """

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [
                SimpleNamespace(name="FastHit", information={}),
                SimpleNamespace(name="FastMiss", information={}),
                SimpleNamespace(name="BrowserMiss", information={}),
            ]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    def _row(name: str, status: str, transport: str) -> dict:
        return {
            "site_name": name,
            "site_url": f"https://{name.lower()}.example/blue",
            "status": status,
            "status_code": 200,
            "query_time_ms": 1.0,
            "error_context": None,
            "confidence": None,
            "transport": transport,
        }

    class FakeDB:
        closed = False

        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {
                "FastHit": _row("FastHit", "Claimed", "http"),
                "FastMiss": _row("FastMiss", "Available", "http"),
                "BrowserMiss": _row("BrowserMiss", "Available", "browser"),
            }

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()
    scanned_sites: list[str] = []

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_scan(**kwargs: object) -> dict:
        scanned_sites.extend(kwargs["site_data"])
        return {}

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)

    await sherlock_module.main()

    # The unconfirmed fast answer, and only that one: a fast HIT really did
    # find its marker, and a browser answer is already the best available.
    assert scanned_sites == ["FastMiss"]


@pytest.mark.parametrize(
    ("argv", "expected_engine"),
    [
        (["sherlock", "--local", "blue"], "browser"),
        (["sherlock", "--local", "--no-webbrowser", "blue"], "http"),
    ],
)
async def test_transport_choice_decides_which_engine_is_built(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_engine: str,
):
    """Not starting Chromium at all is most of what the fast mode buys.

    Constructing the browser engine and then not fetching with it would keep
    the startup cost the mode exists to avoid, and the saving would be
    invisible to anyone measuring it.
    """
    built: list[str] = []

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        closed = False

        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def close(self) -> None:
            self.closed = True

    def _engine(name: str):
        class FakeEngine:
            fixed_transport = None if name == "browser" else "http"

            def __init__(self, **_kwargs: object) -> None:
                built.append(name)

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_args: object) -> None:
                pass

        return FakeEngine

    database = FakeDB()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_scan(**_kwargs: object) -> dict:
        return {}

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", _engine("browser"))
    monkeypatch.setattr(sherlock_module, "HttpEngine", _engine("http"))
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)

    await sherlock_module.main()

    assert built == [expected_engine]


@pytest.mark.parametrize("value", ["0", "-1"])
async def test_concurrency_option_rejects_values_below_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
):
    """Zero is the one that has to fail loudly.

    asyncio.Semaphore(0) does not raise -- it blocks every acquire forever, so
    without this check the scan would hang silently instead of erroring.
    """
    monkeypatch.setattr(sys, "argv", ["sherlock", "--concurrency", value, "blue"])

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    assert "Concurrency must be at least 1" in capsys.readouterr().err


async def test_fully_cached_username_reports_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Every site stored: report from the database, do not start a scan.

    Running the scan anyway would print "checking 0 sites" and "0 found"
    directly beneath the stored results, which is what made a repeat run look
    like it had failed.
    """

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Saved", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {
                "Saved": {
                    "site_name": "Saved",
                    "site_url": "https://saved.example/blue",
                    "status": "Claimed",
                    "status_code": 200,
                    "query_time_ms": 1.0,
                    "error_context": None,
                    "confidence": "Confirmed",
                }
            }

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_scan(**_kwargs: object) -> dict:
        raise AssertionError("no scan may run when every site is already stored")

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", "--no-color", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)

    await sherlock_module.main()

    out = capsys.readouterr().out
    # The stored hit is reported even though nothing was scanned.
    assert "https://saved.example/blue" in out
    assert "--fresh" in out
    assert database.closed is True


async def test_ai_mode_without_configuration_fails_before_scanning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sherlock_module,
        "load_ai_settings",
        lambda: (_ for _ in ()).throw(
            AIConfigError("AI is not configured. Run `sherlock setup ai` first.")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["sherlock", "--ai", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network scan must not start")
        ),
    )

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    assert "sherlock setup ai" in capsys.readouterr().err


async def test_targeted_ai_mode_processes_only_fresh_selected_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [
                SimpleNamespace(name="Example", information={}),
                SimpleNamespace(name="Other", information={}),
            ]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        def __init__(self) -> None:
            self.pending_calls = 0
            self.closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            raise AssertionError("targeted sites must always be fetched again")

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            self.pending_calls += 1
            return [999]

        async def close(self) -> None:
            self.closed = True

    class FakeAIService:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()
    service = FakeAIService()
    processed_ids: list[int] = []
    scan_kwargs: dict[str, object] = {}

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_ai_create(**_kwargs: object) -> FakeAIService:
        return service

    async def fake_worker(ai_queue: asyncio.Queue[int], **_kwargs: object):
        while True:
            try:
                site_id = await ai_queue.get()
            except asyncio.QueueShutDown:
                return
            processed_ids.append(site_id)
            ai_queue.task_done()

    async def fake_scan(**kwargs: object) -> dict:
        scan_kwargs.update(kwargs)
        enqueue = kwargs["enqueue_ai"]
        assert callable(enqueue)
        await enqueue(41)
        return {}

    async def unexpected_synthesis(**_kwargs: object) -> None:
        raise AssertionError("pass two must not run in targeted AI mode")

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--local", "--ai", "--site", "Example", "blue"],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", fake_ai_create)
    monkeypatch.setattr(sherlock_module, "ai_worker", fake_worker)
    # Starting llama-server is not what this test is about, and letting it run
    # would spawn a real process -- or fail on a machine with no models. Its
    # own behaviour is covered in test_llama_server.py.
    monkeypatch.setattr(sherlock_module, "ManagedLlamaServer", FakeManagedServer)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        unexpected_synthesis,
    )

    await sherlock_module.main()

    assert set(scan_kwargs["site_data"]) == {"Example"}
    assert scan_kwargs["force_ai_extraction"] is True
    assert processed_ids == [41]
    assert database.pending_calls == 0
    assert database.closed is True
    assert service.closed is True
    output = capsys.readouterr().out
    assert "Targeted AI mode" in output
    assert "profile synthesis skipped" in output
    assert "AI pass-one concurrency" not in output


class FakeManagedServer:
    """No-op stand-in for the llama-server launcher."""

    def __init__(self, _settings, **_kwargs) -> None:
        pass

    async def ensure_running(self):
        return SimpleNamespace(running=True, started_by_us=False, detail="stub")

    async def stop(self) -> None:
        return None


async def test_normal_ai_scan_overlaps_model_loading_and_waits_before_synthesis(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []

    class RecordingReporter(sherlock_module.QueryNotifyPrint):
        def ai_pass_started(self) -> None:
            events.append("pass_started")
            super().ai_pass_started()

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.closed = True

    class FakeAIService:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            events.append("browser_started")
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    release_model = asyncio.Event()
    database = FakeDB()
    service = FakeAIService()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_ai_create(**_kwargs: object) -> FakeAIService:
        events.append("model_load_started")
        await release_model.wait()
        events.append("model_ready")
        return service

    async def fake_worker(ai_queue: asyncio.Queue[int], **kwargs: object):
        reporter = kwargs["reporter"]
        while True:
            try:
                site_id = await ai_queue.get()
            except asyncio.QueueShutDown:
                return
            reporter.ai_job_started("Example")
            events.append(f"extracted-{site_id}")
            reporter.ai_job_finished("with_facts")
            ai_queue.task_done()

    async def fake_scan(**kwargs: object) -> dict:
        assert events == ["model_load_started", "browser_started"]
        enqueue = kwargs["enqueue_ai"]
        assert callable(enqueue)
        await enqueue(41)
        events.append("scan_enqueued")
        assert "pass_started" not in events
        release_model.set()
        return {}

    async def fake_synthesis(**_kwargs: object) -> None:
        events.append("synthesis")
        assert "extracted-41" in events

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", "--ai", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", fake_ai_create)
    monkeypatch.setattr(sherlock_module, "ai_worker", fake_worker)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(sherlock_module, "synthesize_profiles", fake_synthesis)
    monkeypatch.setattr(sherlock_module, "QueryNotifyPrint", RecordingReporter)
    # Starting llama-server is not what this test is about, and leaving it real
    # would spawn a process -- or fail on a machine with no models. Its own
    # behaviour lives in test_llama_server.py.
    monkeypatch.setattr(sherlock_module, "ManagedLlamaServer", FakeManagedServer)

    await sherlock_module.main()

    assert events.index("model_load_started") < events.index("browser_started")
    assert events.index("scan_enqueued") < events.index("model_ready")
    assert events.index("model_ready") < events.index("pass_started")
    assert events.index("extracted-41") < events.index("synthesis")
    assert database.closed is True
    assert service.closed is True


async def test_model_load_failure_finishes_scan_and_skips_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()
    scan_finished = False

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def failing_ai_create(**_kwargs: object) -> None:
        raise RuntimeError("private model startup failure")

    async def fake_scan(**kwargs: object) -> dict:
        nonlocal scan_finished
        enqueue = kwargs["enqueue_ai"]
        assert callable(enqueue)
        await enqueue(52)
        scan_finished = True
        return {}

    async def unexpected_synthesis(**_kwargs: object) -> None:
        raise AssertionError("synthesis must be skipped when model loading fails")

    monkeypatch.setattr(sys, "argv", ["sherlock", "--local", "--ai", "blue"])
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", failing_ai_create)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        unexpected_synthesis,
    )

    await sherlock_module.main()

    assert scan_finished is True
    assert database.closed is True
    output = capsys.readouterr().out
    assert output.count("Local AI model unavailable") == 1
    assert "private model startup failure" not in output
    assert "Profile extraction unavailable · 1 pending" in output


async def test_synthesis_only_bypasses_network_sites_and_browser(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def fake_run_synthesis_only(**kwargs):
        captured.update(kwargs)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("scan/network setup must not run in synthesis-only mode")

    monkeypatch.setattr(sys, "argv", ["sherlock", "--ai-synthesize-only", "blue"])
    monkeypatch.setattr(sherlock_module, "run_synthesis_only", fake_run_synthesis_only)
    monkeypatch.setattr(sherlock_module.requests, "get", unexpected_call)
    monkeypatch.setattr(sherlock_module, "SitesInformation", unexpected_call)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", unexpected_call)

    await sherlock_module.main()

    assert captured["usernames"] == ["blue"]
    assert captured["force"] is True


async def test_synthesis_only_expands_username_placeholders(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def fake_run_synthesis_only(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--ai-synthesize-only", "blue{?}team"],
    )
    monkeypatch.setattr(sherlock_module, "run_synthesis_only", fake_run_synthesis_only)

    await sherlock_module.main()

    assert captured["usernames"] == [
        "blue_team",
        "blue-team",
        "blue.team",
    ]


async def test_synthesis_only_applies_inline_anchors_to_every_username(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def fake_run_synthesis_only(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "--ai-synthesize-only",
            "blue{?}team",
            "other",
            "--anchor",
            "verified:full_name=Avery Stone",
            "--anchor",
            "roles=Ethical Hacker",
        ],
    )
    monkeypatch.setattr(sherlock_module, "run_synthesis_only", fake_run_synthesis_only)

    await sherlock_module.main()

    assert captured["usernames"] == [
        "blue_team",
        "blue-team",
        "blue.team",
        "other",
    ]
    assert [anchor.trust for anchor in captured["inline_anchors"]] == [
        "verified",
        "context",
    ]
    assert all(
        anchor.source == "command_line"
        for anchor in captured["inline_anchors"]
    )


async def test_inline_anchor_requires_ai_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "blue", "--anchor", "roles=Pentester"],
    )

    with pytest.raises(SystemExit):
        await sherlock_module.main()

    assert "--anchor requires --ai or --ai-synthesize-only" in capsys.readouterr().err


async def test_synthesis_only_deduplicates_exact_username_inputs(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def fake_run_synthesis_only(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--ai-synthesize-only", "blue", "blue"],
    )
    monkeypatch.setattr(sherlock_module, "run_synthesis_only", fake_run_synthesis_only)

    await sherlock_module.main()

    assert captured["usernames"] == ["blue"]


async def test_anchorless_synthesis_only_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        closed = False

        async def close(self):
            self.closed = True

    db = FakeDB()

    async def fake_db_create(_path):
        return db

    async def unexpected_model_create(*_args, **_kwargs):
        raise AssertionError("anchorless synthesis must not load LM Studio")

    async def fake_synthesize_profiles(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", unexpected_model_create)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        fake_synthesize_profiles,
    )

    await sherlock_module.run_synthesis_only(
        usernames=["blue"],
        force=False,
    )

    assert captured["ai_service"]._provider is None
    assert db.closed is True


async def test_synthesize_profiles_uses_anchor_context_and_continues(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []
    events: list[tuple[str, str, object | None]] = []

    class RecordingReporter:
        def synthesis_started(self, username: str) -> None:
            events.append(("started", username, None))

        def synthesis_failed(self, username: str, error: Exception) -> None:
            events.append(("failed", username, error))

        def synthesis_finished(
            self,
            username: str,
            profile: ProfileSynthesis,
            *,
            cache_hit: bool,
        ) -> None:
            events.append(("finished", username, cache_hit))

    async def fake_synthesize_username_profile(**kwargs):
        calls.append(kwargs)
        if kwargs["username"] == "broken":
            raise RuntimeError("model failed")
        return type(
            "Result",
            (),
            {
                "cache_hit": False,
                "profile": ProfileSynthesis(
                    username=kwargs["username"],
                    input_hash="hash",
                    mode="anchored",
                    resolution_status="resolved",
                    completeness="complete",
                ),
            },
        )()

    monkeypatch.setattr(
        sherlock_module,
        "synthesize_username_profile",
        fake_synthesize_username_profile,
    )

    await sherlock_module.synthesize_profiles(
        db=object(),  # type: ignore[arg-type]
        ai_service=object(),  # type: ignore[arg-type]
        usernames=["broken", "blue"],
        force=True,
        inline_anchors=[parse_inline_anchor("roles=Pentester")],
        reporter=RecordingReporter(),  # type: ignore[arg-type]
    )

    assert [call["username"] for call in calls] == ["broken", "blue"]
    assert all(
        call["context"].anchors[0].value == "Pentester"
        for call in calls
    )
    assert calls[1]["force"] is True
    assert [event[:2] for event in events] == [
        ("started", "broken"),
        ("failed", "broken"),
        ("started", "blue"),
        ("finished", "blue"),
    ]
    assert isinstance(events[1][2], RuntimeError)


async def test_main_scan_cancellation_returns_130_and_skips_exports(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def close(self) -> None:
            self.close_calls += 1

    class FakeEngine:
        fixed_transport = None
        exit_calls = 0
        exit_error: type[BaseException] | None = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            *_args: object,
        ) -> None:
            self.exit_calls += 1
            self.exit_error = error_type

    database = FakeDB()
    engine = FakeEngine()
    scan_started = asyncio.Event()
    scan_cancelled = asyncio.Event()
    export_path = tmp_path / "partial.txt"

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def blocked_scan(**_kwargs: object) -> dict:
        scan_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            scan_cancelled.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "--local",
            "--txt",
            "--output",
            str(export_path),
            "blue",
        ],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(
        sherlock_module,
        "PlaywrightEngine",
        lambda **_kwargs: engine,
    )
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "sherlock", blocked_scan)

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(scan_started.wait(), timeout=1)
    main_task.cancel()

    assert await main_task == 130
    await asyncio.wait_for(scan_cancelled.wait(), timeout=1)
    assert database.close_calls == 1
    assert engine.exit_calls == 1
    assert engine.exit_error is asyncio.CancelledError
    assert not export_path.exists()
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Scan complete" not in output
    assert "Processing complete" not in output


async def test_main_ai_generation_cancellation_closes_once_and_skips_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.close_calls += 1

    class FakeAIService:
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class FakeEngine:
        fixed_transport = None
        exit_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.exit_calls += 1

    database = FakeDB()
    service = FakeAIService()
    engine = FakeEngine()
    generation_started = asyncio.Event()
    generation_cancelled = asyncio.Event()
    synthesis_called = False
    export_path = tmp_path / "partial-ai.txt"

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_ai_create(**_kwargs: object) -> FakeAIService:
        return service

    async def blocked_worker(
        ai_queue: asyncio.Queue[int],
        **_kwargs: object,
    ) -> None:
        site_id = await ai_queue.get()
        assert site_id == 41
        generation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ai_queue.task_done()
            generation_cancelled.set()

    async def fake_scan(**kwargs: object) -> dict:
        enqueue = kwargs["enqueue_ai"]
        assert callable(enqueue)
        await enqueue(41)
        return {}

    async def unexpected_synthesis(**_kwargs: object) -> None:
        nonlocal synthesis_called
        synthesis_called = True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sherlock",
            "--local",
            "--ai",
            "--txt",
            "--output",
            str(export_path),
            "blue",
        ],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(
        sherlock_module,
        "PlaywrightEngine",
        lambda **_kwargs: engine,
    )
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", fake_ai_create)
    monkeypatch.setattr(sherlock_module, "ai_worker", blocked_worker)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        unexpected_synthesis,
    )

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(generation_started.wait(), timeout=1)
    main_task.cancel()

    assert await main_task == 130
    await asyncio.wait_for(generation_cancelled.wait(), timeout=1)
    assert synthesis_called is False
    assert service.close_calls == 1
    assert database.close_calls == 1
    assert engine.exit_calls == 1
    assert not export_path.exists()
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "[+] Profile extraction" not in output
    assert "Processing complete" not in output


async def test_synthesis_only_cancellation_returns_130_without_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    synthesis_started = asyncio.Event()
    synthesis_cancelled = asyncio.Event()
    resources_closed = False

    async def blocked_synthesis(**_kwargs: object) -> None:
        nonlocal resources_closed
        synthesis_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            resources_closed = True
            synthesis_cancelled.set()

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--ai-synthesize-only", "blue"],
    )
    monkeypatch.setattr(
        sherlock_module,
        "run_synthesis_only",
        blocked_synthesis,
    )

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(synthesis_started.wait(), timeout=1)
    main_task.cancel()

    assert await main_task == 130
    await asyncio.wait_for(synthesis_cancelled.wait(), timeout=1)
    assert resources_closed is True
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Processing complete" not in output


async def test_sherlock_cancellation_gathers_done_and_unfinished_site_tasks(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeResponse:
        elapsed = 0.01
        status = 200
        text = "profile"
        url = "https://example.test/blue"

    class FakeEngine:
        fixed_transport = None
        blocked_cancelled = asyncio.Event()

        def get_request_fn(self, _method: str) -> object:
            return object()

        async def fetch_with_api(self, *, url: str, **_kwargs: object):
            if "blocked" not in url:
                return FakeResponse()
            try:
                await asyncio.Event().wait()
            finally:
                self.blocked_cancelled.set()
            raise AssertionError("unreachable")

        # GET sites take the browser transport; only POST rules use the API.
        async def fetch_with_page(self, *, url: str, **kwargs: object):
            return await self.fetch_with_api(url=url, **kwargs)

    class BlockingDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        save_started = asyncio.Event()

        async def save_result(self, **_kwargs: object) -> int:
            self.save_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    engine = FakeEngine()
    database = BlockingDB()
    original_gather = asyncio.gather
    gathered_tasks: list[asyncio.Task[object]] = []
    cancel_callback_calls = 0

    async def recording_gather(*tasks, **kwargs):
        gathered_tasks.extend(tasks)
        return await original_gather(*tasks, **kwargs)

    def failing_cancel_callback() -> None:
        nonlocal cancel_callback_calls
        cancel_callback_calls += 1
        raise RuntimeError("callback failed")

    monkeypatch.setattr(sherlock_module.asyncio, "gather", recording_gather)
    scan_task = asyncio.create_task(
        sherlock_module.sherlock(
            username="blue",
            engine=engine,  # type: ignore[arg-type]
            db=database,  # type: ignore[arg-type]
            site_data={
                "Done": {
                    "urlMain": "https://example.test",
                    "url": "https://example.test/api/done/{}",
                    "urlProfile": "https://example.test/done/{}",
                    "detection": {
                        "exists": {"code": 200, "string": "profile"},
                        "missing": {"code": 404, "string": ""},
                    },
                },
                "Blocked": {
                    "urlMain": "https://example.test",
                    "url": "https://example.test/api/blocked/{}",
                    "urlProfile": "https://example.test/blocked/{}",
                    "detection": {
                        "exists": {"code": 200, "string": "profile"},
                        "missing": {"code": 404, "string": ""},
                    },
                },
            },
            query_notify=QueryNotify(),
            on_cancel=failing_cancel_callback,
        )
    )
    await asyncio.wait_for(database.save_started.wait(), timeout=1)
    scan_task.cancel()

    with pytest.raises(asyncio.CancelledError) as cancellation:
        await scan_task

    await asyncio.wait_for(engine.blocked_cancelled.wait(), timeout=1)
    assert cancel_callback_calls == 1
    assert cancellation.value.__notes__ == [
        "Cancellation callback failed: RuntimeError: callback failed"
    ]
    assert len(gathered_tasks) == 2
    assert all(task.done() for task in gathered_tasks)


async def test_main_scan_cancellation_stops_ai_before_site_cleanup_finishes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [
                SimpleNamespace(
                    name="Fast",
                    information={
                        "urlMain": "https://example.test",
                        "url": "https://example.test/api/fast/{}",
                        "urlProfile": "https://example.test/fast/{}",
                        "detection": {
                            "exists": {"code": 200, "string": "profile"},
                            "missing": {"code": 404, "string": ""},
                        },
                    },
                ),
                SimpleNamespace(
                    name="Slow",
                    information={
                        "urlMain": "https://example.test",
                        "url": "https://example.test/api/slow/{}",
                        "urlProfile": "https://example.test/slow/{}",
                        "detection": {
                            "exists": {"code": 200, "string": "profile"},
                            "missing": {"code": 404, "string": ""},
                        },
                    },
                ),
            ]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeResponse:
        elapsed = 0.01
        status = 200
        text = "profile"
        url = "https://example.test/fast/blue"

    class FakeEngine:
        fixed_transport = None
        exit_calls = 0

        def get_request_fn(self, _method: str) -> object:
            return object()

        async def fetch_with_api(self, *, url: str, **_kwargs: object):
            if "/fast/" in url:
                return FakeResponse()
            blocked_scan_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                scan_cleanup_started.set()
                await release_scan_cleanup.wait()
                raise

        # GET sites take the browser transport; only POST rules use the API.
        async def fetch_with_page(self, *, url: str, **kwargs: object):
            return await self.fetch_with_api(url=url, **kwargs)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.exit_calls += 1

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def save_result(self, **_kwargs: object) -> int:
            return 41

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.close_calls += 1

    class FakeAIService:
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    database = FakeDB()
    engine = FakeEngine()
    service = FakeAIService()
    blocked_scan_started = asyncio.Event()
    scan_cleanup_started = asyncio.Event()
    release_scan_cleanup = asyncio.Event()
    generation_started = asyncio.Event()
    generation_cancelled = asyncio.Event()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_ai_create(**_kwargs: object) -> FakeAIService:
        return service

    async def blocked_worker(
        ai_queue: asyncio.Queue[int],
        **_kwargs: object,
    ) -> None:
        site_id = await ai_queue.get()
        assert site_id == 41
        generation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            generation_cancelled.set()
            ai_queue.task_done()

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--local", "--ai", "blue"],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(
        sherlock_module,
        "PlaywrightEngine",
        lambda **_kwargs: engine,
    )
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", fake_ai_create)
    monkeypatch.setattr(sherlock_module, "ai_worker", blocked_worker)

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(blocked_scan_started.wait(), timeout=1)
    await asyncio.wait_for(generation_started.wait(), timeout=1)
    main_task.cancel()

    await asyncio.wait_for(scan_cleanup_started.wait(), timeout=1)
    await asyncio.wait_for(generation_cancelled.wait(), timeout=1)
    assert not main_task.done()
    assert engine.exit_calls == 0

    release_scan_cleanup.set()
    assert await main_task == 130
    assert engine.exit_calls == 1
    assert service.close_calls == 1
    assert database.close_calls == 1
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Processing complete" not in output


async def test_main_interruption_survives_ai_and_database_close_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeEngine:
        fixed_transport = None
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("database close failed")

    class FakeAIService:
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("AI close failed")

    database = FakeDB()
    service = FakeAIService()
    synthesis_started = asyncio.Event()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_pipeline(
        ai_queue: asyncio.Queue[int],
        **_kwargs: object,
    ) -> FakeAIService:
        with pytest.raises(asyncio.QueueShutDown):
            await ai_queue.get()
        return service

    async def fake_scan(**_kwargs: object) -> dict:
        return {}

    async def blocked_synthesis(**_kwargs: object) -> None:
        synthesis_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--local", "--ai", "blue"],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(
        sherlock_module,
        "PlaywrightEngine",
        lambda **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "run_ai_pipeline", fake_pipeline)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        blocked_synthesis,
    )

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(synthesis_started.wait(), timeout=1)
    main_task.cancel()

    assert await main_task == 130
    assert service.close_calls == 1
    assert database.close_calls == 1
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Processing complete" not in output


async def test_synthesis_only_interruption_survives_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("database close failed")

    class FakeAIService:
        close_calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def close(self) -> None:
            type(self).close_calls += 1
            raise RuntimeError("AI close failed")

    database = FakeDB()
    synthesis_started = asyncio.Event()

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def blocked_synthesis(**_kwargs: object) -> None:
        synthesis_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--ai-synthesize-only", "blue"],
    )
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "AIService", FakeAIService)
    monkeypatch.setattr(
        sherlock_module,
        "synthesize_profiles",
        blocked_synthesis,
    )

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(synthesis_started.wait(), timeout=1)
    main_task.cancel()

    assert await main_task == 130
    assert FakeAIService.close_calls == 1
    assert database.close_calls == 1
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Processing complete" not in output


async def test_main_saved_sites_cancellation_stops_ai_before_engine_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeEngine:
        fixed_transport = None

        def __init__(
            self,
            *,
            cancellation_callback=None,
            **_kwargs: object,
        ) -> None:
            self.cancellation_callback = cancellation_callback

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            assert self.cancellation_callback is not None
            self.cancellation_callback()
            self.cancellation_callback()
            engine_cleanup_started.set()
            await release_engine_cleanup.wait()

    class FakeDB:
        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {}

        close_calls = 0

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            saved_sites_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def close(self) -> None:
            self.close_calls += 1

    database = FakeDB()
    saved_sites_started = asyncio.Event()
    generation_started = asyncio.Event()
    generation_cancelled = asyncio.Event()
    engine_cleanup_started = asyncio.Event()
    release_engine_cleanup = asyncio.Event()
    cancel_signals = 0
    real_cancel = sherlock_module._cancel_ai_pipeline_now

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def blocked_pipeline(**_kwargs: object) -> None:
        generation_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            generation_cancelled.set()

    def recording_cancel(**kwargs: object) -> None:
        nonlocal cancel_signals
        cancel_signals += 1
        real_cancel(**kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        ["sherlock", "--local", "--ai", "blue"],
    )
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module, "run_ai_pipeline", blocked_pipeline)
    monkeypatch.setattr(
        sherlock_module,
        "_cancel_ai_pipeline_now",
        recording_cancel,
    )

    main_task = asyncio.create_task(sherlock_module.main())
    await asyncio.wait_for(generation_started.wait(), timeout=1)
    await asyncio.wait_for(saved_sites_started.wait(), timeout=1)
    main_task.cancel()

    await asyncio.wait_for(engine_cleanup_started.wait(), timeout=1)
    await asyncio.wait_for(generation_cancelled.wait(), timeout=1)
    assert not main_task.done()
    assert cancel_signals == 1

    release_engine_cleanup.set()
    assert await main_task == 130
    assert cancel_signals == 1
    assert database.close_calls == 1
    output = capsys.readouterr().out
    assert output.count("Processing interrupted") == 1
    assert "Processing complete" not in output


@pytest.mark.parametrize(
    ("argv", "expected_force", "expects_warning"),
    [
        (["sherlock", "--local", "--ai", "blue"], False, True),
        (["sherlock", "--local", "--ai", "--fresh", "blue"], True, False),
    ],
)
async def test_fresh_redoes_extraction_and_silences_the_model_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected_force: bool,
    expects_warning: bool,
):
    """--fresh is how a newly configured model reaches results you already have.

    Without it the scan re-fetched every page and then kept the stored
    extraction whenever the page came back identical, so switching model and
    re-running produced byte-identical AI output at full price. The warning is
    the other half: it says so when the extractions are being kept, and stays
    quiet under --fresh, which is about to redo them anyway.
    """

    class FakeSites:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.items = [SimpleNamespace(name="Example", information={})]

        def __iter__(self):
            return iter(self.items)

        def remove_nsfw_sites(self, **_kwargs: object) -> None:
            pass

    class FakeDB:
        closed = False

        async def get_extraction_model_counts(
            self,
            _username: str,
        ) -> dict[str | None, int]:
            return {"other/model": 7}

        async def get_saved_results(self, **_kwargs: object) -> dict[str, dict]:
            return {}

        async def get_pending_ai_extraction_ids(
            self,
            _username: str,
            *,
            contract_hash: str,
        ) -> list[int]:
            return []

        async def close(self) -> None:
            self.closed = True

    class FakeAIService:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeEngine:
        fixed_transport = None

        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

    database = FakeDB()
    scan_kwargs: dict[str, object] = {}

    async def fake_db_create(_path: str) -> FakeDB:
        return database

    async def fake_ai_create(**_kwargs: object) -> FakeAIService:
        return FakeAIService()

    async def fake_worker(ai_queue: asyncio.Queue[int], **_kwargs: object):
        while True:
            try:
                await ai_queue.get()
            except asyncio.QueueShutDown:
                return
            ai_queue.task_done()

    async def fake_scan(**kwargs: object) -> dict:
        scan_kwargs.update(kwargs)
        return {}

    async def fake_synthesis(**_kwargs: object) -> None:
        return None

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        sherlock_module.requests,
        "get",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=f'{{"tag_name": "v{sherlock_module.__version__}"}}'
        ),
    )
    monkeypatch.setattr(sherlock_module, "SitesInformation", FakeSites)
    monkeypatch.setattr(sherlock_module, "PlaywrightEngine", FakeEngine)
    monkeypatch.setattr(sherlock_module.SherlockDB, "create", fake_db_create)
    monkeypatch.setattr(sherlock_module.AIService, "create", fake_ai_create)
    monkeypatch.setattr(sherlock_module, "ai_worker", fake_worker)
    monkeypatch.setattr(sherlock_module, "sherlock", fake_scan)
    monkeypatch.setattr(sherlock_module, "synthesize_profiles", fake_synthesis)

    await sherlock_module.main()
    output = capsys.readouterr().out

    assert scan_kwargs["force_ai_extraction"] is expected_force
    assert ("did not come from example/model" in output) is expects_warning
