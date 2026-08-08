import asyncio
import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from sherlock_project.ai_engine import StructuredResponseError
from sherlock_project.database import SherlockDB
from sherlock_project.notify import QueryNotify, TerminalReporter
from sherlock_project.result import QueryStatus
from sherlock_project.sherlock import ai_worker, run_ai_pipeline, sherlock

pytestmark = pytest.mark.asyncio

PASS_ONE_CONTRACT_HASH = "pass-one-contract-v2"


def _reporter(*, verbose: bool = False) -> tuple[TerminalReporter, StringIO]:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        no_color=True,
        width=200,
    )
    return (
        TerminalReporter(
            verbose=verbose,
            no_color=True,
            console=console,
            error_console=console,
        ),
        output,
    )


class FakeAIService:
    pass_one_contract_hash = PASS_ONE_CONTRACT_HASH
    model_key = "fake/model"

    def __init__(self, fail_sites: set[str] | None = None) -> None:
        self.fail_sites = fail_sites or set()
        self.calls: list[dict[str, object]] = []
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def extract_profile(
        self,
        username: str,
        site_name: str,
        site_content: str,
        *,
        known_profile_keys: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "username": username,
                "site_name": site_name,
                "site_content": site_content,
                "known_profile_keys": list(known_profile_keys),
            }
        )
        if site_name in self.fail_sites:
            raise RuntimeError("model failed")

        return SimpleNamespace(
            extraction={
                "profile_label": [username],
                "source_site": [site_name],
            }
        )


class BlockingAIService(FakeAIService):
    def __init__(self, expected_concurrency: int) -> None:
        super().__init__()
        self.expected_concurrency = expected_concurrency
        self.active = 0
        self.max_active = 0
        self.completed = 0
        self.at_capacity = asyncio.Event()
        self.release = asyncio.Event()

    async def extract_profile(
        self,
        username: str,
        site_name: str,
        site_content: str,
        *,
        known_profile_keys: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "username": username,
                "site_name": site_name,
                "site_content": site_content,
                "known_profile_keys": list(known_profile_keys),
            }
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == self.expected_concurrency:
            self.at_capacity.set()

        try:
            await self.release.wait()
        finally:
            self.active -= 1

        self.completed += 1
        return SimpleNamespace(extraction={"source_site": [site_name]})


class OutcomeAIService(FakeAIService):
    def __init__(self, outcomes: list[object]) -> None:
        super().__init__()
        self.outcomes = outcomes

    async def extract_profile(
        self,
        username: str,
        site_name: str,
        site_content: str,
        *,
        known_profile_keys: list[str],
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "username": username,
                "site_name": site_name,
                "site_content": site_content,
                "known_profile_keys": list(known_profile_keys),
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


def _structured_error(
    *,
    stop_reason: str = "maxPredictedTokensReached",
    predicted_tokens: int = 1024,
    max_tokens: int = 1024,
) -> StructuredResponseError:
    return StructuredResponseError(
        "OSINT extraction",
        stop_reason=stop_reason,
        predicted_tokens=predicted_tokens,
        max_tokens=max_tokens,
        parsed_type="str",
        final_content_chars=31,
        validation_error="json_invalid",
    )


async def _get_ai_extraction(db: SherlockDB, site_id: int) -> str | None:
    assert db.db is not None
    async with db.db.execute(
        "SELECT ai_extraction FROM results WHERE id = ?",
        (site_id,),
    ) as cur:
        row = await cur.fetchone()

    assert row is not None
    return row["ai_extraction"]


async def _get_ai_contract_hash(
    db: SherlockDB,
    site_id: int,
) -> str | None:
    assert db.db is not None
    async with db.db.execute(
        """
        SELECT ai_extraction_contract_hash
        FROM results
        WHERE id = ?
        """,
        (site_id,),
    ) as cur:
        row = await cur.fetchone()

    assert row is not None
    return row["ai_extraction_contract_hash"]


async def _finish_worker(
    queue: asyncio.Queue[int],
    worker_task: asyncio.Task[None],
) -> None:
    queue.shutdown()
    await worker_task
    await asyncio.wait_for(queue.join(), timeout=1)


async def test_ai_worker_loads_joined_job_and_saves_valid_json(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **kwargs: f"cleaned: {content}",
    )
    site_id = await db.save_result(
        username="blue",
        site_name="instagram",
        status=str(QueryStatus.CLAIMED),
        response_text="raw profile",
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = FakeAIService()
    worker_task = asyncio.create_task(ai_worker(queue, db, service))

    await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert service.calls == [
        {
            "username": "blue",
            "site_name": "instagram",
            "site_content": "cleaned: raw profile",
            "known_profile_keys": [],
        }
    ]
    saved = await _get_ai_extraction(db, site_id)
    assert saved is not None
    assert json.loads(saved) == {
        "profile_label": ["blue"],
        "source_site": ["instagram"],
    }
    assert "reasoning" not in saved
    assert await _get_ai_contract_hash(db, site_id) == PASS_ONE_CONTRACT_HASH


async def test_ai_worker_feeds_all_committed_key_names_to_later_sites(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    site_ids = [
        await db.save_result(
            username="blue",
            site_name=site_name,
            status=str(QueryStatus.CLAIMED),
            response_text=f"profile {site_name}",
        )
        for site_name in ("speaker", "portfolio", "community")
    ]
    service = OutcomeAIService(
        [
            SimpleNamespace(
                extraction={
                    "conference_talks": ["Defending Small Networks"],
                    "full_name": ["Avery Chen"],
                }
            ),
            SimpleNamespace(
                extraction={
                    "conference_talks": ["Practical Threat Modeling"],
                    "bug_bounty_programs": ["Example Security"],
                }
            ),
            SimpleNamespace(extraction={}),
        ]
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    worker_task = asyncio.create_task(ai_worker(queue, db, service))

    for site_id in site_ids:
        await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert [call["known_profile_keys"] for call in service.calls] == [
        [],
        ["conference_talks", "full_name"],
        ["conference_talks", "full_name", "bug_bounty_programs"],
    ]
    assert all(
        "Avery Chen" not in call["known_profile_keys"]
        and "Defending Small Networks" not in call["known_profile_keys"]
        for call in service.calls
    )


async def test_ai_worker_keeps_key_feedback_isolated_by_username(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    jobs = [
        ("blue", "blue-one"),
        ("red", "red-one"),
        ("blue", "blue-two"),
    ]
    site_ids = [
        await db.save_result(
            username=username,
            site_name=site_name,
            status=str(QueryStatus.CLAIMED),
            response_text="profile",
        )
        for username, site_name in jobs
    ]
    service = OutcomeAIService(
        [
            SimpleNamespace(extraction={"conference_talks": ["LakeSec"]}),
            SimpleNamespace(extraction={"certifications": ["OSCP"]}),
            SimpleNamespace(extraction={}),
        ]
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    worker_task = asyncio.create_task(ai_worker(queue, db, service))

    for site_id in site_ids:
        await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert [call["known_profile_keys"] for call in service.calls] == [
        [],
        [],
        ["conference_talks"],
    ]


async def test_ai_worker_learns_keys_only_after_database_commit(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    first_id = await db.save_result(
        username="blue",
        site_name="speaker",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    second_id = await db.save_result(
        username="blue",
        site_name="portfolio",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    original_update = db.update_result_ai_extraction

    async def fail_first_commit(
        site_id: int,
        ai_extraction: str,
        *,
        contract_hash: str,
    ) -> None:
        if site_id == first_id:
            raise RuntimeError("database write failed")
        await original_update(
            site_id,
            ai_extraction,
            contract_hash=contract_hash,
        )

    monkeypatch.setattr(db, "update_result_ai_extraction", fail_first_commit)
    service = OutcomeAIService(
        [
            SimpleNamespace(extraction={"conference_talks": ["LakeSec"]}),
            SimpleNamespace(extraction={"full_name": ["Avery Chen"]}),
        ]
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    worker_task = asyncio.create_task(ai_worker(queue, db, service))
    await queue.put(first_id)
    await queue.put(second_id)
    await _finish_worker(queue, worker_task)

    assert [call["known_profile_keys"] for call in service.calls] == [[], []]
    assert await _get_ai_extraction(db, first_id) is None
    assert await _get_ai_extraction(db, second_id) is not None


async def test_ai_worker_resume_hydration_matches_uninterrupted_hints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )

    async def third_call_hints(path: Path, *, restart: bool) -> list[str]:
        run_db = await SherlockDB.create(str(path))
        try:
            site_ids = [
                await run_db.save_result(
                    username="blue",
                    site_name=f"site-{index}",
                    status=str(QueryStatus.CLAIMED),
                    response_text=f"profile-{index}",
                )
                for index in range(3)
            ]
            first_two = [
                SimpleNamespace(extraction={"full_name": ["Avery Chen"]}),
                SimpleNamespace(
                    extraction={"conference_talks": ["LakeSec"]}
                ),
            ]

            if restart:
                first_service = OutcomeAIService(first_two)
                first_queue: asyncio.Queue[int] = asyncio.Queue()
                first_worker = asyncio.create_task(
                    ai_worker(first_queue, run_db, first_service)
                )
                for site_id in site_ids[:2]:
                    await first_queue.put(site_id)
                await _finish_worker(first_queue, first_worker)

                final_service = OutcomeAIService(
                    [SimpleNamespace(extraction={})]
                )
                final_queue: asyncio.Queue[int] = asyncio.Queue()
                final_worker = asyncio.create_task(
                    ai_worker(final_queue, run_db, final_service)
                )
                await final_queue.put(site_ids[2])
                await _finish_worker(final_queue, final_worker)
                return final_service.calls[0]["known_profile_keys"]  # type: ignore[return-value]

            service = OutcomeAIService(
                [*first_two, SimpleNamespace(extraction={})]
            )
            queue: asyncio.Queue[int] = asyncio.Queue()
            worker = asyncio.create_task(ai_worker(queue, run_db, service))
            for site_id in site_ids:
                await queue.put(site_id)
            await _finish_worker(queue, worker)
            return service.calls[2]["known_profile_keys"]  # type: ignore[return-value]
        finally:
            await run_db.close()

    uninterrupted = await third_call_hints(
        tmp_path / "uninterrupted.db",
        restart=False,
    )
    resumed = await third_call_hints(
        tmp_path / "resumed.db",
        restart=True,
    )

    assert uninterrupted == resumed == ["full_name", "conference_talks"]


async def test_ai_worker_hydrates_only_valid_current_contract_keys(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    await db.save_result(
        username="blue",
        site_name="current",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction=json.dumps({"conference_talks": ["LakeSec"]}),
        ai_extraction_contract_hash=PASS_ONE_CONTRACT_HASH,
    )
    await db.save_result(
        username="blue",
        site_name="stale",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction=json.dumps({"stale_key": ["ignore"]}),
        ai_extraction_contract_hash="old-contract",
    )
    await db.save_result(
        username="blue",
        site_name="malformed-current",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction=json.dumps({"bad_value": "scalar"}),
        ai_extraction_contract_hash=PASS_ONE_CONTRACT_HASH,
    )
    await db.save_result(
        username="red",
        site_name="other-user",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
        ai_extraction=json.dumps({"certifications": ["OSCP"]}),
        ai_extraction_contract_hash=PASS_ONE_CONTRACT_HASH,
    )
    pending_id = await db.save_result(
        username="blue",
        site_name="pending",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )

    service = OutcomeAIService([SimpleNamespace(extraction={})])
    queue: asyncio.Queue[int] = asyncio.Queue()
    worker = asyncio.create_task(ai_worker(queue, db, service))
    await queue.put(pending_id)
    await _finish_worker(queue, worker)

    assert service.calls[0]["known_profile_keys"] == ["conference_talks"]


async def test_ai_worker_continues_after_job_failure(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **kwargs: content,
    )
    failed_id = await db.save_result(
        username="blue",
        site_name="broken",
        status=str(QueryStatus.CLAIMED),
        response_text="broken profile",
    )
    successful_id = await db.save_result(
        username="blue",
        site_name="github",
        status=str(QueryStatus.CLAIMED),
        response_text="working profile",
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = FakeAIService(fail_sites={"broken"})
    reporter, output = _reporter()
    worker_task = asyncio.create_task(
        ai_worker(queue, db, service, reporter=reporter)
    )

    await queue.put(failed_id)
    await queue.put(successful_id)
    await _finish_worker(queue, worker_task)

    assert await _get_ai_extraction(db, failed_id) is None
    assert await _get_ai_extraction(db, successful_id) is not None
    assert sum(
        call["site_name"] == "broken" for call in service.calls
    ) == 1
    rendered = output.getvalue()
    assert "broken: profile extraction failed; left pending" in rendered
    assert "model failed" not in rendered
    assert reporter.ai_stats.with_facts == 1
    assert reporter.ai_stats.pending == 1


async def test_ai_worker_leaves_malformed_response_pending_after_one_call(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    extraction_calls = 0

    def fake_extract(content: str, **_kwargs: object) -> str:
        nonlocal extraction_calls
        extraction_calls += 1
        return f"cleaned: {content}"

    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        fake_extract,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="threads",
        status=str(QueryStatus.CLAIMED),
        response_text="raw profile",
    )
    later_id = await db.save_result(
        username="blue",
        site_name="github",
        status=str(QueryStatus.CLAIMED),
        response_text="later profile",
    )
    update_calls = 0
    original_update = db.update_result_ai_extraction

    async def count_update(**kwargs: object) -> None:
        nonlocal update_calls
        update_calls += 1
        await original_update(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db, "update_result_ai_extraction", count_update)
    service = OutcomeAIService(
        [
            _structured_error(),
            SimpleNamespace(extraction={"full_name": ["Blue Example"]}),
        ]
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    reporter, output = _reporter()
    worker_task = asyncio.create_task(
        ai_worker(queue, db, service, reporter=reporter)
    )

    await queue.put(site_id)
    await queue.put(later_id)
    await _finish_worker(queue, worker_task)

    assert [call["site_name"] for call in service.calls] == [
        "threads",
        "github",
    ]
    assert service.outcomes == []
    assert extraction_calls == 2
    assert update_calls == 1
    assert await _get_ai_extraction(db, site_id) is None
    assert json.loads((await _get_ai_extraction(db, later_id)) or "") == {
        "full_name": ["Blue Example"]
    }
    assert await db.get_pending_ai_extraction_ids(
        "blue",
        contract_hash=PASS_ONE_CONTRACT_HASH,
    ) == [site_id]
    rendered = output.getvalue()
    assert "profile extraction failed; left pending" in rendered
    assert "retry" not in rendered.lower()
    assert reporter.ai_stats.pending == 1
    assert reporter.ai_stats.with_facts == 1


async def test_ai_worker_retries_pending_malformed_response_on_later_run(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    extraction_calls = 0

    def fake_extract(content: str, **_kwargs: object) -> str:
        nonlocal extraction_calls
        extraction_calls += 1
        return content

    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        fake_extract,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="threads",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    first_service = OutcomeAIService([_structured_error()])
    first_queue: asyncio.Queue[int] = asyncio.Queue()
    first_worker = asyncio.create_task(ai_worker(first_queue, db, first_service))
    await first_queue.put(site_id)
    await _finish_worker(first_queue, first_worker)

    assert await _get_ai_extraction(db, site_id) is None

    second_service = OutcomeAIService(
        [SimpleNamespace(extraction={"full_name": ["Blue Example"]})]
    )
    second_queue: asyncio.Queue[int] = asyncio.Queue()
    second_worker = asyncio.create_task(
        ai_worker(second_queue, db, second_service)
    )
    await second_queue.put(site_id)
    await _finish_worker(second_queue, second_worker)

    assert extraction_calls == 2
    assert len(first_service.calls) == 1
    assert len(second_service.calls) == 1
    assert json.loads((await _get_ai_extraction(db, site_id)) or "") == {
        "full_name": ["Blue Example"]
    }


async def test_ai_worker_processes_jobs_serially_and_drains_before_returning(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **kwargs: content,
    )
    site_ids = [
        await db.save_result(
            username="blue",
            site_name=f"site-{index}",
            status=str(QueryStatus.CLAIMED),
            response_text=f"profile-{index}",
        )
        for index in range(3)
    ]
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = BlockingAIService(expected_concurrency=1)
    worker_task = asyncio.create_task(ai_worker(queue, db, service))

    for site_id in site_ids:
        await queue.put(site_id)
    queue.shutdown()

    await asyncio.wait_for(service.at_capacity.wait(), timeout=1)
    await asyncio.sleep(0)
    assert service.max_active == 1
    assert not worker_task.done()

    service.release.set()
    await asyncio.wait_for(worker_task, timeout=2)
    await asyncio.wait_for(queue.join(), timeout=1)

    assert service.max_active == 1
    assert service.completed == len(site_ids)
    saved_extractions = [
        await _get_ai_extraction(db, site_id) for site_id in site_ids
    ]
    assert all(extraction is not None for extraction in saved_extractions)


async def test_ai_worker_cancellation_balances_queue_accounting(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **kwargs: content,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="site-1",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    completed_site_id = await db.save_result(
        username="blue",
        site_name="completed-site",
        status=str(QueryStatus.CLAIMED),
        response_text="completed profile",
        ai_extraction='{"full_name": ["Blue Example"]}',
        ai_extraction_contract_hash=PASS_ONE_CONTRACT_HASH,
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = BlockingAIService(expected_concurrency=1)
    worker_task = asyncio.create_task(ai_worker(queue, db, service))
    await queue.put(site_id)
    await asyncio.wait_for(service.at_capacity.wait(), timeout=1)

    worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker_task
    await asyncio.wait_for(queue.join(), timeout=1)

    assert service.active == 0
    assert service.completed == 0
    assert await _get_ai_extraction(db, site_id) is None
    assert json.loads(
        (await _get_ai_extraction(db, completed_site_id)) or "{}"
    ) == {"full_name": ["Blue Example"]}
    assert await db.get_pending_ai_extraction_ids(
        "blue",
        contract_hash=PASS_ONE_CONTRACT_HASH,
    ) == [site_id]


async def test_ai_worker_balances_missing_and_duplicate_jobs(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **kwargs: content,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="github",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = FakeAIService()
    reporter, _ = _reporter()
    worker_task = asyncio.create_task(
        ai_worker(queue, db, service, reporter=reporter)
    )

    await queue.put(999)
    await queue.put(site_id)
    await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert len(service.calls) == 1
    assert await _get_ai_extraction(db, site_id) is not None
    assert reporter.ai_stats.with_facts == 1
    assert reporter.ai_stats.skipped == 2


async def test_ai_worker_saves_empty_extraction_without_calling_model(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content: "",
    )
    site_id = await db.save_result(
        username="blue",
        site_name="empty",
        status=str(QueryStatus.CLAIMED),
        response_text="<html><body></body></html>",
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = FakeAIService()
    reporter, _ = _reporter()
    worker_task = asyncio.create_task(
        ai_worker(queue, db, service, reporter=reporter)
    )

    await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert service.calls == []
    saved = await _get_ai_extraction(db, site_id)
    assert saved is not None
    assert json.loads(saved) == {}
    assert await _get_ai_contract_hash(db, site_id) == PASS_ONE_CONTRACT_HASH
    assert reporter.ai_stats.no_facts == 1


async def test_ai_worker_reports_valid_model_empty_object_as_no_facts(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="username-only",
        status=str(QueryStatus.CLAIMED),
        response_text="7ghost",
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    service = OutcomeAIService([SimpleNamespace(extraction={})])
    reporter, _ = _reporter()
    worker_task = asyncio.create_task(
        ai_worker(queue, db, service, reporter=reporter)
    )

    await queue.put(site_id)
    await _finish_worker(queue, worker_task)

    assert len(service.calls) == 1
    saved = await _get_ai_extraction(db, site_id)
    assert saved is not None
    assert json.loads(saved) == {}
    assert await _get_ai_contract_hash(db, site_id) == PASS_ONE_CONTRACT_HASH
    assert reporter.ai_stats.with_facts == 0
    assert reporter.ai_stats.no_facts == 1
    assert reporter.ai_stats.pending == 0


async def test_ai_pipeline_buffers_fifo_jobs_while_model_loads(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    site_ids = [
        await db.save_result(
            username="blue",
            site_name=f"site-{index}",
            status=str(QueryStatus.CLAIMED),
            response_text=f"profile-{index}",
        )
        for index in range(3)
    ]
    service = FakeAIService()
    loading_started = asyncio.Event()
    release_model = asyncio.Event()

    async def fake_create(**_kwargs: object) -> FakeAIService:
        loading_started.set()
        await release_model.wait()
        return service

    monkeypatch.setattr(
        "sherlock_project.sherlock.AIService.create",
        fake_create,
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    pipeline_task = asyncio.create_task(run_ai_pipeline(queue, db))

    await asyncio.wait_for(loading_started.wait(), timeout=1)
    for site_id in site_ids:
        await queue.put(site_id)
    await asyncio.sleep(0)

    assert service.calls == []
    assert queue.qsize() == len(site_ids)

    queue.shutdown()
    release_model.set()
    returned_service = await asyncio.wait_for(pipeline_task, timeout=2)
    await asyncio.wait_for(queue.join(), timeout=1)

    assert returned_service is service
    assert [call["site_name"] for call in service.calls] == [
        "site-0",
        "site-1",
        "site-2",
    ]
    saved_extractions = [
        await _get_ai_extraction(db, site_id) for site_id in site_ids
    ]
    assert all(extraction is not None for extraction in saved_extractions)
    await service.close()
    assert service.close_calls == 1


async def test_ai_pipeline_ready_model_waits_for_queue_without_polling(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sherlock_project.sherlock.extract_profile_content",
        lambda content, **_kwargs: content,
    )
    site_id = await db.save_result(
        username="blue",
        site_name="late-site",
        status=str(QueryStatus.CLAIMED),
        response_text="profile",
    )
    service = FakeAIService()
    create_calls = 0
    model_ready = asyncio.Event()

    async def fake_create(**_kwargs: object) -> FakeAIService:
        nonlocal create_calls
        create_calls += 1
        model_ready.set()
        return service

    monkeypatch.setattr(
        "sherlock_project.sherlock.AIService.create",
        fake_create,
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    pipeline_task = asyncio.create_task(run_ai_pipeline(queue, db))

    await asyncio.wait_for(model_ready.wait(), timeout=1)
    await asyncio.sleep(0)
    assert create_calls == 1
    assert not pipeline_task.done()
    assert service.calls == []

    await queue.put(site_id)
    queue.shutdown()
    assert await asyncio.wait_for(pipeline_task, timeout=2) is service
    await asyncio.wait_for(queue.join(), timeout=1)
    assert [call["site_name"] for call in service.calls] == ["late-site"]
    await service.close()


async def test_ai_pipeline_model_failure_drains_jobs_as_pending(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    site_ids = [
        await db.save_result(
            username="blue",
            site_name=f"site-{index}",
            status=str(QueryStatus.CLAIMED),
            response_text=f"profile-{index}",
        )
        for index in range(2)
    ]
    loading_started = asyncio.Event()
    release_failure = asyncio.Event()

    async def failing_create(**_kwargs: object) -> FakeAIService:
        loading_started.set()
        await release_failure.wait()
        raise RuntimeError("sensitive LM Studio startup details")

    monkeypatch.setattr(
        "sherlock_project.sherlock.AIService.create",
        failing_create,
    )
    reporter, output = _reporter(verbose=True)
    reporter.ai_model_starting()
    queue: asyncio.Queue[int] = asyncio.Queue()
    pipeline_task = asyncio.create_task(
        run_ai_pipeline(queue, db, reporter=reporter)
    )

    await asyncio.wait_for(loading_started.wait(), timeout=1)
    for site_id in site_ids:
        await queue.put(site_id)
        reporter.ai_scheduled()
    release_failure.set()
    queue.shutdown()

    assert await asyncio.wait_for(pipeline_task, timeout=2) is None
    await asyncio.wait_for(queue.join(), timeout=1)
    reporter.ai_pass_finished()

    saved_extractions = [
        await _get_ai_extraction(db, site_id) for site_id in site_ids
    ]
    assert all(extraction is None for extraction in saved_extractions)
    assert reporter.ai_stats.completed == 2
    assert reporter.ai_stats.pending == 2
    assert reporter._ai_started is False
    rendered = output.getvalue()
    assert rendered.count("Local AI model unavailable") == 1
    assert "RuntimeError" in rendered
    assert "sensitive LM Studio startup details" not in rendered
    assert "Profile extraction 1/2" not in rendered
    assert "Profile extraction unavailable · 2 pending" in rendered


async def test_ai_pipeline_cancellation_closes_loaded_service_once(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    service = FakeAIService()
    model_ready = asyncio.Event()

    async def fake_create(**_kwargs: object) -> FakeAIService:
        model_ready.set()
        return service

    monkeypatch.setattr(
        "sherlock_project.sherlock.AIService.create",
        fake_create,
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    pipeline_task = asyncio.create_task(run_ai_pipeline(queue, db))

    await asyncio.wait_for(model_ready.wait(), timeout=1)
    await asyncio.sleep(0)
    pipeline_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pipeline_task

    queue.shutdown(immediate=True)
    await asyncio.wait_for(queue.join(), timeout=1)
    assert service.close_calls == 1


async def test_ai_pipeline_cancellation_during_model_loading_is_collected(
    db: SherlockDB,
    monkeypatch: pytest.MonkeyPatch,
):
    loading_started = asyncio.Event()
    cancellation_observed = asyncio.Event()
    never_ready = asyncio.Event()

    async def blocked_create(**_kwargs: object) -> FakeAIService:
        loading_started.set()
        try:
            await never_ready.wait()
        finally:
            cancellation_observed.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(
        "sherlock_project.sherlock.AIService.create",
        blocked_create,
    )
    queue: asyncio.Queue[int] = asyncio.Queue()
    pipeline_task = asyncio.create_task(run_ai_pipeline(queue, db))

    await asyncio.wait_for(loading_started.wait(), timeout=1)
    pipeline_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pipeline_task

    await asyncio.wait_for(cancellation_observed.wait(), timeout=1)
    queue.shutdown(immediate=True)
    await asyncio.wait_for(queue.join(), timeout=1)


async def test_sherlock_without_ai_keeps_caller_database_open(db: SherlockDB):
    results = await sherlock(
        username="invalid!",
        engine=object(),  # type: ignore[arg-type]
        db=db,
        site_data={
            "Example": {
                "urlMain": "https://example.com",
                "url": "https://example.com/{}",
                "regexCheck": "^[a-z]+$",
            }
        },
        query_notify=QueryNotify(),
    )

    assert results["Example"]["status"].status is QueryStatus.ILLEGAL
    assert await db.get_or_create_username_id("still-open") > 0


async def test_sherlock_forces_ai_refresh_for_fresh_targeted_result():
    class FakeResponse:
        elapsed = 0.1
        status = 200
        text = "profile content"
        url = "https://example.com/blue"

    class FakeEngine:
        def get_request_fn(self, _method: str) -> object:
            return object()

        async def fetch_with_api(self, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    class CapturingDB:
        def __init__(self) -> None:
            self.saved: list[dict[str, object]] = []

        async def save_result(self, **kwargs: object) -> int:
            self.saved.append(kwargs)
            return 73

    database = CapturingDB()
    enqueued: list[int] = []

    async def enqueue(site_id: int) -> None:
        enqueued.append(site_id)

    await sherlock(
        username="blue",
        engine=FakeEngine(),  # type: ignore[arg-type]
        db=database,  # type: ignore[arg-type]
        site_data={
            "Example": {
                "urlMain": "https://example.com",
                "url": "https://example.com/{}",
                "errorType": "status_code",
                "errorCode": 404,
            }
        },
        query_notify=QueryNotify(),
        enqueue_ai=enqueue,
        force_ai_extraction=True,
    )

    assert database.saved[0]["force_ai_extraction"] is True
    assert enqueued == [73]
