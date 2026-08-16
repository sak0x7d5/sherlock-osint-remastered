"""Assembles one scan for the TUI, out of the same parts `main()` uses.

Every function called here -- `sherlock`, `run_ai_pipeline`,
`synthesize_profiles`, `is_resumable`, `restore_saved_results`,
`_close_ai_service_and_db` -- is imported from `sherlock.py` rather than
reimplemented. That is the rule the whole TUI is built on: the screens are a
view, and there is one scan engine underneath. A second copy of the resume
filter or the AI shutdown order would drift from the CLI's, and only one of the
two would get the next fix.

What this module is NOT is a second `main()`. It deliberately does not handle
`--json` manifests, PR numbers, output files, multiple usernames, or
`--site`. Those are command-line concerns with no control on screen, and
inventing UI for them here would be building a worse version of a CLI that
already works. The TUI scans one username against the bundled manifest, which is
the default path and the one the resume behaviour is designed around.

**The shutdown order is copied exactly, and it matters.** `ai_queue.shutdown()`
then await the pipeline, then `join()`, then synthesis, then close. Reordering
any of it either hangs the scan or drops extractions -- the CLI paid for that
sequence and this inherits it rather than rediscovering it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sherlock_project.ai_config import AISettings, try_load_settings
from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.http_engine import HttpEngine
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.profile_synthesis import IdentityAnchor
from sherlock_project.settings import (
    NO_DEFAULT,
    SETTING_FIELDS,
    field_default,
)
from sherlock_project.sites import SitesInformation
from sherlock_project.tui.reporter import TuiReporter


def resolved(values: dict[str, Any], key: str) -> Any:
    """A stored setting, or what this build ships when nothing is stored.

    Reads the default off `field_default` rather than restating it, for the
    reason that function's own docstring gives: a default restated in a second
    place is a default free to disagree with the config loader. `None` means
    "nobody chose", never "off" -- the same `is None` test the CLI resolver
    uses, because falsiness would break every toggle whose real value is False.
    """
    value = values.get(key)
    if value is not None:
        return value
    for field in SETTING_FIELDS:
        if field.key == key:
            default = field_default(field)
            return None if default is NO_DEFAULT else default
    return None


@dataclass(frozen=True, slots=True)
class ScanPlan:
    """What a scan of this username would actually do, before it does it.

    Computed by `build_scan_plan` and used twice: once by the screen, to ask
    whether to resume or start over, and once by the scan itself. ONE function
    produces both, so the numbers in the question cannot disagree with what
    happens when it is answered -- which is the whole reason the resume state is
    worth showing up front at all.
    """

    site_data: dict[str, Any]
    saved_rows: dict[str, Any]
    site_data_all: dict[str, Any]

    @property
    def stored(self) -> int:
        return len(self.saved_rows)

    @property
    def to_scan(self) -> int:
        return len(self.site_data)

    @property
    def total(self) -> int:
        return len(self.site_data_all)

    @property
    def is_known(self) -> bool:
        """Whether anything is stored for this username at all."""
        return bool(self.saved_rows)


async def build_scan_plan(
    *,
    username: str,
    settings_values: dict[str, Any],
    fresh: bool = False,
    db: SherlockDB | None = None,
) -> ScanPlan:
    """Work out which sites a scan would check, and which it would skip.

    `db` is reused when the caller already has one open; otherwise a connection
    is made and closed here, which is what the screen's pre-scan peek does. The
    manifest load is around 11ms, cheap enough to repeat for a question asked
    once per scan.
    """
    from sherlock_project.sherlock import is_resumable

    use_browser = resolved(settings_values, "scan.webbrowser")

    sites = SitesInformation(honor_exclusions=False)
    if not resolved(settings_values, "scan.nsfw"):
        sites.remove_nsfw_sites(do_not_remove=[])
    site_data_all = {site.name: site.information for site in sites}

    owned = db is None
    connection = db or await SherlockDB.create(str(default_database_path()))
    try:
        stored_rows = await connection.get_saved_results(username=username)
    finally:
        if owned:
            await connection.close()

    # `fresh` reuses nothing, so nothing is resumable. The stored rows are still
    # reported, because "you are about to redo 680 sites" is the fact that makes
    # the choice a choice.
    saved_rows = (
        {}
        if fresh
        else {
            site_name: row
            for site_name, row in stored_rows.items()
            if is_resumable(row, using_browser=bool(use_browser))
        }
    )
    site_data = {
        name: info
        for name, info in site_data_all.items()
        if name not in saved_rows
    }
    return ScanPlan(
        site_data=site_data,
        saved_rows=saved_rows,
        site_data_all=site_data_all,
    )


async def peek_stored(
    *, username: str, settings_values: dict[str, Any]
) -> ScanPlan:
    """The resume view of a username, for the question asked before scanning."""
    return await build_scan_plan(
        username=username, settings_values=settings_values, fresh=False
    )


def ai_is_configured(values: dict[str, Any]) -> bool:
    """Whether a model has been chosen, and so whether analysis runs.

    The TUI has no `--ai` flag, and adding a per-run toggle would make a scan's
    behaviour depend on a control that is not visible in the record afterwards.
    A configured model means analysis runs; the scan pane states that on the
    line under the target field, so it is never a surprise.

    `reporter.analysis_is_on` is what draws that line, and a test asserts the
    two agree -- a config line promising analysis the scan will not run is
    worse than one that says nothing.
    """
    return bool(resolved(values, "ai.model"))


async def run_scan_session(
    *,
    username: str,
    reporter: TuiReporter,
    settings_values: dict[str, Any],
    use_ai: bool = False,
    fresh: bool = False,
    anchors: Sequence[IdentityAnchor] = (),
) -> None:
    """Scan one username, reporting through `reporter`.

    `use_ai` and `fresh` are per-run choices with no stored default, exactly
    like `--ai` and `--fresh` on the command line. They are NOT settings: both
    default to off on every run, and neither is read from the config file.

    That is a deliberate refusal to add a config layer the CLI does not have.
    A stored "always analyse" would mean the UI ran a model the identical CLI
    command would not, which is the two-homes-for-one-value problem that kept
    the AI tab from existing. If `--ai` ever grows a stored default it should
    grow one on both surfaces, through the settings resolver, at once.

    Raises nothing on a normal failure -- the reporter carries errors to the
    screen, and an exception escaping here would kill the Textual worker and
    take the pane's state with it. `CancelledError` is the exception: it is how
    the stop button reaches the engine, and it must propagate.
    """
    values = settings_values
    use_browser = resolved(values, "scan.webbrowser")
    concurrency = resolved(values, "scan.concurrency")
    timeout = resolved(values, "scan.timeout")
    proxy = resolved(values, "scan.proxy")

    if anchors and not use_ai:
        # Said before the scan rather than discovered after it. Anchors only
        # reach the second AI pass, so without analysis they are typed, stored
        # for the run, and never used -- and the profile that comes back would
        # carry the "may describe different people" caveat the anchors were
        # meant to remove.
        reporter.warning(
            f"{len(anchors)} anchor(s) were set but AI analysis is off, so "
            "they will not be used."
        )

    want_ai = use_ai
    if want_ai and not ai_is_configured(values):
        # Asked for, but impossible. Said out loud rather than quietly skipped:
        # a scan that silently produced no profile would look like the model
        # failed, and the actual problem is one unset setting.
        reporter.warning(
            "AI analysis was requested but no model is configured. "
            "Set one on the SETTINGS tab. Scanning without analysis."
        )
        want_ai = False

    # Imported before anything is opened, not at module scope. `sherlock.py`
    # pulls in the whole CLI at import time, and the private helpers below are
    # part of its scan lifecycle rather than a public surface -- importing them
    # here keeps that coupling visible at the point of use instead of buried in
    # a header. Done FIRST so that an ImportError cannot strand an open database
    # handle: everything after this line has a `finally` that closes it.
    from sherlock_project.ai_engine import pass_one_contract_hash
    from sherlock_project.sherlock import (
        _cancel_ai_pipeline_now,
        _close_ai_service_and_db,
        report_cached_ai_evidence,
        restore_saved_results,
        run_ai_pipeline,
        sherlock,
        synthesize_profiles,
    )

    db = await SherlockDB.create(str(default_database_path()))

    ai_settings: AISettings | None = None
    if want_ai:
        stored = try_load_settings()
        ai_settings = getattr(stored, "ai", None)
        if ai_settings is None:
            reporter.warning("No AI settings stored; analysis is off.")
            want_ai = False

    ai_queue: asyncio.Queue[int] | None = None
    ai_pipeline_task: asyncio.Task[Any] | None = None
    ai_pipeline_awaited = False
    ai_service: Any = None
    scheduled_ai_ids: set[int] = set()
    contract_hash: str | None = None
    interrupted = False
    cancellation_signalled = False

    def cancel_ai_pipeline() -> None:
        nonlocal cancellation_signalled
        if cancellation_signalled:
            return
        cancellation_signalled = True
        _cancel_ai_pipeline_now(
            ai_queue=ai_queue,
            ai_pipeline_task=ai_pipeline_task,
        )

    try:
        if want_ai:
            contract_hash = pass_one_contract_hash()
            await report_cached_ai_evidence(
                db=db,
                usernames=[username],
                contract_hash=contract_hash,
                reporter=reporter,
            )
            reporter.ai_model_starting()
            ai_queue = asyncio.Queue()
            ai_pipeline_task = asyncio.create_task(
                run_ai_pipeline(
                    ai_queue=ai_queue,
                    sherlock_db=db,
                    reporter=reporter,
                    ai_settings=ai_settings,
                ),
                name="tui-ai-pipeline",
            )
            # Let model loading start before the browser does. The model is the
            # slowest thing to become ready, and starting it first is most of
            # why the two overlap instead of queueing.
            await asyncio.sleep(0)

        async def enqueue_ai_result(site_id: int) -> None:
            if ai_queue is None or site_id in scheduled_ai_ids:
                return
            scheduled_ai_ids.add(site_id)
            try:
                await ai_queue.put(site_id)
                reporter.ai_scheduled()
            except BaseException:
                scheduled_ai_ids.remove(site_id)
                raise

        enqueue_ai = enqueue_ai_result if want_ai else None

        if not want_ai and ai_is_configured(values):
            # A model is configured and this run will not use it. Said only when
            # there is something it WOULD have done, so it cannot become noise
            # on every fast scan.
            #
            # This is the gap that made changing model look broken: analysis is
            # off by default on every run, matching `--ai`. Someone who has just
            # been to SETTINGS to choose a model has plainly signalled they want
            # it, then scans, and nothing runs -- no extraction, and not even the
            # stale-extraction warning, because that is only reported when
            # analysis is on. Silence, and the old model still on the profile.
            waiting = await db.get_pending_ai_extraction_ids(
                username, contract_hash=pass_one_contract_hash()
            )
            if waiting:
                reporter.warning(
                    f"Analysis is off, so {len(waiting)} sites with stored "
                    f"pages were not analysed."
                )
                reporter.hint(
                    "Turn 'analysis' on in OPTIONS to use the model you have "
                    "configured."
                )

        if ai_settings is not None:
            reporter.ai_extractions_from_other_models(
                username=username,
                configured_model=ai_settings.model,
                counts=await db.get_extraction_model_counts(username),
            )

        # The same function the screen used to ask whether to resume, so the
        # counts it offered and the work done here cannot diverge. It carries
        # the resume rule itself: skip what is already stored, gated on how it
        # was fetched, because `is_resumable` is what stops one browserless run
        # from freezing a username's record.
        plan = await build_scan_plan(
            username=username,
            settings_values=values,
            fresh=fresh,
            db=db,
        )
        saved_rows = plan.saved_rows
        site_data = plan.site_data
        site_data_all = plan.site_data_all

        restored = restore_saved_results(
            username=username,
            saved_rows=saved_rows,
            site_data_all=site_data_all,
        )
        reporter.restored_results(
            username=username,
            results=restored,
            to_scan=len(site_data),
        )

        # The engine is built ONLY if there is something to fetch, and this is
        # computed before it rather than inside it for exactly that reason. A
        # fully resumed username used to start Chromium, report it ready, and
        # then scan nothing -- seconds of visible browser startup for a run with
        # no work in it, which is indistinguishable on screen from a real scan
        # and is why a resumed run looked like it had re-checked everything.
        if site_data:
            if use_browser:
                engine: PlaywrightEngine | HttpEngine = PlaywrightEngine(
                    concurrency=concurrency,
                    headless=True,
                    proxy={"server": proxy} if proxy else None,
                    status_callback=reporter.browser_status,
                    cancellation_callback=cancel_ai_pipeline if want_ai else None,
                )
            else:
                reporter.browserless_transport()
                engine = HttpEngine(
                    concurrency=concurrency,
                    proxy=proxy,
                    cancellation_callback=cancel_ai_pipeline if want_ai else None,
                )

            async with engine as active_engine:
                started_at = perf_counter()
                await sherlock(
                    username=username,
                    engine=active_engine,
                    db=db,
                    site_data=site_data,
                    query_notify=reporter,
                    enqueue_ai=enqueue_ai,
                    proxy=proxy,
                    timeout=timeout,
                    # `fresh` means "reuse nothing stored for this username",
                    # extractions included. It is the only way a newly chosen
                    # model reaches hits already on disk -- without it a scan
                    # after a model change costs full price for byte-identical
                    # AI output.
                    force_ai_extraction=fresh,
                    on_cancel=cancel_ai_pipeline if want_ai else None,
                )
                reporter.finish_scan(elapsed_time=perf_counter() - started_at)
        elif not saved_rows:
            # Nothing stored AND nothing to scan means an empty manifest, which
            # `restored_results` says nothing about because there is nothing to
            # restore. Every other "already stored" case is reported there,
            # where the counts are.
            reporter.warning("No sites to check.")

        if enqueue_ai is not None:
            if contract_hash is None:
                raise RuntimeError("Pass 1 contract hash is unavailable")
            for site_id in await db.get_pending_ai_extraction_ids(
                username, contract_hash=contract_hash
            ):
                await enqueue_ai(site_id)

        if ai_queue is not None and ai_pipeline_task is not None:
            reporter.ai_draining()
            ai_queue.shutdown()
            ai_service = await ai_pipeline_task
            ai_pipeline_awaited = True
            await ai_queue.join()
            reporter.ai_pass_finished()

        if ai_service is not None:
            await synthesize_profiles(
                db=db,
                ai_service=ai_service,
                usernames=[username],
                # An anchored run must rebuild rather than reuse. The cached
                # summary was synthesised from the same evidence but WITHOUT
                # these anchors, so a cache hit would return the aggregate
                # profile and silently ignore everything the user just typed.
                force=bool(anchors),
                inline_anchors=list(anchors),
                reporter=reporter,
            )

    except asyncio.CancelledError:
        interrupted = True
        raise

    finally:
        if ai_pipeline_task is not None and not ai_pipeline_awaited:
            cancel_ai_pipeline()
            outcome = (
                await asyncio.gather(ai_pipeline_task, return_exceptions=True)
            )[0]
            if (
                ai_service is None
                and outcome is not None
                and not isinstance(outcome, BaseException)
            ):
                ai_service = outcome

        # Both closers are attempted even when the first fails, and the DB is
        # closed whatever happened to the model. Same reasoning as the CLI's
        # `_close_ai_service_and_db`: a scan that cannot shut its model down
        # must still not leave the database handle open.
        cleanup_error, cleanup_cancellation = await _close_ai_service_and_db(
            ai_service=ai_service,
            db=db,
        )
        if cleanup_cancellation is not None:
            interrupted = True
        if cleanup_error is not None and not interrupted:
            reporter.failure(f"Cleanup failed: {cleanup_error}")

        if interrupted:
            reporter.processing_interrupted()
