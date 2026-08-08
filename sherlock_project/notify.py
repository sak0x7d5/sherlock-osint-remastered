"""Terminal reporting for Sherlock scans and AI processing."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from sherlock_project.result import QueryResult, QueryStatus

if TYPE_CHECKING:
    from sherlock_project.ai_engine import AIRequestTrace, StructuredResponseError
    from sherlock_project.profile_synthesis import ProfileSynthesis


AIOutcome = Literal["with_facts", "no_facts", "pending", "skipped"]
BrowserStatus = Literal["installing", "starting", "ready"]
INTERRUPTION_MESSAGE = (
    "Processing interrupted; committed results were saved and pending AI "
    "work can resume on the next --ai run."
)


@dataclass(frozen=True, slots=True)
class AIProgressStats:
    scheduled: int
    completed: int
    with_facts: int
    no_facts: int
    pending: int
    skipped: int


class QueryNotify:
    """No-op notification interface used by library callers and tests."""

    def __init__(self, result: QueryResult | None = None) -> None:
        self.result = result

    def start(self, message: str | None = None, total: int | None = None) -> None:
        pass

    def update(self, result: QueryResult) -> None:
        self.result = result

    def finish_scan(self, elapsed_time: float = 0) -> None:
        pass

    def raw(self, message: str) -> None:
        print(message)

    def finish(
        self,
        elapsed_time: float = 0,
        message: str = "The processing has been finished.",
    ) -> None:
        pass

    def processing_interrupted(self) -> None:
        pass

    def __str__(self) -> str:
        return str(self.result)


class TerminalReporter(QueryNotify):
    """One progress-safe output surface for scan, browser, and AI events."""

    def __init__(
        self,
        result: QueryResult | None = None,
        verbose: bool = False,
        print_all: bool = False,
        browse: bool = False,
        no_color: bool = False,
        *,
        console: Console | None = None,
        error_console: Console | None = None,
    ) -> None:
        super().__init__(result)
        self.verbose = verbose
        self.print_all = print_all
        self.browse = browse
        self.no_color = no_color
        console_options = {
            "highlight": False,
            "no_color": no_color,
            "color_system": None if no_color else "auto",
        }
        self.console = console or Console(**console_options)
        self.error_console = error_console or Console(stderr=True, **console_options)
        self.interactive = bool(self.console.is_terminal and not no_color)
        self._progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn(
                "{task.description}",
                markup=False,
                table_column=Column(max_width=18, no_wrap=True, overflow="ellipsis"),
            ),
            BarColumn(bar_width=12),
            TextColumn(
                "{task.fields[status]}",
                markup=False,
                table_column=Column(max_width=32, no_wrap=True, overflow="ellipsis"),
            ),
            TimeElapsedColumn(),
            console=self.console,
            disable=not self.interactive,
            transient=False,
        )
        self._progress_started = False
        self._scan_task_id: TaskID | None = None
        self._scan_username = ""
        self._scan_total = 0
        self._scan_completed = 0
        self._scan_found = 0
        self._ai_model_task_id: TaskID | None = None
        self._ai_model_started_at: float | None = None
        self._ai_model_finished = False
        self._ai_model_unavailable = False
        self._web_scanner_task_id: TaskID | None = None
        self._web_scanner_started_at: float | None = None
        self._web_scanner_finished = False
        self._web_scanner_status: BrowserStatus | None = None
        self._ai_task_id: TaskID | None = None
        self._ai_started = False
        self._ai_finished = False
        self._ai_scheduling_closed = False
        self._ai_current_site = ""
        self._ai_scheduled = 0
        self._ai_completed = 0
        self._ai_with_facts = 0
        self._ai_no_facts = 0
        self._ai_pending = 0
        self._ai_skipped = 0
        self._ai_started_at: float | None = None
        self._ai_calls = 0
        self._ai_input_tokens = 0
        self._ai_output_tokens = 0
        self._ai_reasoning_tokens = 0
        self._ai_generation_seconds = 0.0
        self._ai_verbose_warning_shown = False
        self._ai_cache_reasoning_notice_shown = False
        self._synthesis_task_id: TaskID | None = None
        self._interruption_reported = False

    @property
    def ai_stats(self) -> AIProgressStats:
        return AIProgressStats(
            scheduled=self._ai_scheduled,
            completed=self._ai_completed,
            with_facts=self._ai_with_facts,
            no_facts=self._ai_no_facts,
            pending=self._ai_pending,
            skipped=self._ai_skipped,
        )

    def _ensure_progress(self) -> None:
        if self.interactive and not self._progress_started:
            self._progress.start()
            self._progress_started = True

    def close(self) -> None:
        for task_id in (
            self._ai_model_task_id,
            self._web_scanner_task_id,
            self._scan_task_id,
            self._ai_task_id,
            self._synthesis_task_id,
        ):
            self._remove_progress_task(task_id)
        self._ai_model_task_id = None
        self._web_scanner_task_id = None
        self._scan_task_id = None
        self._ai_task_id = None
        self._synthesis_task_id = None
        self._scan_username = ""
        if self._progress_started:
            self._progress.stop()
            self._progress_started = False

    def _remove_progress_task(self, task_id: TaskID | None) -> None:
        if task_id is None:
            return
        try:
            self._progress.remove_task(task_id)
        except KeyError:
            # Finalization and interruption may race through the same cleanup.
            pass

    @staticmethod
    def _elapsed_since(started_at: float | None) -> float:
        return perf_counter() - started_at if started_at is not None else 0.0

    @staticmethod
    def _elapsed_detail(elapsed: float) -> str | None:
        return f"{elapsed:.2f}s" if elapsed else None

    def _write(self, message: RenderableType, *, error: bool = False) -> None:
        if error:
            self.error_console.print(message)
        elif self._progress_started:
            self._progress.console.print(message)
        else:
            self.console.print(message)

    def _event(
        self,
        marker: str,
        marker_style: str,
        message: str,
        *,
        message_style: str = "",
        detail: str | None = None,
        error: bool = False,
    ) -> None:
        line = Text()
        line.append("[", style="bold white")
        line.append(marker, style=f"bold {marker_style}")
        line.append("]", style="bold white")
        line.append(f" {message}", style=message_style)
        if detail:
            line.append(f" ({detail})", style="dim")
        self._write(line, error=error)

    def info(self, message: str, *, detail: str | None = None) -> None:
        self._event("*", "yellow", message, detail=detail)

    def active(self, message: str, *, detail: str | None = None) -> None:
        self._event(">", "cyan", message, detail=detail)

    def success(self, message: str, *, detail: str | None = None) -> None:
        self._event("+", "green", message, message_style="green", detail=detail)

    def warning(self, message: str, *, detail: str | None = None) -> None:
        self._event("!", "yellow", message, message_style="yellow", detail=detail)

    def failure(self, message: str, *, detail: str | None = None) -> None:
        self._event("x", "red", message, message_style="red", detail=detail)

    def fatal(self, message: str, *, detail: str | None = None) -> None:
        self.close()
        self._event(
            "x",
            "red",
            message,
            message_style="red",
            detail=detail,
            error=True,
        )

    def debug(self, message: str) -> None:
        if self.verbose:
            self._event("*", "blue", message, message_style="dim")

    def raw(self, message: str) -> None:
        self._write(Text(message))

    def update_available(self, current: str, latest: str, url: str) -> None:
        self.info(
            f"Sherlock {latest} is available",
            detail=f"current {current}; {url}",
        )

    def browser_status(self, status: BrowserStatus) -> None:
        if self._web_scanner_finished:
            return
        if status == self._web_scanner_status:
            return
        self._web_scanner_status = status

        if status == "installing":
            if self._web_scanner_started_at is None:
                self._web_scanner_started_at = perf_counter()
            if self.interactive:
                self._ensure_progress()
                if self._web_scanner_task_id is None:
                    self._web_scanner_task_id = self._progress.add_task(
                        "Web scanner",
                        total=None,
                        status="installing runtime",
                    )
                else:
                    self._progress.update(
                        self._web_scanner_task_id,
                        status="installing runtime",
                    )
            else:
                self.warning(
                    "Web scanner runtime is missing; installing it now"
                )
        elif status == "starting":
            if self._web_scanner_started_at is None:
                self._web_scanner_started_at = perf_counter()
            if self.interactive:
                self._ensure_progress()
                if self._web_scanner_task_id is None:
                    self._web_scanner_task_id = self._progress.add_task(
                        "Web scanner",
                        total=None,
                        status="preparing",
                    )
                else:
                    self._progress.update(
                        self._web_scanner_task_id,
                        status="preparing",
                    )
            else:
                self.active("Preparing web scanner")
        elif status == "ready":
            elapsed = self._elapsed_since(self._web_scanner_started_at)
            self._remove_progress_task(self._web_scanner_task_id)
            self._web_scanner_task_id = None
            self._web_scanner_finished = True
            self.success(
                "Web scanner ready",
                detail=self._elapsed_detail(elapsed),
            )
        else:
            raise ValueError(f"Unknown browser status: {status!r}")

    def targeted_ai_mode(self) -> None:
        self.info(
            "Targeted AI mode: refreshing selected sites and extracting "
            "profile facts only"
        )

    def targeted_ai_complete(self) -> None:
        self.success(
            "Targeted profile extraction complete; profile synthesis skipped"
        )

    def start(self, message: str | None = None, total: int | None = None) -> None:
        if self._scan_task_id is not None:
            self.finish_scan()

        self._scan_username = message or ""
        self._scan_total = max(total or 0, 0)
        self._scan_completed = 0
        self._scan_found = 0
        site_word = "site" if self._scan_total == 1 else "sites"
        total_text = (
            f" across {self._scan_total} {site_word}"
            if total is not None
            else ""
        )
        if self.interactive:
            self._ensure_progress()
            self._scan_task_id = self._progress.add_task(
                f"Scan: {self._scan_username}",
                total=self._scan_total or None,
                status=f"0/{self._scan_total or '?'} · 0 found",
            )
        else:
            self.info(f"Checking username {self._scan_username!r}{total_text}")

    def update(self, result: QueryResult) -> None:
        super().update(result)
        self._scan_completed += 1
        response_time = None
        if result.query_time is not None and self.verbose:
            response_time = f"{round(result.query_time * 1000)} ms"

        if result.status is QueryStatus.CLAIMED:
            self._scan_found += 1
            # A hit backed by only one of the rule's two signals is worth
            # flagging. Silently presenting it alongside a fully confirmed
            # match is what invites a weak result to be read as a certainty.
            qualifier = ""
            if result.confidence is not None and str(result.confidence) != "Confirmed":
                qualifier = f" [{result.confidence}]"
            self.success(
                f"{result.site_name}: {result.site_url_user}{qualifier}",
                detail=response_time,
            )
            if self.browse:
                webbrowser.open(result.site_url_user, 2)
        elif result.status is QueryStatus.AVAILABLE:
            if self.print_all:
                self.failure(f"{result.site_name}: not found", detail=response_time)
        elif result.status is QueryStatus.UNKNOWN:
            if self.print_all:
                self.failure(
                    f"{result.site_name}: {result.context or 'unknown response'}",
                    detail=response_time,
                )
        elif result.status is QueryStatus.ILLEGAL:
            if self.print_all:
                self.warning(f"{result.site_name}: illegal username format")
        elif result.status is QueryStatus.WAF:
            if self.print_all:
                self.warning(
                    f"{result.site_name}: blocked by bot detection; a proxy may help",
                    detail=response_time,
                )
        else:
            raise ValueError(
                f"Unknown Query Status {result.status!r} for site {result.site_name!r}"
            )

        if self._scan_task_id is not None:
            self._progress.update(
                self._scan_task_id,
                completed=self._scan_completed,
                description=f"Scan: {result.site_name}",
                status=(
                    f"{self._scan_completed}/{self._scan_total or '?'} · "
                    f"{self._scan_found} found"
                ),
            )

    def finish_scan(self, elapsed_time: float = 0) -> None:
        if not self._scan_username and self._scan_task_id is None:
            return

        if self._scan_task_id is not None:
            if self._scan_total:
                self._progress.update(
                    self._scan_task_id,
                    completed=min(self._scan_completed, self._scan_total),
                    status=(
                        f"{self._scan_completed}/{self._scan_total} · "
                        f"{self._scan_found} found"
                    ),
                )
            self._remove_progress_task(self._scan_task_id)
            self._scan_task_id = None

        elapsed = f"{elapsed_time:.2f}s" if elapsed_time else None
        site_word = "site" if self._scan_completed == 1 else "sites"
        self.success(
            f"Scan complete for {self._scan_username!r}: "
            f"{self._scan_found} found across {self._scan_completed} {site_word}",
            detail=elapsed,
        )
        self._scan_username = ""

    def ai_model_starting(self) -> None:
        if self._ai_model_finished or self._ai_model_started_at is not None:
            return
        self._ai_model_started_at = perf_counter()
        if self.interactive:
            self._ensure_progress()
            self._ai_model_task_id = self._progress.add_task(
                "Local AI model",
                total=None,
                status="preparing",
            )
        else:
            self.active("Preparing local AI model")

    def ai_configuration(
        self,
        *,
        model: str,
        base_url: str,
        temperature: float,
        context_length: int,
    ) -> None:
        if not self.verbose:
            return
        self.debug(
            f"AI configuration: model={model}, provider={base_url}, "
            f"temperature={temperature}, context={context_length}, "
            "native reasoning=pass1 off/pass2 on"
        )

    def ai_cached_evidence(
        self,
        username: str,
        records: Sequence[tuple[str, str | None]],
    ) -> None:
        if not self.verbose:
            return
        cached = [
            (site_name, payload)
            for site_name, payload in records
            if payload is not None
        ]
        pending = sum(payload is None for _, payload in records)
        self.info(
            f"Saved AI evidence for {username!r}",
            detail=f"cached {len(cached)}, pending {pending}",
        )
        if not cached:
            return
        if not self._ai_cache_reasoning_notice_shown:
            self._ai_cache_reasoning_notice_shown = True
            self.warning(
                "Cached pass-one extractions do not retain transient reasoning; "
                "refresh a site to observe a new rationale"
            )
        for site_name, payload in cached:
            assert payload is not None
            try:
                parsed = json.loads(payload)
                rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
                border_style = "green"
            except (json.JSONDecodeError, TypeError):
                rendered = payload
                border_style = "red"
            self._write(
                Panel(
                    Text(rendered),
                    title=f"Cached pass-one extraction: {site_name}",
                    border_style=border_style,
                )
            )

    def ai_model_ready(self) -> None:
        if self._ai_model_finished:
            return
        elapsed = self._elapsed_since(self._ai_model_started_at)
        self._remove_progress_task(self._ai_model_task_id)
        self._ai_model_task_id = None
        self._ai_model_finished = True
        self.success(
            "Local AI model ready",
            detail=self._elapsed_detail(elapsed),
        )

    def ai_model_failed(self, error: Exception) -> None:
        if self._ai_model_finished:
            return
        elapsed = self._elapsed_since(self._ai_model_started_at)
        self._remove_progress_task(self._ai_model_task_id)
        self._ai_model_task_id = None
        self._ai_model_finished = True
        self._ai_model_unavailable = True
        detail = type(error).__name__ if self.verbose else None
        if self.verbose and elapsed:
            detail = (
                f"{detail}; {elapsed:.2f}s"
                if detail is not None
                else f"{elapsed:.2f}s"
            )
        self.failure(
            "Local AI model unavailable; web scanning will continue",
            detail=detail,
        )

    def ai_pass_started(self) -> None:
        if self._ai_started or self._ai_finished:
            return
        self._ai_started = True
        self._ai_started_at = perf_counter()
        if self.interactive:
            self._ensure_progress()
            self._ai_task_id = self._progress.add_task(
                "Profile extraction",
                total=(
                    self._ai_scheduled
                    if self._ai_scheduling_closed and self._ai_scheduled
                    else None
                ),
                status=self._ai_status(),
            )
        else:
            detail = (
                f"{self._ai_scheduled} queued"
                if self._ai_scheduled
                else "waiting for sites"
            )
            self.info("Profile extraction started", detail=detail)

    def ai_scheduled(self) -> None:
        if self._ai_finished:
            return
        self._ai_scheduled += 1
        if self._ai_task_id is not None:
            update: dict[str, object] = {"status": self._ai_status()}
            if self._ai_scheduling_closed:
                update["total"] = self._ai_scheduled
            self._progress.update(self._ai_task_id, **update)

    def ai_job_started(self, site_name: str) -> None:
        if not self._ai_started:
            self.ai_pass_started()
        if self._ai_scheduled <= self._ai_completed:
            self.ai_scheduled()
        self._ai_current_site = site_name
        if self._ai_task_id is not None:
            self._progress.update(
                self._ai_task_id,
                status=self._ai_status(),
            )
        elif not self.interactive:
            self.active(
                f"Profile extraction "
                f"{self._ai_completed}/{self._ai_scheduled} · {site_name}"
            )

    @staticmethod
    def _structured_diagnostics(error: StructuredResponseError) -> str:
        return error.safe_diagnostics()

    def ai_failed(self, site_name: str, error: Exception) -> None:
        if self.verbose and hasattr(error, "validation_error"):
            detail = self._structured_diagnostics(error)  # type: ignore[arg-type]
        elif self.verbose:
            detail = type(error).__name__
        else:
            detail = None
        self.failure(
            f"{site_name}: profile extraction failed; left pending for a future run",
            detail=detail,
        )

    @staticmethod
    def _trace_metrics(trace: AIRequestTrace) -> str:
        parts = [f"{trace.elapsed_seconds:.2f}s"]
        if trace.stats.input_tokens is not None:
            parts.append(f"in {trace.stats.input_tokens}")
        if trace.stats.output_tokens is not None:
            parts.append(f"out {trace.stats.output_tokens}")
        if trace.stats.reasoning_tokens is not None:
            parts.append(f"native reasoning {trace.stats.reasoning_tokens}")
        if trace.stats.tokens_per_second is not None:
            parts.append(f"{trace.stats.tokens_per_second:.2f} tok/s")
        if trace.stats.time_to_first_token_seconds is not None:
            parts.append(f"TTFT {trace.stats.time_to_first_token_seconds:.2f}s")
        return " | ".join(parts)

    def ai_trace(self, trace: AIRequestTrace) -> None:
        if trace.phase == "pass_one":
            self._ai_calls += 1
            self._ai_input_tokens += trace.stats.input_tokens or 0
            self._ai_output_tokens += trace.stats.output_tokens or 0
            self._ai_reasoning_tokens += trace.stats.reasoning_tokens or 0
            self._ai_generation_seconds += trace.elapsed_seconds

        if not self.verbose:
            return
        if not self._ai_verbose_warning_shown:
            self._ai_verbose_warning_shown = True
            self.warning(
                "Verbose AI diagnostics may contain OSINT evidence or personal data"
            )

        renderables: list[RenderableType] = []
        if trace.native_reasoning or (trace.stats.reasoning_tokens or 0) > 0:
            if trace.phase == "pass_one":
                self.warning(
                    f"{trace.site_name}: LM Studio returned native reasoning "
                    "despite reasoning-off mode",
                    detail=f"{trace.stats.reasoning_tokens or 0} tokens",
                )
            if trace.native_reasoning:
                renderables.append(
                    Panel(
                        Text(trace.native_reasoning),
                        title=(
                            "Unexpected native reasoning"
                            if trace.phase == "pass_one"
                            else "Native reasoning (transient)"
                        ),
                        border_style=(
                            "yellow" if trace.phase == "pass_one" else "cyan"
                        ),
                    )
                )
        if trace.structured_reasoning:
            renderables.append(
                Panel(
                    Text(trace.structured_reasoning),
                    title="Manual reasoning (transient)",
                    border_style="cyan",
                )
            )

        if trace.validated_output is not None:
            visible_output = dict(trace.validated_output)
            visible_output.pop("reasoning", None)
            rendered_output = json.dumps(
                visible_output,
                ensure_ascii=False,
                indent=2,
            )
            output_title = (
                "Validated extraction"
                if trace.phase == "pass_one"
                else "Validated target decision"
            )
        else:
            rendered_output = trace.final_text or "<empty response>"
            output_title = "Malformed final response"
        renderables.append(
            Panel(
                Text(rendered_output),
                title=output_title,
                border_style="green" if trace.validated_output is not None else "red",
            )
        )
        title = (
            f"AI {trace.phase.replace('_', ' ')}: {trace.site_name} "
            f"(attempt {trace.attempt})"
        )
        if trace.validation_error is not None:
            title += f" [{trace.validation_error}]"
        self._write(
            Panel(
                Group(*renderables),
                title=title,
                subtitle=self._trace_metrics(trace),
                border_style="blue",
            )
        )

    def ai_job_finished(
        self,
        outcome: AIOutcome,
    ) -> None:
        self._record_ai_outcome(
            outcome,
            announce_checkpoint=True,
        )

    def ai_job_deferred(self) -> None:
        """Count a queued job as pending without printing per-site failures."""
        self._record_ai_outcome(
            "pending",
            announce_checkpoint=False,
        )

    def _record_ai_outcome(
        self,
        outcome: AIOutcome,
        *,
        announce_checkpoint: bool,
    ) -> None:
        self._ai_completed += 1
        self._ai_current_site = ""
        if outcome == "with_facts":
            self._ai_with_facts += 1
        elif outcome == "no_facts":
            self._ai_no_facts += 1
        elif outcome == "pending":
            self._ai_pending += 1
        elif outcome == "skipped":
            self._ai_skipped += 1
        else:
            raise ValueError(f"Unknown AI outcome: {outcome!r}")

        if self._ai_task_id is not None:
            self._progress.update(
                self._ai_task_id,
                completed=self._ai_completed,
                status=self._ai_status(),
            )
        elif not self.interactive and announce_checkpoint and self.verbose:
            self.info(
                f"Profile extraction "
                f"{self._ai_completed}/{self._ai_scheduled}",
                detail=self._ai_counts(),
            )

    def ai_draining(self) -> None:
        if self._ai_finished:
            return
        self._ai_scheduling_closed = True
        if self._ai_task_id is not None:
            update: dict[str, object] = {
                "completed": self._ai_completed,
                "status": self._ai_status(draining=True),
            }
            if self._ai_scheduled:
                update["total"] = self._ai_scheduled
            self._progress.update(self._ai_task_id, **update)
        elif self._ai_started and self._ai_scheduled > self._ai_completed:
            self.info(
                "Web scanning is complete; finishing profile extraction",
                detail=f"{self._ai_completed}/{self._ai_scheduled} complete",
            )

    def ai_pass_finished(self) -> None:
        if self._ai_finished:
            return
        self._ai_finished = True
        self._ai_scheduling_closed = True
        if self._ai_task_id is not None:
            if self._ai_scheduled or self._ai_completed:
                self._progress.update(
                    self._ai_task_id,
                    completed=self._ai_completed,
                    total=max(self._ai_scheduled, self._ai_completed),
                    status=self._ai_status(),
                )
            self._remove_progress_task(self._ai_task_id)
            self._ai_task_id = None

        if not self._ai_scheduled:
            self.info("Profile extraction · no sites required processing")
            return

        if self._ai_model_unavailable:
            self.warning(
                f"Profile extraction unavailable · {self._ai_pending} pending"
            )
            return

        parts = [
            f"Profile extraction {self._ai_completed}/{self._ai_scheduled}",
            f"{self._ai_with_facts} with facts",
            f"{self._ai_no_facts} no facts",
        ]
        if self._ai_pending:
            parts.append(f"{self._ai_pending} pending")
        if self._ai_skipped:
            parts.append(f"{self._ai_skipped} skipped")
        message = " · ".join(parts)

        diagnostics = self._ai_verbose_metrics()
        if diagnostics is not None:
            self.debug(f"Profile extraction diagnostics: {diagnostics}")
        if self._ai_pending:
            self.warning(message)
        else:
            self.success(message)

    def processing_interrupted(self) -> None:
        if self._interruption_reported:
            return
        self._interruption_reported = True
        self._ai_model_finished = True
        self._web_scanner_finished = True
        self._ai_finished = True
        self._remove_progress_task(self._ai_model_task_id)
        self._ai_model_task_id = None
        self._remove_progress_task(self._web_scanner_task_id)
        self._web_scanner_task_id = None

        if self._scan_task_id is not None:
            self._remove_progress_task(self._scan_task_id)
            self._scan_task_id = None
        self._scan_username = ""

        if self._ai_task_id is not None:
            self._remove_progress_task(self._ai_task_id)
            self._ai_task_id = None

        self._clear_synthesis_task()
        self.close()
        self.warning(INTERRUPTION_MESSAGE)

    def _ai_status(self, *, draining: bool = False) -> str:
        status = f"{self._ai_completed}/{self._ai_scheduled}"
        if self._ai_current_site:
            return f"{status} · {self._ai_current_site}"
        if draining and self._ai_completed < self._ai_scheduled:
            return f"{status} · finishing queue"
        if not self._ai_scheduled:
            return f"{status} · waiting for sites"
        return status

    def _ai_counts(self) -> str:
        parts = [
            f"{self._ai_with_facts} with facts",
            f"{self._ai_no_facts} no facts",
        ]
        if self._ai_pending:
            parts.append(f"{self._ai_pending} pending")
        if self._ai_skipped:
            parts.append(f"skipped {self._ai_skipped}")
        return ", ".join(parts)

    def _ai_verbose_metrics(self) -> str | None:
        if not self.verbose:
            return None

        metrics = [f"requests {self._ai_calls}"]
        if self._ai_input_tokens or self._ai_output_tokens:
            metrics.append(
                f"tokens {self._ai_input_tokens} in / "
                f"{self._ai_output_tokens} out"
            )
        if self._ai_reasoning_tokens:
            metrics.append(f"native reasoning {self._ai_reasoning_tokens}")
        if self._ai_generation_seconds:
            metrics.append(
                f"request time {self._ai_generation_seconds:.2f}s"
            )
        elapsed = self._elapsed_since(self._ai_started_at)
        if elapsed:
            metrics.append(f"wall time {elapsed:.2f}s")
        return "; ".join(metrics)

    def synthesis_started(self, username: str) -> None:
        self._clear_synthesis_task()
        if self.interactive:
            self._ensure_progress()
            self._synthesis_task_id = self._progress.add_task(
                f"Building profile for {username}",
                total=None,
                status="resolving identity",
            )
        else:
            self.active(f"Building AI profile for {username!r}")

    def synthesis_failed(self, username: str, error: Exception) -> None:
        self._clear_synthesis_task()
        detail = type(error).__name__ if self.verbose else None
        self.failure(
            f"AI profile synthesis failed for {username!r}; previous profile retained",
            detail=detail,
        )

    def synthesis_finished(
        self,
        username: str,
        profile: ProfileSynthesis,
        *,
        cache_hit: bool,
    ) -> None:
        self._clear_synthesis_task()
        cache_status = "loaded from cache" if cache_hit else "updated"
        self.success(
            f"AI profile {cache_status} for {username!r}",
            detail=profile.resolution_status,
        )
        self.render_profile(profile)

    def _clear_synthesis_task(self) -> None:
        if self._synthesis_task_id is not None:
            self._remove_progress_task(self._synthesis_task_id)
            self._synthesis_task_id = None

    def render_profile(self, profile: ProfileSynthesis) -> None:
        source_labels = {
            decision.site_id: (
                f"{decision.site_name} — {decision.site_url}"
                if decision.site_url
                else decision.site_name
            )
            for decision in profile.source_decisions
        }
        provenance = {
            (item.field, item.value): item
            for item in profile.provenance
        }
        color_matches = profile.mode == "anchored" and not self.no_color

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()

        display_data: dict[
            str,
            list[tuple[str, str | None, list[str]]],
        ] = {}
        seen: set[tuple[str, str]] = set()

        def add_profile_values(
            values_by_field: dict[str, object],
            style: str | None,
        ) -> None:
            for field_name, raw_values in values_by_field.items():
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for raw_value in values:
                    if not isinstance(raw_value, str):
                        continue
                    value = raw_value.strip()
                    key = (field_name, value)
                    if value and key not in seen:
                        seen.add(key)
                        item = (
                            provenance.get((field_name, raw_value))
                            or provenance.get(key)
                        )
                        labels = (
                            [
                                source_labels.get(site_id, f"site {site_id}")
                                for site_id in item.source_site_ids
                            ]
                            if item is not None
                            else []
                        )
                        display_data.setdefault(field_name, []).append(
                            (value, style, list(dict.fromkeys(labels)))
                        )

        add_profile_values(
            profile.strong_profile,
            "green" if color_matches else None,
        )
        add_profile_values(
            profile.unsure_profile,
            "yellow" if color_matches else None,
        )

        has_colored_value = bool(display_data) and color_matches

        for field_name in sorted(display_data):
            rendered_values = Text()
            for index, (value, style, labels) in enumerate(display_data[field_name]):
                if index:
                    rendered_values.append("\n")
                rendered_values.append(value, style=style)
                if labels:
                    rendered_values.append(
                        f" [{', '.join(labels)}]",
                        style="dim",
                    )
            table.add_row(field_name, rendered_values)

        if profile.mode == "anchored":
            source_groups = (
                (
                    "strong matches",
                    lambda decision: decision.identity_status == "strong_match",
                ),
                (
                    "unsure matches",
                    lambda decision: decision.identity_status == "unsure",
                ),
                (
                    "excluded",
                    lambda decision: decision.disposition == "excluded",
                ),
                (
                    "failed",
                    lambda decision: decision.disposition == "failed",
                ),
            )
            for label, predicate in source_groups:
                sites = [
                    decision.site_name
                    for decision in profile.source_decisions
                    if predicate(decision)
                ]
                if sites:
                    table.add_row(label, ", ".join(sites))

        if display_data or profile.mode == "anchored":
            if has_colored_value:
                legend = Text("Match key: ", style="bold")
                legend.append("strong identity match", style="green")
                legend.append("  ")
                legend.append("unsure identity match", style="yellow")
                self._write(legend)
            self._write(table)
        else:
            self.info("No profile facts were available")
        for warning in profile.warnings:
            self.warning(warning)

    def finish(
        self,
        elapsed_time: float = 0,
        message: str = "The processing has been finished.",
    ) -> None:
        if self._scan_username or self._scan_task_id is not None:
            self.finish_scan()
        self._clear_synthesis_task()
        self.close()
        detail = f"{elapsed_time:.2f}s total" if elapsed_time else None
        self.info("Processing complete", detail=detail)

    def __str__(self) -> str:
        return str(self.result)


class QueryNotifyPrint(TerminalReporter):
    """Backward-compatible name for Sherlock's terminal reporter."""
