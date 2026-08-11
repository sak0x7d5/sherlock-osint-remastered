"""Terminal reporting for Sherlock scans and AI processing."""

from __future__ import annotations

import json
import sys
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
    from sherlock_project.profile_synthesis import IdentityAnchor, ProfileSynthesis


AIOutcome = Literal["with_facts", "no_facts", "pending", "skipped"]
BrowserStatus = Literal["installing", "starting", "ready"]
INTERRUPTION_MESSAGE = (
    "Processing interrupted; committed results were saved and pending AI "
    "work can resume on the next --ai run."
)


def _make_encoding_safe(stream: object) -> None:
    """Stop unencodable characters from truncating or mangling output.

    Redirected stdout on Windows encodes as cp1252 and profiles routinely
    carry names outside it, so writing one raised UnicodeEncodeError partway
    through the report: `show --profile > report.txt` died mid-file.

    Two different situations, two answers:

    - Redirected to a file or pipe, there is no console to please, so switch to
      UTF-8 and keep every character. Falling back to '?' here would quietly
      destroy exactly the non-Latin names an investigator wants to keep.
    - On a real terminal, leave the encoding alone -- it is chosen to match
      what the console can draw -- and only stop it from raising.

    Deliberately forgiving: injected test streams and anything without
    reconfigure() are left untouched.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    isatty = getattr(stream, "isatty", None)
    try:
        redirected = not (isatty is not None and isatty())
    except (ValueError, OSError):
        redirected = False

    try:
        if redirected:
            reconfigure(encoding="utf-8")
        else:
            reconfigure(errors="replace")
    except (ValueError, OSError, AttributeError):
        pass


def _format_sources(names: list[str], *, limit: int = 2) -> str:
    """Summarise which sites backed a value.

    The old form inlined every site's full URL per value, so one value with
    three sources wrapped across three lines and buried the value itself.
    """
    if not names:
        return ""
    noun = "site" if len(names) == 1 else "sites"
    shown = names[:limit]
    remainder = len(names) - len(shown)
    listed = ", ".join(shown)
    if remainder > 0:
        listed += f" +{remainder}"
    return f"{len(names)} {noun}: {listed}"


def _format_anchor(anchor: IdentityAnchor) -> str:
    """Render one anchor as the user typed it, plus what qualifies it.

    Trust is only shown when it is not the default, so the common case stays
    as short as the `--anchor field=value` the user actually wrote.
    """
    rendered = f"{anchor.field}={anchor.value}"
    if anchor.trust != "strong":
        rendered += f" [{anchor.trust}]"
    if anchor.source:
        rendered += f" (from {anchor.source})"
    return rendered


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

    def restored_results(
        self,
        *,
        username: str,
        results: dict,
        to_scan: int,
    ) -> None:
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
        if console is None:
            _make_encoding_safe(sys.stdout)
        if error_console is None:
            _make_encoding_safe(sys.stderr)
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
        self._scan_absent = 0
        self._scan_unknown = 0
        self._scan_waf = 0
        self._scan_illegal = 0
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

    def hint(self, message: str) -> None:
        """Indented dim follow-up to the event line above it.

        Carries the "so what do I do about it" half of a warning without
        spending a second marker, which would read as a second event.
        """
        self._write(Text(f"    {message}", style="dim"))

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

    def restored_results(
        self,
        *,
        username: str,
        results: dict,
        to_scan: int,
    ) -> None:
        """Show results carried over from earlier scans of this username.

        Runs before the scan starts, so these land above the live hits and the
        progress bar still counts only what is actually being checked now.
        """
        if not results:
            return

        claimed = [
            entry["status"]
            for entry in results.values()
            if entry["status"].status is QueryStatus.CLAIMED
        ]

        site_word = "site" if len(results) == 1 else "sites"
        if to_scan:
            self.info(
                f"{len(results)} {site_word} already checked for {username!r}; "
                f"{to_scan} left to check"
            )
        else:
            self.info(
                f"All {len(results)} {site_word} already checked for "
                f"{username!r}. Showing stored results; re-check them with "
                f"--fresh"
            )

        for result in sorted(claimed, key=lambda item: item.site_name.lower()):
            qualifier = ""
            if result.confidence is not None and str(result.confidence) != "Confirmed":
                qualifier = f" [{result.confidence}]"
            self.success(
                f"{result.site_name}: {result.site_url_user}{qualifier}",
                detail="stored",
            )
            if self.browse:
                webbrowser.open(result.site_url_user, 2)

    def start(self, message: str | None = None, total: int | None = None) -> None:
        if self._scan_task_id is not None:
            self.finish_scan()

        self._scan_username = message or ""
        self._scan_total = max(total or 0, 0)
        self._scan_completed = 0
        self._scan_found = 0
        self._scan_absent = 0
        self._scan_unknown = 0
        self._scan_waf = 0
        self._scan_illegal = 0
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
            self._scan_absent += 1
            if self.print_all:
                self.failure(f"{result.site_name}: not found", detail=response_time)
        elif result.status is QueryStatus.UNKNOWN:
            self._scan_unknown += 1
            if self.print_all:
                self.failure(
                    f"{result.site_name}: {result.context or 'unknown response'}",
                    detail=response_time,
                )
        elif result.status is QueryStatus.ILLEGAL:
            self._scan_illegal += 1
            if self.print_all:
                self.warning(f"{result.site_name}: illegal username format")
        elif result.status is QueryStatus.WAF:
            self._scan_waf += 1
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
        username = self._scan_username
        self.success(
            f"Scan complete for {username!r}: "
            f"{self._scan_found} found, {self._scan_absent} not found",
            detail=elapsed,
        )
        self._report_unresolved(username)
        self._scan_username = ""

    def _report_unresolved(self, username: str) -> None:
        """Warn when sites produced no answer, so silence is not read as absence.

        This is the whole reason the per-status counters exist. Reporting only
        found-vs-total lets "we could not determine this" reach the user as
        "they are not there", which is the one wrong conclusion an OSINT tool
        must not encourage. Sites that time out, answer ambiguously, or sit
        behind bot protection are unresolved, NOT absent.

        Deliberately a separate warning line rather than more numbers on the
        success line above: the marker and colour are what stop the count being
        skimmed as a statistic. Silent when everything resolved, so a clean run
        costs nothing.

        UNKNOWN and WAF stay separate because they ask for different things --
        retry or widen the timeout versus route around bot detection -- and
        collapsing them hides which one the user is facing.
        """
        unresolved = self._scan_unknown + self._scan_waf + self._scan_illegal
        if not unresolved:
            return

        parts: list[str] = []
        if self._scan_unknown:
            parts.append(f"{self._scan_unknown} inconclusive")
        if self._scan_waf:
            parts.append(f"{self._scan_waf} blocked by bot protection")
        if self._scan_illegal:
            parts.append(f"{self._scan_illegal} rejected the username format")

        site_word = "site" if unresolved == 1 else "sites"
        self.warning(
            f"{unresolved} {site_word} of {self._scan_completed} gave no "
            f"answer: {', '.join(parts)}"
        )
        self.hint(
            f'Not the same as "not found". List them: sherlock show '
            f"{username} --unresolved"
        )

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

    def render_profile(
        self,
        profile: ProfileSynthesis,
        *,
        show_sources: bool = False,
    ) -> None:
        """Render a synthesised profile.

        Presentation only: values keep the order synthesis produced and nothing
        is dropped. `show_sources` swaps the compact "3 sites: A, B +1" summary
        for the full site URLs.
        """
        site_names = {
            decision.site_id: decision.site_name
            for decision in profile.source_decisions
        }
        site_urls = {
            decision.site_id: (
                f"{decision.site_name} - {decision.site_url}"
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

        sections: list[tuple[str, dict[str, object], str | None]] = [
            ("CONFIDENT", profile.strong_profile, "green" if color_matches else None),
            ("UNSURE", profile.unsure_profile, "yellow" if color_matches else None),
        ]

        seen: set[tuple[str, str]] = set()
        rendered_sections: list[tuple[str, Table]] = []
        total_values = 0
        total_fields = 0

        for heading, values_by_field, style in sections:
            display_data: dict[str, list[tuple[str, list[str]]]] = {}
            for field_name, raw_values in values_by_field.items():
                values = raw_values if isinstance(raw_values, list) else [raw_values]
                for raw_value in values:
                    if not isinstance(raw_value, str):
                        continue
                    value = raw_value.strip()
                    key = (field_name, value)
                    if not value or key in seen:
                        continue
                    seen.add(key)
                    item = (
                        provenance.get((field_name, raw_value))
                        or provenance.get(key)
                    )
                    lookup = site_urls if show_sources else site_names
                    labels = (
                        [
                            lookup.get(site_id, f"site {site_id}")
                            for site_id in item.source_site_ids
                        ]
                        if item is not None
                        else []
                    )
                    display_data.setdefault(field_name, []).append(
                        (value, list(dict.fromkeys(labels)))
                    )

            if not display_data:
                continue

            table = self._profile_table()
            for field_name in sorted(display_data):
                entries = display_data[field_name]
                total_fields += 1
                total_values += len(entries)
                for index, (value, labels) in enumerate(entries):
                    # The field name is printed once per group so the eye has a
                    # single column to follow instead of a repeated label.
                    table.add_row(
                        Text(field_name if index == 0 else ""),
                        Text(value, style=style),
                        Text(
                            ", ".join(labels)
                            if show_sources
                            else _format_sources(labels)
                        ),
                    )
                table.add_section()
            rendered_sections.append((heading, table))

        blocks: list[RenderableType] = []

        if profile.anchors:
            # Text(), not a bare string: Rich parses markup in table cells and
            # would silently swallow the "[context]" trust marker as a style
            # tag. Anchor values are user data and may contain brackets too.
            anchors = self._profile_table()
            anchors.add_row(
                Text("anchored to"),
                Text(", ".join(_format_anchor(a) for a in profile.anchors)),
                Text(""),
            )
            blocks.append(anchors)

        for heading, table in rendered_sections:
            if blocks:
                blocks.append(Text(""))
            blocks.append(Text(heading, style="bold"))
            blocks.append(table)

        matching = self._matching_table(profile)
        if matching is not None:
            if blocks:
                blocks.append(Text(""))
            blocks.append(Text("MATCHING", style="bold"))
            blocks.append(matching)

        if not blocks:
            self.info("No profile facts were available")
            for warning in profile.warnings:
                self.warning(warning)
            return

        # Only two things earn space above the values: the caveat that changes
        # how they should be read, and an honest note that some sites are
        # missing. Everything else the synthesis records is diagnostic -- site
        # ids, token counts, validation codes -- and belongs behind --verbose.
        notes: list[Text] = []
        if profile.mode != "anchored" and rendered_sections:
            notes.append(
                Text(
                    "[!] No anchors used, so these values may describe "
                    "different people who share this username.",
                    style="yellow",
                )
            )
        if profile.warnings:
            if self.verbose:
                notes.extend(
                    Text(f"[!] {warning}", style="yellow")
                    for warning in profile.warnings
                )
            else:
                count = len(profile.warnings)
                noun = "note" if count == 1 else "notes"
                notes.append(
                    Text(
                        f"[!] {count} {noun} about how this was built; "
                        f"run with --verbose to read them.",
                        style="yellow",
                    )
                )
        if color_matches and rendered_sections:
            legend = Text("Match key: ", style="bold")
            legend.append("strong identity match", style="green")
            legend.append("  ")
            legend.append("unsure identity match", style="yellow")
            notes.append(legend)

        if notes:
            blocks = [*notes, Text(""), *blocks]

        self._write(
            Panel(
                Group(*blocks),
                title=f"Profile: {profile.username}",
                title_align="left",
                border_style="cyan" if not self.no_color else "none",
                padding=(0, 1),
            )
        )

    @staticmethod
    def _profile_table() -> Table:
        """Three aligned columns: field, value, where it came from."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True, min_width=14)
        table.add_column(overflow="fold")
        table.add_column(style="dim", overflow="fold")
        return table

    @staticmethod
    def _matching_table(profile: ProfileSynthesis) -> Table | None:
        """How each site scored against the anchors, for anchored runs."""
        if profile.mode != "anchored":
            return None

        source_groups = (
            ("strong matches", lambda d: d.identity_status == "strong_match"),
            ("unsure matches", lambda d: d.identity_status == "unsure"),
            ("excluded", lambda d: d.disposition == "excluded"),
            ("failed", lambda d: d.disposition == "failed"),
        )
        table = TerminalReporter._profile_table()
        populated = False
        for label, predicate in source_groups:
            sites = [
                decision.site_name
                for decision in profile.source_decisions
                if predicate(decision)
            ]
            if sites:
                populated = True
                table.add_row(
                    Text(label),
                    Text(", ".join(sites)),
                    Text(f"{len(sites)}"),
                )
        return table if populated else None

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
