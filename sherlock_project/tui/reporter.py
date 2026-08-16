"""The scan's output surface when the output surface is a TUI.

`sherlock()` reports through a `QueryNotify`, and `TerminalReporter` is one
implementation of it, not the only one allowed. That seam is why the scan loop
needs no changes to drive a full-screen app: the TUI supplies a different
reporter and the scan never learns the difference.

Two decisions here are load-bearing.

**This subclasses `TerminalReporter` rather than `QueryNotify`.** The bare base
class is a no-op stub with none of the methods the scan and the AI pipeline
actually call -- `info`, `warning`, `debug`, `ai_model_ready` and a dozen more
live only on the terminal implementation, so a reporter built on the stub dies
with `AttributeError` partway through a run, at whichever call site happens to
be reached first. Inheriting means every method exists by construction, and any
method added to the reporter later keeps working here without being ported.

**It counts nothing of its own.** `TerminalReporter` already tallies found,
absent, inconclusive, blocked and rejected, plus the AI job stats, and the TUI
reads those fields on each repaint. A second tally in this class would be a
second implementation of the same arithmetic, free to disagree with the CLI's
summary -- and the two are supposed to describe the same scan.

What it does override is where text goes. Every printed line in the reporter
funnels through `_write`, so intercepting that one method captures all of it;
the alternative is overriding every event method and missing the next one added.
"""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rich.console import Console, RenderableType
from rich.text import Text

from sherlock_project.notify import TerminalReporter
from sherlock_project.result import QueryResult, QueryStatus

# How many log lines to hold between one screen tick and the next. A bounded
# deque rather than a list because a long verbose run is otherwise an unbounded
# allocation for text nobody will scroll back far enough to read.
#
# This is a HAND-OFF buffer, not the scrollback: drawn lines live in the pane's
# RichLog, which keeps its own history. The bound only has to survive one tick,
# and in verbose mode a single burst is much larger than it looks -- one cached
# extraction is a JSON panel of dozens of lines, and `ai_cached_evidence` emits
# one per stored site before the scan even starts. At 500 a resumed username
# with a hundred extractions overflowed this on the first tick, which is what
# froze the log (see `ScanPane._append_log` for the other half of that bug).
LOG_LIMIT = 4000

# How many findings to keep in memory before the oldest are dropped. Generous,
# because these are the results the operator came for -- unlike log lines, they
# are not disposable -- but still bounded so a pathological manifest cannot
# exhaust memory mid-scan.
FEED_LIMIT = 5000


@dataclass(frozen=True, slots=True)
class Finding:
    """One scan result worth showing as its own row.

    A flat record rather than the `QueryResult` itself, so the render path
    cannot accidentally reach back into live scan objects while the scan is
    still mutating them.
    """

    site_name: str
    url: str
    status: QueryStatus
    detail: str
    # How long the site took to answer, in seconds, or None for a row restored
    # from the database -- stored results carry no timing. This is the one thing
    # the per-hit log line said that the table did not, which is why the log
    # line is gone and this field exists.
    elapsed: float | None = None


@dataclass(frozen=True, slots=True)
class Phase:
    """One slow startup step and how far along it is.

    `elapsed` keeps growing while the step runs and freezes when it lands, so
    one field answers both "how long has this been going" and "how long did it
    take" without the screen having to know which question is being asked.
    """

    label: str
    state: str
    elapsed: float


class TuiReporter(TerminalReporter):
    """Feeds a Textual screen instead of a terminal.

    Nothing here touches a widget. Every callback appends to a buffer and
    returns immediately; the screen drains those buffers on its own timer. That
    split is what keeps a fast scan from turning into a repaint storm -- see
    `ScanPane._flush` for the other half of the argument.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        # Consoles pointed at a throwaway buffer, for two reasons. The obvious
        # one is that anything written to the real stdout would land on top of
        # the Textual canvas and corrupt it. The subtle one is that passing
        # consoles at all is what stops the parent constructor calling
        # `_make_encoding_safe(sys.stdout)` -- reconfiguring the stream while
        # Textual owns it is not something to do for a stream nothing prints to.
        #
        # `no_color=True` additionally leaves `interactive` False, which keeps
        # the parent's Rich `Progress` disabled. Two progress renderers fighting
        # over one terminal is the corruption this avoids.
        self._sink = io.StringIO()
        sink_console = Console(
            file=self._sink,
            no_color=True,
            color_system=None,
            highlight=False,
            width=200,
        )
        super().__init__(
            verbose=verbose,
            no_color=True,
            console=sink_console,
            error_console=sink_console,
        )
        self.log_lines: deque[Text] = deque(maxlen=LOG_LIMIT)
        # Every line ever produced, including those the deque has since dropped.
        # The screen tracks its position against THIS, not against the length of
        # the buffer: once the deque is full its length stops changing, so a
        # reader comparing lengths concludes nothing new has arrived and stops
        # drawing for the rest of the run.
        self.log_total = 0
        self.findings: deque[Finding] = deque(maxlen=FEED_LIMIT)
        # Findings not yet drawn. The screen drains this; `findings` keeps the
        # full history for anything that needs to redraw from scratch.
        self.pending: deque[Finding] = deque()
        # Step name -> how long it took, frozen at the moment it landed.
        self._phase_done: dict[str, float] = {}
        self.username = ""
        self.total = 0
        self.finished = False
        self.interrupted = False
        # Results skipped by the resume filter, kept so they can be replayed
        # into the tally after `start` zeroes it.
        self._restored: list[QueryResult] = []
        # Replaying stored results into the tally; their per-hit lines are a
        # transcript of a scan that already happened.
        self._replaying = False
        # Dropping narration a panel already carries -- see `_quiet`.
        self._suppressing = False

    # -- output interception ------------------------------------------------

    @contextmanager
    def _quiet(self) -> Iterator[None]:
        """Drop narration that a panel on screen already carries.

        The reporter was written for a terminal whose only output is a stream of
        lines, so it narrates everything: one line per hit, one per startup
        step, one summary per scan. On this screen each of those facts already
        has a panel -- hits are the findings table, startup is the phase block,
        counts are the SITES tallies -- and printing them again put the same
        information twice on one screen, side by side, in two different formats.

        So the log keeps only what has nowhere else to go. That is what makes it
        worth reading: a channel that repeats the screen is a channel nobody
        checks when something actually goes wrong.

        APPLIES IN VERBOSE MODE TOO, reversing the earlier decision that verbose
        should defeat it. The argument then was that the per-job lines are "most
        of what there is to watch"; running it showed the opposite. What verbose
        actually adds is not routed through here at all -- request traces with
        their token counts and timings, failure diagnostics, cached extractions,
        `debug` lines -- and none of that is suppressed. What verbose was adding
        through here was the findings table rewritten as prose, one hit per line,
        beside the findings table. The single fact those lines carried that the
        table did not was the response time, and that is a column now.

        Kept separate from `_replaying`, which suppresses the replay of restored
        results. The two look alike and are not: one is de-duplication of what
        is on screen right now, the other is a few hundred lines of a scan that
        already happened.
        """
        previous = self._suppressing
        self._suppressing = True
        try:
            yield
        finally:
            self._suppressing = previous

    def _write(self, message: RenderableType, *, error: bool = False) -> None:
        """Capture a line instead of printing it.

        Silent while restored rows are being replayed. The parent logs a line
        per hit, and several hundred of them would bury the run's actual
        narration under a transcript of a scan that already happened -- the
        restored summary above them already gives the count.

        Kept as `Text` rather than a plain string: the reporter builds styled
        renderables, and flattening them here would throw away the colour that
        distinguishes a warning from a note in the log pane.
        """
        if self._replaying or self._suppressing:
            return
        if isinstance(message, Text):
            self._log(message)
        else:
            # Render through the sink console, then recover the plain text. A
            # renderable that is not already Text (a table, a panel) has no
            # meaningful one-line form otherwise, and dropping it would lose an
            # error someone needs.
            self._sink.seek(0)
            self._sink.truncate(0)
            self.console.print(message)
            for line in self._sink.getvalue().splitlines():
                if line.strip():
                    self._log(Text(line, style="dim" if error else ""))

    def _log(self, line: Text) -> None:
        """Buffer one line and count it.

        The count is the only reliable position marker once the deque is full,
        because from then on appending drops a line from the far end and the
        length never changes again.
        """
        self.log_lines.append(line)
        self.log_total += 1

    # -- scan lifecycle -----------------------------------------------------

    def restored_results(
        self,
        *,
        username: str,
        results: dict,
        to_scan: int,
    ) -> None:
        """Take the skipped sites into the tally, not just into the log.

        This screen must describe the USERNAME, not this particular run -- the
        same invariant the CLI keeps by merging restored rows into `results`
        before it reports. Without it a resumed scan of a username with 260
        stored accounts showed `found 0` and an empty findings table, because
        only sites checked in this run ever reached the counters. That reads as
        "scanned everything, found nothing", which is the opposite of the truth.
        """
        # NOT super(). The parent writes one log line per restored hit, which
        # for a fully resumed username is several hundred lines burying the
        # run's own narration -- and those hits belong in the findings table,
        # which is where the replay below puts them. Its summary also advises
        # `--fresh`, a flag nobody in this app can type.
        self._restored = [
            entry["status"]
            for entry in results.values()
            if isinstance(entry.get("status"), QueryResult)
        ]
        if results:
            # "results", not "sites". Restored rows can outnumber the current
            # manifest -- a username scanned across manifest versions keeps
            # rows for sites that have since been removed -- and "all 1016
            # sites" beside a 680-site manifest reads as a miscount rather than
            # as history.
            word = "result" if len(results) == 1 else "results"
            if to_scan:
                self.info(
                    f"{len(results)} {word} already stored for {username!r}; "
                    f"{to_scan} left to check"
                )
            else:
                self.info(
                    f"{len(results)} {word} already stored for {username!r} — "
                    f"nothing left to check; turn re-scan on to check again"
                )
        self._replay_restored()

    def _replay_restored(self) -> None:
        """Feed the stored rows through the normal counting path.

        Through `update` rather than by adding numbers, so restored and live
        results are tallied by exactly one piece of arithmetic. The flag marks
        them on the way past: they are evidence recalled, not evidence found
        just now, and the feed says so.
        """
        self._replaying = True
        try:
            for result in self._restored:
                self.update(result)
        finally:
            self._replaying = False

    def start(self, message: str | None = None, total: int | None = None) -> None:
        # `start` zeroes every counter, and the scan calls it AFTER the restored
        # rows have been reported -- so they have to be replayed on the other
        # side of it or they are silently discarded.
        #
        # Quiet: "Checking username 'x' across N sites" is the feed title and
        # the progress strip, both of which say it while it is still true.
        with self._quiet():
            super().start(message, total)
        self.username = message or ""
        self.total = (total or 0) + len(self._restored)
        self.finished = False
        self._replay_restored()

    def update(self, result: QueryResult) -> None:
        """One site finished.

        Deliberately the cheapest method in this file: it runs once per site,
        several hundred times, sometimes many per second. All it does is let the
        parent do the counting and append a record if the result is worth a row.
        Anything more expensive here is paid for on every site.
        """
        # Quiet: the parent logs a line per hit, which is the findings table
        # rewritten in a second format directly beside the findings table.
        with self._quiet():
            super().update(result)
        # EVERY result is retained, including absent ones. It used to drop
        # anything the feed did not show by default, which made the choice
        # irreversible: the rows were gone, so no filter could ever bring them
        # back. Retention is cheap (bounded by FEED_LIMIT) and the pane decides
        # what to draw.
        #
        # `context` carries why a result came out the way it did -- the timeout,
        # the block reason. It is the difference between "inconclusive" and
        # "inconclusive because the request timed out", and it is already
        # computed, so showing it costs nothing.
        detail = result.context or ""
        if result.confidence and str(result.confidence) != "Confirmed":
            detail = f"{result.confidence}{f'; {detail}' if detail else ''}"
        if self._replaying:
            # Marked, because evidence recalled from disk and evidence found a
            # moment ago are not the same claim about the world -- and an
            # unmarked restored row makes a resumed scan look like it just
            # re-checked every site.
            detail = f"stored{f'; {detail}' if detail else ''}"
        finding = Finding(
            site_name=str(result.site_name),
            url=str(result.site_url_user or ""),
            status=result.status,
            detail=str(detail),
            # None on a replayed row: the database stores the verdict, not how
            # long the request took, and a fabricated 0 there would read as an
            # instant answer rather than as an unknown one.
            elapsed=None if self._replaying else result.query_time,
        )
        self.findings.append(finding)
        self.pending.append(finding)

    def _redo_extractions_hint(self, username: str) -> str:
        """The in-app route to re-extracting with the newly chosen model.

        Switching model deliberately leaves stored extractions alone -- the
        Pass 1 cache is keyed on the prompt contract, not the model, and an
        extraction made by another model is still valid against that contract.
        The consequence is that changing model appears to do nothing until
        something re-extracts, so this line is the only thing standing between
        the user and that conclusion. It has to name a control they have.
        """
        return (
            "Scan again and choose 'Re-scan all' to redo them with the model "
            "you just selected."
        )

    def render_profile(self, profile: Any, **kwargs: Any) -> None:
        """Do not draw the profile into the scan log.

        The CLI renders the whole panel here because the terminal scrollback is
        the only place it has. This app has a better one: the RESULTS tab draws
        the same profile, through the same renderer, in a pane wide enough for
        its three columns and with a key for the diagnostic notes.

        Two things went wrong when this inherited the CLI behaviour. The panel
        was laid out against this reporter's fixed 200-column sink and then
        written into an activity log a third that wide, so it arrived clipped.
        And it carried "run with --verbose to read them" -- advice naming a
        flag nobody inside a running app can type.

        So this logs one line and points at where the profile actually is.
        """
        self.success(
            f"AI profile ready for {profile.username!r}",
            detail="see the RESULTS tab; press v there for the notes",
        )

    # -- startup phases -----------------------------------------------------

    # A finished step's duration has to be CAPTURED, not recomputed. The parent
    # keeps only the start time, so asking it how long the browser took goes on
    # answering "however long ago it started" -- on screen that read
    # `ready <1s`, then `ready 1s`, then `ready 2s`, climbing forever and
    # eventually claiming a step took minutes when it took under a second.

    def browser_status(self, status: Any) -> None:
        # Quiet: the startup block shows this step live, with its own timer.
        with self._quiet():
            super().browser_status(status)
        if status == "ready" and "browser" not in self._phase_done:
            self._phase_done["browser"] = self._elapsed_since(
                self._web_scanner_started_at
            )

    # Per-job AI narration. One line each per extracted site, which on a
    # username with 260 hits is 260 lines saying what the ANALYSIS block says
    # in three numbers. Failures are NOT suppressed -- `ai_failed` is the one
    # AI event with no counter of its own, and it is the reason to look here.

    def ai_scheduled(self) -> None:
        with self._quiet():
            super().ai_scheduled()

    def ai_job_started(self, site_name: str) -> None:
        with self._quiet():
            super().ai_job_started(site_name)

    def ai_job_finished(self, *args: Any, **kwargs: Any) -> None:
        with self._quiet():
            super().ai_job_finished(*args, **kwargs)

    def ai_model_starting(self) -> None:
        # Quiet: the startup block shows the model loading, and how long for.
        with self._quiet():
            super().ai_model_starting()

    def ai_model_ready(self) -> None:
        already = self._ai_model_finished
        with self._quiet():
            super().ai_model_ready()
        if not already:
            self._phase_done["model"] = self._elapsed_since(
                self._ai_model_started_at
            )

    def ai_model_failed(self, error: Exception) -> None:
        already = self._ai_model_finished
        super().ai_model_failed(error)
        if not already:
            self._phase_done["model"] = self._elapsed_since(
                self._ai_model_started_at
            )

    def finish_scan(self, elapsed_time: float = 0) -> None:
        """End the scan, keeping only the part the panels do not already say.

        The parent's summary -- "2 found, 1 not found" -- is the SITES counters
        restated as a sentence. What it says that they do not is the CAVEAT: a
        site that never answered is not a site where nobody was home, and that
        distinction is the one this whole tool refuses to blur. So the counts
        are dropped and the caveat is kept, reworded for a surface that has an
        UNRESOLVED tab instead of a `--unresolved` flag to recommend.
        """
        with self._quiet():
            super().finish_scan(elapsed_time)
        self.finished = True

        unresolved = self._scan_unknown + self._scan_waf + self._scan_illegal
        if not unresolved:
            return
        reasons = [
            (self._scan_unknown, "inconclusive"),
            (self._scan_waf, "blocked by bot protection"),
            (self._scan_illegal, "rejected the username format"),
        ]
        breakdown = ", ".join(
            f"{count} {label}" for count, label in reasons if count
        )
        site_word = "site" if unresolved == 1 else "sites"
        self.warning(
            f"{unresolved} {site_word} gave no answer: {breakdown}. "
            f'Not the same as "not found" — see the UNRESOLVED tab under '
            f"RESULTS."
        )

    def processing_interrupted(self) -> None:
        super().processing_interrupted()
        self.interrupted = True
        self.finished = True

    # -- what the screen reads ---------------------------------------------

    @property
    def counts(self) -> dict[QueryStatus, int]:
        """The parent's own tallies, named.

        Reaching into the parent's private counters is the point: they are the
        numbers the CLI prints, so the TUI showing anything else would mean one
        of the two is wrong about the same scan.
        """
        return {
            QueryStatus.CLAIMED: self._scan_found,
            QueryStatus.AVAILABLE: self._scan_absent,
            QueryStatus.UNKNOWN: self._scan_unknown,
            QueryStatus.WAF: self._scan_waf,
            QueryStatus.ILLEGAL: self._scan_illegal,
        }

    @property
    def completed(self) -> int:
        return self._scan_completed

    @property
    def phases(self) -> list[Phase]:
        """The slow startup steps, and where each one has got to.

        Read off the parent's own tracking for the same reason `counts` is: the
        reporter already records when the browser and the model started and
        whether they finished, and a second copy here could disagree with the
        CLI about the same run.

        A step only appears once it has announced itself. The browser reports
        nothing at all when the fast transport is in use, and the model reports
        nothing when analysis was not asked for, so the block shows exactly the
        steps this particular run is actually waiting on rather than a fixed
        list with permanent blanks in it.
        """
        found: list[Phase] = []

        if self._web_scanner_status is not None:
            if self._web_scanner_finished:
                found.append(
                    Phase(
                        "browser",
                        "ready",
                        self._phase_done.get("browser", 0.0),
                    )
                )
            else:
                # "installing" is worth saying out loud rather than folding into
                # loading: a first run downloads a browser, which is minutes, and
                # a user who thinks a scan hung will kill it.
                word = (
                    "installing"
                    if self._web_scanner_status == "installing"
                    else "loading"
                )
                found.append(
                    Phase(
                        "browser",
                        word,
                        self._elapsed_since(self._web_scanner_started_at),
                    )
                )

        if self._ai_model_started_at is not None:
            if self._ai_model_unavailable:
                state = "failed"
            elif self._ai_model_finished:
                state = "ready"
            else:
                state = "loading"
            # Live while it runs, frozen once it has landed.
            elapsed = self._phase_done.get(
                "model", self._elapsed_since(self._ai_model_started_at)
            )
            found.append(Phase("model", state, elapsed))
        return found

    def drain_pending(self) -> list[Finding]:
        """Hand over the findings not yet drawn, and forget them.

        Drained rather than read so the screen appends only what is new. A
        redraw that rebuilt the whole table from `findings` would be O(n) on
        every tick, which is the cost this whole design exists to avoid.
        """
        drained = list(self.pending)
        self.pending.clear()
        return drained

    def snapshot_log(self) -> list[Text]:
        return list(self.log_lines)


def blank_counts() -> dict[QueryStatus, int]:
    """Zeroed tallies, for a screen that has not run a scan yet.

    The panel is drawn with zeros rather than hidden until a scan starts: an
    empty frame shows what is about to be measured, while a panel that appears
    on first result makes the layout jump at the least welcome moment.
    """
    return {status: 0 for status in QueryStatus}


def analysis_is_on(values: dict[str, Any]) -> bool:
    """Whether a model is configured, and so whether AI analysis will run.

    `ai.model` is the one setting with no default -- which model is right
    depends on what the user has downloaded -- so its absence is exactly the
    signal that analysis is off. `runner.ai_is_configured` is the same predicate
    for the scan itself; a test asserts the two agree, because a config line
    that promises analysis the scan will not run is worse than saying nothing.
    """
    return bool(values.get("ai.model"))


def describe_settings(values: dict[str, Any]) -> str:
    """The one-line summary of how this scan will run.

    Shown next to the target field because these are the settings that change
    what a result *means* -- a browserless run reports real accounts as absent,
    and an operator reading a finding hours later needs that visible at the
    moment of reading, not buried in a config file.

    This line describes STORED settings only. Whether analysis runs is a
    per-run choice shown on its own toggle beside it, so stating it here too
    would give one answer two places to be read from and one of them would go
    stale. What does belong here is the absence of a model, because that is
    stored state, and it is the reason the analysis toggle cannot help.
    """
    parts = []
    parts.append("no browser" if values.get("scan.webbrowser") is False else "browser")
    parts.append(f"{values.get('scan.concurrency', '?')} at a time")
    parts.append(f"{values.get('scan.timeout', '?')}s timeout")
    if not analysis_is_on(values):
        parts.append("no model configured")
    return "  ·  ".join(parts)
