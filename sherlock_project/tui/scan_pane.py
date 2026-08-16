"""The live scan pane: target on top, tallies left, findings right.

The layout answers three questions in the order an operator asks them. *Who am
I scanning* is the field you type in, so it is first and it is the only thing on
its row. *How is it going* is a block of counts that must be readable at a
glance from across a desk, so it is a fixed narrow column with the numbers
right-aligned. *What did we find* is the widest and longest-lived thing on
screen, so it gets every remaining cell.

**The counters are the filter.** Every result is kept, but the feed draws only
hits until told otherwise: on a 680-site manifest the overwhelming majority of
results are "not found", and a feed showing them scrolls the handful of actual
findings off screen within seconds. Clicking a counter row shows or hides that
kind, and a row whose results are hidden is struck through -- so the count stays
exact and readable while saying plainly that the rows behind it are not on
screen. What the panel counts and what the feed displays are two different
facts, and only the second one is being toggled.

**Why the scan does not draw anything itself.** Results arrive in bursts, many
per second, and a widget update per result means a repaint per result. Textual
coalesces some of that, but the table work -- building cells, remeasuring
columns, scrolling -- is per call and lands on the same event loop the scan is
running on, so the scan slows itself down by reporting. Instead the reporter
appends to a buffer, and a timer here drains it at a fixed rate. The scan's cost
to report is an append; the screen's cost to draw is one pass regardless of how
many results arrived since the last one. That is the whole reason this stays
smooth at 680 sites.
"""

from __future__ import annotations

import webbrowser
from typing import Any, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Input, RichLog, Static
from textual.worker import Worker, WorkerState

from sherlock_project.profile_synthesis import IdentityAnchor
from sherlock_project.result import QueryStatus
from sherlock_project.tui.anchor_screen import AnchorScreen
from sherlock_project.tui.reporter import (
    Finding,
    TuiReporter,
    analysis_is_on,
    blank_counts,
    describe_settings,
)
from sherlock_project.tui.theme import (
    REDRAW_INTERVAL,
    STATUS_ORDER,
    anchor_line,
    default_visible_statuses,
    elapsed_label,
    phase_line,
    progress_bar,
    response_label,
    stat_row,
    status_cell,
    status_style,
)

# Feed column widths. Fixed, because the point of a table is that column two
# starts at the same cell on every row; `auto` width re-measures as rows arrive
# and the whole table shifts sideways mid-scan, which is unreadable while it is
# happening and is the single most common way a live table looks broken.
STATUS_WIDTH = 14
SITE_WIDTH = 24
# How long the site took to answer. Narrow, right-aligned, and BEFORE the detail
# column so the URL keeps taking everything that is left.
#
# It exists because the per-hit log line was deleted: verbose used to narrate
# every finding into ACTIVITY beside the table already showing it, and the one
# thing that line said which the table did not was this number. So it moved here
# rather than being lost -- and it is drawn on every run, not only verbose ones,
# because a table whose columns change shape depending on a toggle is harder to
# read than one extra column.
#
# Six cells is the widest value this can hold: under a second reads `999ms` and
# anything above it collapses to `12.3s`. Wider would be spent on padding, and
# every cell here comes out of the URL.
TIME_WIDTH = 6

# How many lines the ACTIVITY pane keeps to scroll back through. Generous, but
# finite: this is the drawn history, unlike the reporter's LOG_LIMIT, which is
# only the hand-off buffer between the scan and one screen tick.
ACTIVITY_SCROLLBACK = 10000

# How many anchors the scan pane lists before collapsing the rest into "+N
# more". Capped so the block's height cannot grow with the list and push the
# counters below it down the column -- the full list lives in the editor, which
# is where you go to change them anyway.
ANCHOR_PREVIEW = 3
# Cells available to one anchor line inside the 26-wide counters column.
ANCHOR_LINE_WIDTH = 24


class CounterRow(Static):
    """One line of the SITES panel, and the control that filters on it.

    The counters were already the summary of what the scan found; making them
    the filter means the panel answers "how many" and "show me which" in one
    place, instead of growing a second list of the same five words somewhere
    else. Clicking a row shows or hides that kind of result in the feed.

    Focusable so the filter is reachable without a mouse. There is no spare key
    for five toggles -- the scan pane's text field claims the letters and digits,
    and alt+1..3 are the tabs -- so Tab-and-Enter is the keyboard route, with
    `alt+f` beside it for the one-press "everything / hits only" flip.
    """

    can_focus = True

    class Toggled(Message):
        def __init__(self, status: QueryStatus) -> None:
            super().__init__()
            self.status = status

    def __init__(self, status: QueryStatus) -> None:
        super().__init__(id=f"count-{status.name.lower()}", classes="counter-row")
        self._status = status

    def on_click(self) -> None:
        self.post_message(self.Toggled(self._status))

    def key_enter(self) -> None:
        self.post_message(self.Toggled(self._status))

    def key_space(self) -> None:
        self.post_message(self.Toggled(self._status))


class ScanPane(Vertical):
    """Everything for running one scan and watching it happen."""

    # Keys this pane's own Input does not already claim. `Input` binds a lot of
    # them -- ctrl+a/c/d/e/k/u/v/w/x are all editing commands -- and a binding
    # here that collides is simply swallowed, because the field has focus from
    # the moment the app opens. STOP was ctrl+x, which `Input` uses for cut, so
    # the key did nothing at all while a scan was running. Check `Input.BINDINGS`
    # before adding one.
    BINDINGS: ClassVar = [
        Binding("ctrl+r", "run", "run scan"),
        Binding("escape", "stop", "stop"),
        Binding("alt+a", "edit_anchors", "anchors"),
        Binding("alt+f", "toggle_all_statuses", "filter"),
    ]

    def __init__(self, settings_values: dict[str, Any]) -> None:
        super().__init__()
        self._settings_values = settings_values
        self._reporter: TuiReporter | None = None
        # NOT `_running`. Textual's own MessagePump keeps an attribute by that
        # name and sets it True when the widget mounts, so a field called
        # `_running` on a widget is already True before anything has run --
        # `action_run` read it, believed a scan was in progress, and returned
        # immediately every single time. Pressing SCAN or Enter did nothing at
        # all, with no error, because the guard was doing its job on the wrong
        # value. Any state added here needs a name of its own for the same
        # reason; a widget subclass shares a namespace with the framework.
        self._scan_running = False
        self._worker: Worker | None = None
        # Off every time the app opens, matching `--ai`, which is also off
        # unless typed. There is no stored `_fresh`: re-scanning is chosen per
        # scan, in the dialog that appears when the username already has
        # results, rather than pre-armed on a toggle.
        self._use_ai = False
        # Seeded from the stored `output.verbose`, overridable for one run --
        # the flag > config shape `-v` already has on the CLI.
        self._verbose = bool(settings_values.get("output.verbose"))
        # Run-only, exactly like `--anchor`: never stored, gone next launch.
        self._anchors: list[IdentityAnchor] = []
        self._elapsed = 0.0
        self._log_drawn = 0
        self._tick = 0
        # Which kinds of result the feed draws. Hits only, until someone clicks
        # a counter -- see StatusStyle.default_visible for why.
        self._visible: set[QueryStatus] = default_visible_statuses()
        self._scan_username = ""
        # Row key -> the URL that row is about, so Enter can open it.
        self._feed_urls: dict[str, str] = {}
        # Set when a scan is running so the timer knows whether the elapsed
        # clock should still be advancing. Kept separate from `_running` because
        # a finished scan must keep its final time on screen, not reset to zero.
        self._started_at: float | None = None

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Grid(id="target-row"):
            yield Static("TARGET", id="target-label")
            yield Input(
                placeholder="username to look for",
                id="target-input",
            )
            yield Button("SCAN", variant="primary", id="scan-button")

        # Per-run choices, deliberately NOT settings. They are this screen's
        # equivalent of `--ai` and `--fresh`: both start off on every run and
        # neither is stored, so the UI cannot quietly do something the same
        # command on the CLI would not.
        #
        # Buttons rather than a key binding each. The username field has focus
        # from the moment the app opens and `Input` swallows most letter and
        # ctrl combinations, so a keyboard-only toggle would either collide or
        # need an obscure key nobody would find. A button is clickable, reachable
        # with Tab, activated with Enter, and -- the part that matters -- it
        # shows its own state without anyone having to look for it.
        with Grid(id="options-row"):
            yield Static("OPTIONS", id="options-label")
            yield Button(id="toggle-ai", classes="toggle")
            # Verbose is a stored setting (`output.verbose`) AND a per-run
            # override, which is the flag > config shape the CLI already has for
            # `-v`. Analysis and re-scan have no config layer because `--ai` and
            # `--fresh` have none either; this one does, so it gets one.
            yield Button(id="toggle-verbose", classes="toggle")
            # Anchors are NOT a button here. `‹ ›` means "press to step through
            # values" -- that is what it means on the settings screen and on the
            # two toggles beside this. A count in those brackets promises a
            # value you can cycle and delivers a dialog, which is a broken
            # affordance however tidy it looks. They have their own section in
            # the left column, where the actual anchors can be shown rather than
            # just counted.
            # No `hint` class here: it carries bottom padding, and padding is
            # drawn inside a fixed height, so on a one-row cell it left zero
            # rows for the text and the line simply vanished.
            yield Static(id="scan-config")

        with Grid(id="scan-body"):
            with Vertical(id="counters-col"):
                # Startup first, because it is the only thing happening for the
                # first stretch of a run -- a model can take minutes to load and
                # a screen of zeroes with no explanation is what makes someone
                # kill a scan that was working.
                yield Static(id="startup")

                # Anchors, above the counters because they are set BEFORE a
                # scan and read while one runs. Shown only when analysis is on:
                # they feed the AI's second pass and nothing else, so with
                # analysis off this is a control that cannot do anything.
                # Revealing it with the toggle that makes it matter is what
                # keeps that from costing discoverability -- and the empty
                # state says what anchors are for rather than just sitting
                # blank.
                with Vertical(id="anchors-block"):
                    with Grid(id="anchors-head"):
                        yield Static("ANCHORS", id="anchors-title")
                        yield Button("+", id="add-anchor")
                    yield Static(id="anchors-list")

                yield Static("SITES", id="counters-title")
                # One widget per row rather than one Static holding five lines:
                # a single Static cannot tell which of its lines was clicked.
                with Vertical(id="counters"):
                    for status in STATUS_ORDER:
                        yield CounterRow(status)
                with Vertical(id="ai-block"):
                    yield Static("ANALYSIS", id="ai-title", classes="dim")
                    yield Static(id="ai-counters")
            with Vertical(id="feed-col"):
                yield Static(id="feed-title")
                # Directly under the heading, where the eye already is during a
                # scan. It used to sit at the very bottom of the pane, below the
                # activity log and hard against the keybinding footer -- which
                # reads as chrome rather than as live state, and was missed
                # entirely by the person watching the scan it was reporting on.
                yield Static(id="progress-strip")
                yield DataTable(id="feed", cursor_type="row")
                # What has no panel of its own: warnings, failures, and the
                # reasons behind them. Everything with a home elsewhere is
                # suppressed rather than repeated -- hits are the table above,
                # startup is the phase block, totals are the counters -- because
                # a log that restates the screen is a log nobody reads when
                # something actually goes wrong. A 6-site scan wrote 12 lines
                # here before that rule; it writes 3 now, and all 3 say
                # something no panel does.
                yield Static("ACTIVITY", id="activity-title")
                # Bounded scrollback. A verbose run prints a panel per AI
                # request, so an unbounded log is tens of thousands of styled
                # segments held for a session nobody scrolls that far back
                # through -- and the widget re-measures what it holds.
                yield RichLog(
                    id="activity",
                    markup=False,
                    wrap=False,
                    max_lines=ACTIVITY_SCROLLBACK,
                )

    def on_mount(self) -> None:
        table = self.query_one("#feed", DataTable)
        # Width on the first two columns, none on the third: the URL takes what
        # is left. Given a width it would either truncate short URLs for no
        # reason or overflow the pane on long ones.
        table.add_column("", key="status", width=STATUS_WIDTH)
        table.add_column("site", key="site", width=SITE_WIDTH)
        table.add_column("time", key="time", width=TIME_WIDTH)
        table.add_column("detail", key="detail")

        self._redraw_options()
        self._show_activity_placeholder()
        self._redraw_counters(blank_counts())
        self._redraw_progress(0, 0)
        self._redraw_feed_title()
        # The drain timer. One timer for the whole pane rather than one per
        # widget, so everything on screen describes the same instant -- counters
        # and feed updated a tick apart is how a total stops matching the rows
        # above it.
        self.set_interval(REDRAW_INTERVAL, self._flush)

    def prepare_analysis_scan(self, username: str) -> None:
        """Set up a scan that will collect AI evidence, without starting it.

        Sent here when the results pane could not build a profile because
        nothing was extracted. It fills the field and turns analysis on, then
        stops: starting a 680-site scan because someone pressed a button on
        another tab would be doing considerably more than was asked.
        """
        self.query_one("#target-input", Input).value = username
        self._use_ai = True
        self._redraw_options()
        self.query_one("#target-input", Input).focus()
        self.notify(
            f"Analysis is on — press SCAN to collect evidence for {username}.",
        )

    def refresh_settings(self, values: dict[str, Any]) -> None:
        """Adopt settings edited elsewhere in this session.

        The config line under the target field is the only place the screen
        states how the next scan will run, so it has to follow a change made on
        the settings tab. Left stale it would claim a browser scan while the
        next run went browserless -- the one difference that changes whether a
        result can be trusted.
        """
        self._settings_values = values
        self._redraw_options()

    def _redraw_options(self) -> None:
        """Draw the two per-run toggles and the stored-settings line.

        The toggles carry the same ‹ › brackets the settings editor uses for
        anything that answers to a keypress, so one visual language means one
        thing across the whole app.
        """
        self.query_one("#toggle-ai", Button).label = (
            f"analysis ‹ {'on' if self._use_ai else 'off'} ›"
        )
        self.query_one("#toggle-verbose", Button).label = (
            f"verbose ‹ {'on' if self._verbose else 'off'} ›"
        )
        self._redraw_anchors()
        self.query_one("#scan-config", Static).update(
            Text(describe_settings(self._settings_values), style="dim")
        )

    @on(Button.Pressed, "#toggle-ai")
    def _toggle_ai(self) -> None:
        self._use_ai = not self._use_ai
        # Said at the moment of asking rather than at the end of the scan. The
        # run would otherwise look like the model failed, when the real problem
        # is one unset setting two tabs away.
        if self._use_ai and not analysis_is_on(self._settings_values):
            self.notify(
                "No model is configured — set one on the SETTINGS tab.",
                severity="warning",
            )
        self._redraw_options()

    @on(Button.Pressed, "#toggle-verbose")
    def _toggle_verbose(self) -> None:
        """Show what the model is actually doing.

        Off, ACTIVITY carries only what has no panel of its own. On, it carries
        everything the reporter narrates -- per-site extraction progress, the
        token counts and timings of each request, and the DIAGNOSTICS behind a
        failure: stop reason, predicted against maximum tokens, and which
        validation step rejected the reply.

        That last part is the difference between "extraction failed" and "the
        model ran out of tokens before it finished the JSON", which is the
        answer when a smaller model starts failing where a larger one did not.
        Every bit of it already existed and was being thrown away, because this
        reporter was always built with verbose off.

        Takes effect on the NEXT scan: the reporter is constructed when a scan
        starts, and its verbosity is fixed for that run.
        """
        self._verbose = not self._verbose
        self._redraw_options()
        if self._verbose and self._scan_running:
            self.notify("Verbose applies from the next scan.")

    def _redraw_anchors(self) -> None:
        """Draw the anchor block, or take it off the screen entirely.

        Capped at ANCHOR_PREVIEW rows with a "+N more" tail. Uncapped, a long
        list would push SITES and ANALYSIS down the column as it grew, so the
        counters someone is watching would sit somewhere different depending on
        how much setup they had done. The full list is in the editor, which is
        where you go to act on them anyway.
        """
        block = self.query_one("#anchors-block", Vertical)
        block.display = self._use_ai
        if not self._use_ai:
            return

        listing = self.query_one("#anchors-list", Static)
        if not self._anchors:
            # Teaches rather than sits blank. This is the only place the app
            # explains why anchors matter, and an empty section is exactly when
            # someone is deciding whether to bother.
            listing.update(
                Text(
                    "none — results may\nmix different people",
                    style="dim italic",
                )
            )
            return

        text = Text()
        for index, anchor in enumerate(self._anchors[:ANCHOR_PREVIEW]):
            if index:
                text.append("\n")
            text.append_text(anchor_line(anchor, width=ANCHOR_LINE_WIDTH))
        remaining = len(self._anchors) - ANCHOR_PREVIEW
        if remaining > 0:
            text.append(f"\n+{remaining} more", style="dim")
        listing.update(text)

    @on(Button.Pressed, "#add-anchor")
    def _add_anchor_pressed(self) -> None:
        self.action_edit_anchors()

    def action_edit_anchors(self) -> None:
        """Open the editor. Bound to a key as well as the + button.

        `alt+a` because the button alone would be mouse-only: the username
        field has focus from the moment the app opens and `Input` swallows the
        letter and ctrl combinations, so alt is the only namespace left -- the
        same reason the tabs are alt+digit.
        """
        if not self._use_ai:
            # Nothing to open. The section is hidden precisely because anchors
            # cannot do anything with analysis off, so the key says why rather
            # than opening an editor whose result would be discarded.
            self.notify(
                "Anchors need AI analysis — turn it on first.",
                severity="warning",
            )
            return

        def adopt(updated: list[IdentityAnchor] | None) -> None:
            # None means the editor was left without changing anything, which
            # is not the same as leaving it with an empty list -- one keeps
            # what was there, the other clears it.
            if updated is None:
                return
            self._anchors = updated
            self._redraw_anchors()

        self.app.push_screen(AnchorScreen(self._anchors), adopt)

    # -- the drain ----------------------------------------------------------

    def _flush(self) -> None:
        """Draw whatever arrived since the last tick.

        Guarded on there being a reporter at all, so the timer is free to run
        from mount: a pane that has never scanned has nothing to drain and this
        costs one attribute check per tick.
        """
        reporter = self._reporter
        if reporter is None:
            return
        # A finished scan leaves its final numbers on screen and stops being
        # redrawn. Without this the timer keeps rebuilding identical counters
        # ten times a second for as long as the app stays open, which is real
        # work for a screen that cannot change.
        if (
            not self._scan_running
            and not reporter.pending
            and self._log_drawn == reporter.log_total
        ):
            return
        # Advances once per redraw, which is what turns the spinner. Kept here
        # rather than derived from the clock so it only moves when the screen
        # actually repaints -- a spinner that jumps several frames between
        # repaints reads as a stutter rather than as motion.
        self._tick += 1

        new_findings = reporter.drain_pending()
        if new_findings:
            self._append_findings(new_findings)

        self._append_log(reporter)
        self._redraw_startup(reporter)
        self._redraw_counters(reporter.counts)
        self._redraw_ai(reporter)
        self._redraw_progress(reporter.completed, reporter.total)

    @on(DataTable.RowSelected, "#feed")
    def _open_finding(self, event: DataTable.RowSelected) -> None:
        """Enter on a finding opens it in the real browser.

        The whole point of the tool is the list of places an account exists, and
        the next thing anyone does with such a list is look at one. Without this
        the URL is on screen and the only way to visit it is to retype it.

        Deliberately only ever user-initiated -- one row, on one keypress. The
        CLI's `--browse` opens every hit at once, which for a scan with forty
        of them is not a feature.
        """
        url = self._feed_urls.get(event.row_key.value or "")
        if not url:
            self.notify("No link for that row.", severity="warning")
            return
        webbrowser.open(url, new=2)

    @on(CounterRow.Toggled)
    def _toggle_status(self, event: CounterRow.Toggled) -> None:
        """Show or hide one kind of result in the feed.

        Refuses to empty the feed completely: turning the last visible kind off
        would leave a blank table with no indication of why, and "everything is
        hidden" is never what someone means by clicking one row.
        """
        if event.status in self._visible:
            if len(self._visible) == 1:
                self.notify(
                    "That is the only kind being shown — turn another on first.",
                    severity="warning",
                )
                return
            self._visible.discard(event.status)
        else:
            self._visible.add(event.status)
        self._rebuild_feed()

    def action_toggle_all_statuses(self) -> None:
        """Flip between every kind and hits only.

        The two ends of the filter are what anyone actually wants in a hurry --
        "show me everything that happened" and "back to just the accounts" --
        and reaching them by clicking four rows is four chances to lose track of
        which are on.
        """
        if self._visible == set(STATUS_ORDER):
            self._visible = default_visible_statuses()
        else:
            self._visible = set(STATUS_ORDER)
        self._rebuild_feed()

    def _rebuild_feed(self) -> None:
        """Redraw the whole table for the current filter.

        O(n) in results so far, which the per-tick drawing path deliberately
        avoids -- but this runs on a click, not on a timer, and there is no way
        to reveal rows that were never added without walking the history.
        """
        table = self.query_one("#feed", DataTable)
        table.clear()
        self._feed_urls.clear()
        reporter = self._reporter
        if reporter is not None:
            self._append_findings(list(reporter.findings))
        self._redraw_feed_title()
        # ALWAYS redraw the counters, with zeroes when no scan has run. Guarding
        # this on a reporter existing meant that before the first scan the title
        # updated and the strikethrough did not -- the filter visibly worked in
        # one place and appeared broken in the other, on the very screen someone
        # meets first. The rows are on screen from mount, so they have to answer
        # from mount.
        self._redraw_counters(
            reporter.counts if reporter is not None else blank_counts()
        )

    def _redraw_feed_title(self) -> None:
        """Name what the feed is currently showing.

        A filtered table that does not say it is filtered is how someone
        concludes a scan found nothing. The struck-through counters say it too,
        but the title is where the eye already is when reading the rows.
        """
        shown = [
            status_style(status).label
            for status in STATUS_ORDER
            if status in self._visible
        ]
        title = Text("FINDINGS", style="bold")
        if self._scan_username:
            # "scanning" only while it is. The verb was set when the scan
            # started and never cleared, so a finished run went on claiming to
            # be in progress for as long as the app stayed open -- and the one
            # thing a live-looking label must not do is outlive the activity it
            # describes.
            title.append("  scanning " if self._scan_running else "  ", style="dim")
            title.append(self._scan_username, style="cyan")
        if set(self._visible) != set(STATUS_ORDER):
            title.append(f"   showing {', '.join(shown)}", style="dim")
        self.query_one("#feed-title", Static).update(title)

    def _append_findings(self, findings: list[Finding]) -> None:
        table = self.query_one("#feed", DataTable)
        findings = [f for f in findings if f.status in self._visible]
        for finding in findings:
            # Keyed by position so the URL can be recovered on selection. The
            # table cell shows the failure reason instead of the URL for
            # anything that is not a hit, so the cell text cannot be the source.
            key = str(len(self._feed_urls))
            self._feed_urls[key] = finding.url
            table.add_row(
                status_cell(finding.status),
                # Text(), always: site names and URLs come from the manifest and
                # from scanned pages, and Rich reads markup in cells -- a site
                # with brackets in its name would be swallowed as a style tag.
                # The same trap `show` documents for its tables.
                Text(finding.site_name, overflow="ellipsis", no_wrap=True),
                # Right-aligned by padding rather than by `justify`, so the
                # digits stack in the same cell whatever the row -- the same
                # argument `stat_row` makes for the counters.
                Text(
                    response_label(finding.elapsed).rjust(TIME_WIDTH),
                    style="dim",
                ),
                Text(
                    finding.detail or finding.url,
                    style="dim" if finding.detail else "",
                    overflow="ellipsis",
                    no_wrap=True,
                ),
                key=key,
            )
        # Follow the tail only while nobody is reading. Once the feed has focus
        # the operator is scrolling it deliberately, and yanking the viewport
        # back to the bottom every tenth of a second makes it impossible to read
        # a row that is still arriving.
        if not table.has_focus:
            table.scroll_end(animate=False)

    def _show_activity_placeholder(self) -> None:
        """Say what this pane is for while it has nothing in it.

        It stays empty through most of a healthy scan now that everything with
        a panel of its own is suppressed. An empty box reads as broken; a box
        that says it only speaks up when there is a problem reads as quiet.
        """
        log = self.query_one("#activity", RichLog)
        log.clear()
        log.write(
            Text("nothing to report — problems and warnings appear here",
                 style="dim italic")
        )

    def _append_log(self, reporter: TuiReporter) -> None:
        """Write the narration that arrived since the last tick.

        Append-only, tracked by a count, for the same reason the feed drains
        rather than rebuilds: redrawing the whole log every tenth of a second
        would be O(n) in lines so far and would throw away the scroll position
        of anyone reading it.

        THE COUNT IS THE REPORTER'S RUNNING TOTAL, not the length of its buffer,
        and that distinction is a bug already paid for. The buffer is a bounded
        deque: once it is full, appending drops a line off the far end and the
        length stops changing forever. Comparing lengths therefore reported "no
        new lines" from the moment it filled, and the ACTIVITY pane froze for
        the rest of the run -- last line drawn, "Preparing local AI model",
        while the model went on working perfectly. In verbose that took one
        tick, because `ai_cached_evidence` prints a JSON panel per stored site
        before the scan starts.

        A burst larger than the whole buffer between two ticks loses its oldest
        lines. That is said out loud rather than papered over: a diagnostic
        channel that silently discards diagnostics is worse than a short one.
        """
        lines = reporter.snapshot_log()
        produced = reporter.log_total
        if produced == self._log_drawn:
            return
        if not self._log_drawn:
            # Replaces the placeholder the moment there is something real. The
            # pane is usually empty now, so it says why rather than looking
            # broken or forgotten.
            self.query_one("#activity", RichLog).clear()
        log = self.query_one("#activity", RichLog)
        undrawn = produced - self._log_drawn
        dropped = undrawn - len(lines)
        if dropped > 0:
            log.write(
                Text(f"... {dropped} lines dropped (log buffer full)",
                     style="dim italic")
            )
            undrawn = len(lines)
        for line in lines[len(lines) - undrawn :]:
            log.write(line)
        self._log_drawn = produced

    def _redraw_startup(self, reporter: TuiReporter) -> None:
        """Draw the startup steps, one self-replacing line each.

        The whole block is rewritten every tick rather than appended to, which
        is what makes a loading line become a ready line in place instead of
        leaving its own history behind it. It is at most two short lines, so
        rebuilding it costs nothing -- unlike the feed, which is why that one
        appends.

        The lines stay on screen after they land. Knowing the model took 96
        seconds is worth a line for the rest of the run, and a block that
        vanished the moment it succeeded would take the answer with it.
        """
        phases = reporter.phases
        block = self.query_one("#startup", Static)
        if not phases:
            block.update("")
            return
        text = Text()
        for index, phase in enumerate(phases):
            if index:
                text.append("\n")
            text.append_text(
                phase_line(
                    phase.label,
                    phase.state,
                    elapsed=phase.elapsed,
                    tick=self._tick,
                )
            )
        text.append("\n")
        block.update(text)

    def _redraw_counters(self, counts: dict[QueryStatus, int]) -> None:
        for status in STATUS_ORDER:
            style = status_style(status)
            row = self.query_one(f"#count-{status.name.lower()}", CounterRow)
            row.update(
                stat_row(
                    style.label,
                    counts.get(status, 0),
                    style.style,
                    # Absent is dimmed. It is almost always the largest number
                    # on the panel and it is almost never the interesting one;
                    # at full contrast it draws the eye away from the count that
                    # matters. Dimming is not hiding -- it is still exact.
                    muted=status is QueryStatus.AVAILABLE,
                    # Struck through when the feed is not showing this kind.
                    # The count stays exact and readable either way -- what the
                    # line reports and what the feed is displaying are two
                    # different facts, and only the second one is being toggled.
                    struck=status not in self._visible,
                )
            )

    def _redraw_ai(self, reporter: TuiReporter) -> None:
        stats = reporter.ai_stats
        if not stats.scheduled:
            # Nothing scheduled means the model is off or nothing was found
            # worth extracting. Say which, rather than showing a block of
            # zeroes that reads like a stalled pipeline.
            self.query_one("#ai-counters", Static).update(
                Text("not running", style="dim italic")
            )
            return
        lines = Text()
        lines.append_text(stat_row("extracted", stats.completed, "cyan"))
        lines.append("\n")
        lines.append_text(stat_row("with facts", stats.with_facts, "green"))
        lines.append("\n")
        lines.append_text(
            stat_row("queued", stats.scheduled - stats.completed, "dim", muted=True)
        )
        self.query_one("#ai-counters", Static).update(lines)

    def _redraw_progress(self, completed: int, total: int) -> None:
        if self._started_at is not None and self._scan_running:
            self._elapsed = self._monotonic() - self._started_at

        strip = self.query_one("#progress-strip", Static)
        # Absent until there is something to report, rather than sitting under
        # the heading as a row of dashes labelled "idle". It stays after a scan
        # finishes, holding the final figures.
        strip.display = bool(total)
        if total:
            counts = Text.assemble(
                (f"{completed}", "bold"),
                ("/", "dim"),
                (f"{total}", "dim"),
                ("  ", ""),
                (elapsed_label(self._elapsed), "dim"),
            )
        else:
            counts = Text("idle", style="dim italic")

        # The counts are measured first and the bar takes what is left, so the
        # line always ends flush with the panes above rather than growing by a
        # cell each time the completed figure gains a digit.
        available = max(0, strip.size.width - counts.cell_len - 2)
        line = progress_bar(completed, total, available)
        line.append("  ")
        line.append_text(counts)
        strip.update(line)

    @staticmethod
    def _monotonic() -> float:
        from time import perf_counter

        return perf_counter()

    # -- running ------------------------------------------------------------

    @on(Button.Pressed, "#scan-button")
    def _button(self) -> None:
        self.action_stop() if self._scan_running else self.action_run()

    @on(Input.Submitted, "#target-input")
    def _submitted(self) -> None:
        self.action_run()

    def action_run(self) -> None:
        if self._scan_running:
            return
        username = self.query_one("#target-input", Input).value.strip()
        if not username:
            self.query_one("#target-input", Input).focus()
            self.notify("Type a username first.", severity="warning")
            return
        self._check_stored_then_run(username)

    @work(exclusive=True, group="scan-peek")
    async def _check_stored_then_run(self, username: str) -> None:
        """Ask before re-scanning a username that already has results.

        The question replaced a permanent `re-scan` toggle: it mattered on
        almost no scan and sat on screen for all of them, while the one moment
        the answer matters is this one. It also puts the resume state in front
        of the run rather than in the log afterwards -- a scan that skipped all
        680 sites used to be indistinguishable from one that found nothing.

        A username with nothing stored is scanned immediately. A prompt that
        appears every time is a prompt that gets dismissed without reading, and
        there is nothing to warn about on a first scan.
        """
        from sherlock_project.tui.resume_screen import ResumeChoice, ResumeScreen
        from sherlock_project.tui.runner import peek_stored

        plan = await peek_stored(
            username=username, settings_values=self._settings_values
        )
        if not plan.is_known:
            self._begin(username, fresh=False)
            return

        def chosen(choice: ResumeChoice | None) -> None:
            if choice == "resume":
                self._begin(username, fresh=False)
            elif choice == "fresh":
                self._begin(username, fresh=True)
            elif choice == "view":
                self.post_message(self.ShowStored(username))

        self.app.push_screen(ResumeScreen(username, plan), chosen)

    class ShowStored(Message):
        """Asked to look at what is already stored rather than scan again.

        A message rather than a direct tab switch: which tab holds the results
        is the app's business, not this pane's, and a pane that reaches across
        to rearrange its siblings is how two screens end up owning one layout.
        """

        def __init__(self, username: str) -> None:
            super().__init__()
            self.username = username

    def _begin(self, username: str, *, fresh: bool) -> None:
        self._reset_for(username)
        self._worker = self._scan_worker(username, fresh=fresh)

    def action_stop(self) -> None:
        if not self._scan_running or self._worker is None:
            return
        # Cancelling the worker cancels the coroutine, which propagates
        # CancelledError into the scan exactly as ^C does on the CLI -- the
        # engine's own cancellation path then tears down the browser and the AI
        # pipeline. Nothing here needs to know how that works.
        #
        # The scan's own worker, held by reference. `self.workers` is the APP's
        # worker manager, not this widget's, so iterating it and cancelling
        # everything would also kill the results pane's database reads -- one
        # press of STOP tearing down unrelated work elsewhere in the app.
        self._worker.cancel()

    def _reset_for(self, username: str) -> None:
        table = self.query_one("#feed", DataTable)
        table.clear()
        self._feed_urls.clear()
        self._tick = 0
        self.query_one("#startup", Static).update("")
        self._show_activity_placeholder()
        self._log_drawn = 0
        self._reporter = TuiReporter(verbose=self._verbose)
        self._scan_running = True
        self._elapsed = 0.0
        self._started_at = self._monotonic()
        self.query_one("#scan-button", Button).label = "STOP"
        self._scan_username = username
        self._redraw_feed_title()

    @work(exclusive=True)
    async def _scan_worker(self, username: str, *, fresh: bool = False) -> None:
        """Run one scan.

        Imported here rather than at module scope: the runner pulls in the
        engines, which pull in Playwright, and a settings-only session should
        not pay a browser import to draw a screen.
        """
        from sherlock_project.tui.runner import run_scan_session

        assert self._reporter is not None
        await run_scan_session(
            username=username,
            reporter=self._reporter,
            settings_values=self._settings_values,
            use_ai=self._use_ai,
            fresh=fresh,
            # Only when analysis is on. Anchors kept from an earlier toggle are
            # hidden while it is off, and state you cannot see must not be able
            # to change what a run does -- so it is not merely inert here, it
            # is not passed at all. Turning analysis back on restores them
            # rather than making you retype.
            anchors=list(self._anchors) if self._use_ai else [],
        )

    @on(Worker.StateChanged)
    def _worker_state(self, event: Worker.StateChanged) -> None:
        # Only this pane's scan worker. Worker messages bubble, so any other
        # worker added here later would otherwise finish the scan on screen
        # while it was still running.
        if event.worker is not self._worker:
            return
        if event.state in (
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        ):
            self._finish(event)

    def _finish(self, event: Worker.StateChanged) -> None:
        self._scan_running = False
        self._worker = None
        self._started_at = None
        # The title carries the running/finished distinction, so it has to be
        # redrawn here -- `_flush` does not touch it.
        self._redraw_feed_title()
        self.query_one("#scan-button", Button).label = "SCAN"
        # One last drain. Results that landed between the final tick and the
        # scan ending would otherwise never be drawn, so the table would end a
        # few rows short of the count beside it -- a discrepancy that looks like
        # lost data and is impossible to explain after the fact.
        self._flush()

        if event.state is WorkerState.ERROR:
            self.notify(
                f"Scan failed: {event.worker.error}",
                severity="error",
                timeout=10,
            )
        elif event.state is WorkerState.CANCELLED:
            self.notify("Scan stopped. Partial results are saved.", severity="warning")
        else:
            reporter = self._reporter
            found = reporter.counts[QueryStatus.CLAIMED] if reporter else 0
            self.notify(f"Scan complete — {found} found.")
