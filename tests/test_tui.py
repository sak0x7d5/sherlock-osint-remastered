"""The unified UI, driven through Textual's pilot.

The assertions worth having here are about the decisions, not the pixels. That
absent results never reach the feed is a design commitment -- it is what keeps a
680-site scan readable -- and it is invisible to anything that only checks the
app starts. Same for the reporter counting nothing of its own: the moment it
grows its own tally, the TUI and the CLI can disagree about the same scan, and
no test that renders a screen would notice.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import ClassVar

import pytest
from rich.console import Console
from textual.screen import ModalScreen

from sherlock_project.database import SherlockDB
from sherlock_project.result import QueryResult, QueryStatus
from sherlock_project.tui.app import SherlockUI, run_ui
from sherlock_project.tui.reporter import TuiReporter, describe_settings
from sherlock_project.tui.results_pane import ResultsPane
from sherlock_project.tui.runner import ai_is_configured, resolved
from sherlock_project.tui.scan_pane import CounterRow, ScanPane
from sherlock_project.tui.settings_pane import SettingsPane
from sherlock_project.tui.theme import (
    SPINNER_FRAMES,
    STATUS_ORDER,
    STATUS_STYLES,
    elapsed_label,
    phase_line,
    progress_bar,
    spinner,
    stat_row,
    status_cell,
    status_from_name,
    status_key,
    status_style,
)


def _result(site: str, status: QueryStatus, **kwargs) -> QueryResult:
    return QueryResult("someone", site, f"https://{site}/someone", status, **kwargs)


# -- the reporter -----------------------------------------------------------


def test_every_result_is_retained_so_the_filter_can_reveal_it():
    """The reporter keeps everything; the pane decides what to draw.

    It used to drop anything the feed did not show by default, which made that
    choice irreversible -- the rows were gone, so no filter could ever bring
    them back. What keeps a 680-site scan readable is now the default filter,
    not destroying the results.
    """
    reporter = TuiReporter()
    reporter.start("someone", total=4)
    for site, status in (
        ("A", QueryStatus.AVAILABLE),
        ("B", QueryStatus.CLAIMED),
        ("C", QueryStatus.WAF),
        ("D", QueryStatus.UNKNOWN),
    ):
        reporter.update(_result(site, status))

    assert [f.site_name for f in reporter.findings] == ["A", "B", "C", "D"]


def test_only_hits_are_visible_before_the_filter_is_touched():
    """Absent is the overwhelming majority of any scan, so a feed that draws it
    scrolls the actual findings off screen within seconds of starting."""
    from sherlock_project.tui.theme import default_visible_statuses

    assert default_visible_statuses() == {QueryStatus.CLAIMED}


def test_counts_come_from_the_parent_not_a_second_tally():
    """The TUI and the CLI must not be able to disagree about one scan.

    `TerminalReporter` already counts every status. This asserts the TUI reads
    those rather than keeping its own, which is the only way the two surfaces
    stay consistent by construction instead of by vigilance.
    """
    reporter = TuiReporter()
    reporter.start("someone", total=5)
    for status in (
        QueryStatus.CLAIMED,
        QueryStatus.CLAIMED,
        QueryStatus.AVAILABLE,
        QueryStatus.UNKNOWN,
        QueryStatus.WAF,
    ):
        reporter.update(_result("site", status))

    assert reporter.counts[QueryStatus.CLAIMED] == 2
    assert reporter.counts[QueryStatus.AVAILABLE] == 1
    assert reporter.counts[QueryStatus.UNKNOWN] == 1
    assert reporter.counts[QueryStatus.WAF] == 1
    # The parent's own field, not a copy kept beside it.
    assert reporter.counts[QueryStatus.CLAIMED] == reporter._scan_found
    assert reporter.completed == 5


def test_draining_hands_over_only_what_is_new():
    """Append-only drawing depends on this.

    A drain that returned the whole history would make every repaint O(n) in
    results so far, which is the cost the buffer exists to avoid.
    """
    reporter = TuiReporter()
    reporter.start("someone", total=2)
    reporter.update(_result("A", QueryStatus.CLAIMED))

    assert [f.site_name for f in reporter.drain_pending()] == ["A"]
    assert reporter.drain_pending() == []

    reporter.update(_result("B", QueryStatus.CLAIMED))
    assert [f.site_name for f in reporter.drain_pending()] == ["B"]
    # History survives the drain even though the pending queue does not.
    assert len(reporter.findings) == 2


def test_the_reporter_writes_nothing_to_the_real_stdout(capsys):
    """Anything printed would land on top of the Textual canvas.

    The scan reports through the same methods it always did; this asserts they
    are captured rather than printed.
    """
    reporter = TuiReporter()
    reporter.start("someone", total=1)
    reporter.update(_result("GitHub", QueryStatus.CLAIMED))
    reporter.warning("something went sideways")

    assert capsys.readouterr().out == ""
    # Captured, not discarded -- a warning nobody can see is worse than one
    # printed in the wrong place.
    assert any("sideways" in line.plain for line in reporter.snapshot_log())


def test_context_reaches_the_finding():
    """"Inconclusive" and "inconclusive because it timed out" are different."""
    reporter = TuiReporter()
    reporter.start("someone", total=1)
    reporter.update(_result("Slow", QueryStatus.UNKNOWN, context="timed out"))

    assert reporter.findings[0].detail == "timed out"


# -- the activity log carries only what has no panel -------------------------


def _scanned_reporter() -> TuiReporter:
    reporter = TuiReporter()
    reporter.browser_status("starting")
    reporter.browser_status("ready")
    reporter.ai_model_starting()
    reporter.ai_model_ready()
    reporter.start("marcus", total=6)
    for site, status in (
        ("GitHub", QueryStatus.CLAIMED),
        ("Reddit", QueryStatus.WAF),
        ("Bluesky", QueryStatus.CLAIMED),
        ("Slow", QueryStatus.UNKNOWN),
        ("Nowhere", QueryStatus.AVAILABLE),
        ("Odd", QueryStatus.ILLEGAL),
    ):
        reporter.update(_result(site, status))
    reporter.finish_scan(elapsed_time=12.0)
    return reporter


def test_the_activity_log_does_not_repeat_the_panels():
    """A log that restates the screen is a log nobody reads when it matters.

    The reporter was written for a terminal whose only output is a stream of
    lines, so it narrates everything. Here each of those facts already has a
    panel, and printing them again put the same information twice on one
    screen in two formats -- a 6-site scan wrote 12 lines, 10 of them
    duplicates.
    """
    reporter = _scanned_reporter()
    log = " | ".join(line.plain for line in reporter.snapshot_log())

    # Hits are the findings table.
    assert "GitHub" not in log
    assert "Bluesky" not in log
    # Startup is the phase block.
    assert "Web scanner ready" not in log
    assert "Local AI model ready" not in log
    # Totals are the counters.
    assert "Checking username" not in log
    assert "2 found" not in log

    # The findings and the counts are still fully intact.
    assert len(reporter.findings) == 6
    assert reporter.counts[QueryStatus.CLAIMED] == 2


def test_the_unresolved_caveat_survives_and_points_in_app():
    """The one thing the counters cannot say.

    "3 blocked" is a number; "a site that never answered is not a site where
    nobody was home" is the distinction this tool refuses to blur. It also used
    to recommend `sherlock show --unresolved`, a command nobody in the app can
    run -- the third instance of that bug.
    """
    log = " ".join(line.plain for line in _scanned_reporter().snapshot_log())

    assert "gave no answer" in log
    assert "inconclusive" in log and "blocked by bot protection" in log
    assert 'Not the same as "not found"' in log
    assert "UNRESOLVED tab" in log
    assert "--unresolved" not in log


def test_per_site_ai_narration_stays_out_of_the_log():
    """One line per extracted site is 260 lines on a real username, saying what
    the ANALYSIS block says in three numbers. Failures are exempt -- they are
    the one AI event with no counter of its own."""
    reporter = TuiReporter()
    reporter.start("marcus", total=1)
    reporter.ai_pass_started()
    for _ in range(20):
        reporter.ai_scheduled()
        reporter.ai_job_started("GitHub")
        reporter.ai_job_finished("with_facts")

    log = " | ".join(line.plain for line in reporter.snapshot_log())
    assert "GitHub" not in log

    reporter.ai_failed("Reddit", RuntimeError("model refused"))
    failures = " | ".join(line.plain for line in reporter.snapshot_log())
    assert "Reddit" in failures


async def test_the_activity_pane_says_why_it_is_empty():
    """It stays empty through most of a healthy scan now. An empty box reads as
    broken; one that says it only speaks up on a problem reads as quiet."""
    from textual.widgets import RichLog

    app = SherlockUI()
    async with app.run_test():
        log = app.query_one("#activity", RichLog)
        rendered = " ".join(
            segment.text for line in log.lines for segment in line
        )
        assert "nothing to report" in rendered


# -- verbose: what the model is actually doing -------------------------------


def _token_limit_failure():
    from sherlock_project.ai_engine import StructuredResponseError

    return StructuredResponseError(
        "OSINT extraction",
        stop_reason="maxPredictedTokensReached",
        predicted_tokens=1024,
        max_tokens=1024,
        parsed_type="str",
        final_content_chars=4701,
        validation_error="json_invalid",
    )


def test_verbose_turns_a_bare_failure_into_a_diagnosis():
    """"extraction failed" and "it ran out of tokens mid-JSON" are different
    facts, and only the second one tells you what to change.

    All of this already existed on the reporter and was thrown away, because
    the TUI always built it with verbose off.
    """
    quiet = TuiReporter()
    quiet.ai_failed("Bandcamp", _token_limit_failure())
    bare = " ".join(line.plain for line in quiet.snapshot_log())
    assert "profile extraction failed" in bare
    assert "maxPredictedTokensReached" not in bare

    loud = TuiReporter(verbose=True)
    loud.ai_failed("Bandcamp", _token_limit_failure())
    detailed = " ".join(line.plain for line in loud.snapshot_log())
    assert "maxPredictedTokensReached" in detailed
    assert "1024/1024" in detailed
    assert "json_invalid" in detailed


def test_verbose_does_not_repeat_what_a_panel_already_draws():
    """Reverses the earlier "verbose defeats the de-duplication" decision.

    Running it showed the opposite of the argument. What verbose adds is the
    request traces, the failure diagnostics and the cached extractions -- none
    of which go through `_quiet` -- while what it was adding through `_quiet`
    was the findings table rewritten as prose, one hit per line, directly beside
    the findings table. The one fact those lines carried that the table did not
    was the response time, and that is a column now.
    """
    loud = TuiReporter(verbose=True)
    loud.start("someone", total=1)
    loud.update(_result("GitHub", QueryStatus.CLAIMED, query_time=0.4))
    loud.ai_job_started("GitHub")
    loud.ai_model_starting()
    loud.ai_model_ready()
    log = " ".join(line.plain for line in loud.snapshot_log())

    # The hit is in the table, with its URL and its timing; not in the log.
    assert "https://GitHub/someone" not in log
    assert [f.site_name for f in loud.findings] == ["GitHub"]
    # Startup has the phase block, extraction has the ANALYSIS counters.
    assert "Preparing local AI model" not in log
    assert "Local AI model ready" not in log

    # Failures still speak, and still say why -- they have no panel of their own.
    loud.ai_failed("Bandcamp", _token_limit_failure())
    assert "maxPredictedTokensReached" in " ".join(
        line.plain for line in loud.snapshot_log()
    )


def test_the_log_keeps_drawing_after_its_buffer_rolls_over():
    """The freeze this fixes: the ACTIVITY pane stopped updating mid-run.

    The buffer is a bounded deque, so once it is full its length never changes
    again. The pane tracked its position by comparing that length to what it had
    drawn, concluded nothing new had arrived, and drew nothing for the rest of
    the run -- last line on screen "Preparing local AI model", while the model
    went on working perfectly. In verbose that took one tick, because a JSON
    panel per stored site arrives before the scan even starts.
    """
    from sherlock_project.tui.reporter import LOG_LIMIT

    reporter = TuiReporter()
    for index in range(LOG_LIMIT + 50):
        reporter.info(f"line {index}")

    # The buffer is full and its length has stopped moving...
    assert len(reporter.snapshot_log()) == LOG_LIMIT
    # ...but the count of what was produced has not.
    assert reporter.log_total == LOG_LIMIT + 50

    before = reporter.log_total
    reporter.info("something that must still reach the screen")
    assert reporter.log_total == before + 1
    assert len(reporter.snapshot_log()) == LOG_LIMIT


async def test_a_dropped_burst_is_reported_rather_than_silently_lost():
    """A diagnostic channel that quietly discards diagnostics is worse than a
    short one -- so an overflow between two ticks says how much it dropped."""
    from textual.widgets import RichLog

    from sherlock_project.tui.reporter import LOG_LIMIT

    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        pane = app.query_one(ScanPane)
        reporter = TuiReporter()
        pane._reporter = reporter
        for index in range(LOG_LIMIT + 25):
            reporter.info(f"line {index}")
        pane._flush()
        await pilot.pause()

        log = app.query_one("#activity", RichLog)
        rendered = " ".join(
            segment.text for line in log.lines for segment in line
        )
        assert "25 lines dropped" in rendered
        # And the run keeps drawing afterwards.
        reporter.warning("still alive")
        pane._flush()
        await pilot.pause()
        rendered = " ".join(
            segment.text
            for line in app.query_one("#activity", RichLog).lines
            for segment in line
        )
        assert "still alive" in rendered


async def test_the_findings_table_carries_the_response_time():
    """The one thing the deleted per-hit log line said that the table did not.

    Blank rather than zero for a restored row: the database stores the verdict,
    not how long the request took, and a number there would claim an instant
    answer where the truth is that no request was made.
    """
    from textual.widgets import DataTable

    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        pane = app.query_one(ScanPane)
        reporter = TuiReporter()
        reporter.start("someone", total=1)
        reporter.update(_result("GitHub", QueryStatus.CLAIMED, query_time=0.412))
        pane._reporter = reporter
        pane._flush()
        await pilot.pause()

        table = app.query_one("#feed", DataTable)
        row = table.get_row_at(0)
        assert "412ms" in "".join(str(cell) for cell in row)

    restored = TuiReporter()
    restored.restored_results(
        username="someone",
        results={"GitHub": {"status": _result("GitHub", QueryStatus.CLAIMED)}},
        to_scan=0,
    )
    assert [f.elapsed for f in restored.findings] == [None]


def test_a_site_timing_is_not_rounded_to_whole_seconds():
    """`elapsed_label` measures a whole scan; most sites answer in well under a
    second, so through that formatter nearly every row would read `0s`."""
    from sherlock_project.tui.theme import response_label

    assert response_label(0.412) == "412ms"
    assert response_label(2.5) == "2.5s"
    assert response_label(41.0) == "41s"
    assert response_label(None) == ""


async def test_the_verbose_toggle_starts_from_the_stored_setting():
    """`-v` has a config layer on the CLI (`output.verbose`), so this one does
    too -- unlike analysis and re-scan, whose flags have none."""
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        assert pane._verbose is False
        assert "off" in str(app.query_one("#toggle-verbose", Button).label)

        await pilot.click("#toggle-verbose")
        await pilot.pause()
        assert pane._verbose is True
        assert "on" in str(app.query_one("#toggle-verbose", Button).label)


async def test_the_scan_reporter_is_built_with_the_chosen_verbosity():
    """The toggle has to reach the reporter, or it is decoration."""
    app = SherlockUI()
    async with app.run_test():
        pane = app.query_one(ScanPane)
        pane._verbose = True
        pane._reset_for("someone")
        assert pane._reporter is not None
        assert pane._reporter.verbose is True


# -- startup phases ---------------------------------------------------------


def test_no_startup_lines_before_anything_has_announced_itself():
    """A fixed list of steps would show permanent blanks for the ones this run
    does not use -- the fast transport starts no browser, and a scan without
    analysis loads no model."""
    assert TuiReporter().phases == []


def test_the_model_line_goes_loading_then_ready():
    """The swap the whole block exists for: one line that becomes its own
    result, rather than a loading message and a separate ready message."""
    reporter = TuiReporter()
    reporter.ai_model_starting()

    loading = reporter.phases
    assert [p.label for p in loading] == ["model"]
    assert loading[0].state == "loading"

    reporter.ai_model_ready()
    ready = reporter.phases
    assert [p.label for p in ready] == ["model"]
    assert ready[0].state == "ready"
    # Still one line, not two -- it replaced itself.
    assert len(ready) == 1


async def test_a_finished_phase_stops_counting():
    """It reports how long the step took, not how long ago it started.

    The parent reporter keeps only a start time, so recomputing the duration on
    every repaint made a finished step climb forever -- `ready <1s`, then
    `ready 1s`, then `ready 2s` -- and eventually claim minutes for something
    that took under a second. Caught by a real scan, not by a mock.
    """
    import asyncio

    reporter = TuiReporter()
    reporter.browser_status("starting")
    reporter.browser_status("ready")

    settled = reporter.phases[0].elapsed
    await asyncio.sleep(0.25)
    assert reporter.phases[0].elapsed == settled

    # A step still running does keep counting, which is the other half.
    reporter.ai_model_starting()
    first = reporter.phases[1].elapsed
    await asyncio.sleep(0.05)
    assert reporter.phases[1].elapsed > first


def test_a_model_that_never_loads_says_failed_not_loading():
    """A spinner that turns forever is indistinguishable from a hang."""
    reporter = TuiReporter()
    reporter.ai_model_starting()
    reporter.ai_model_failed(RuntimeError("no server"))

    assert [p.state for p in reporter.phases] == ["failed"]


def test_installing_a_browser_is_not_called_loading():
    """A first run downloads a browser, which is minutes rather than seconds,
    and someone who thinks a scan hung will kill it."""
    reporter = TuiReporter()
    reporter.browser_status("installing")
    assert [p.state for p in reporter.phases] == ["installing"]

    reporter.browser_status("ready")
    assert [p.state for p in reporter.phases] == ["ready"]


def test_the_spinner_advances_and_wraps():
    frames = {spinner(tick) for tick in range(len(SPINNER_FRAMES))}
    assert len(frames) == len(SPINNER_FRAMES)
    assert spinner(0) == spinner(len(SPINNER_FRAMES))


def test_a_fast_step_reads_as_under_a_second_not_as_zero():
    """A browser that came up in six tenths of a second reporting "0s" looks
    like a step that never ran."""
    assert "<1s" in phase_line("browser", "ready", elapsed=0.6).plain
    assert "0s" not in phase_line("browser", "ready", elapsed=0.6).plain
    assert "42s" in phase_line("model", "ready", elapsed=42.0).plain
    # Past a minute it reads as a duration, not as a pile of seconds.
    assert "1m36s" in phase_line("model", "ready", elapsed=96.0).plain


def test_phase_lines_are_all_the_same_width():
    """They stack in a column, so a ragged right edge reads as a glitch while
    the spinner is moving."""
    widths = {
        phase_line("browser", "loading", elapsed=12.0, tick=3).cell_len,
        phase_line("model", "ready", elapsed=96.0).cell_len,
        phase_line("model", "failed").cell_len,
    }
    assert len(widths) == 1, f"phase lines are ragged: {widths}"


async def test_the_startup_block_draws_and_then_stops_animating():
    """Live while it loads, still afterwards."""
    from textual.widgets import Static

    app = SherlockUI()
    async with app.run_test():
        pane = app.query_one(ScanPane)
        reporter = TuiReporter()
        reporter.start("someone", total=1)
        reporter.ai_model_starting()
        pane._reporter = reporter
        pane._scan_running = True

        pane._flush()
        first = app.query_one("#startup", Static).render().plain
        assert "loading" in first

        reporter.ai_model_ready()
        pane._flush()
        after = app.query_one("#startup", Static).render().plain
        assert "ready" in after
        assert "loading" not in after


# -- no command-line advice inside the app ----------------------------------


def _profile_with_warnings():
    from sherlock_project.profile_synthesis import ProfileSynthesis

    return ProfileSynthesis.model_validate(
        {
            "username": "0day",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "resolved",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "warnings": [
                (
                    "Pass-two decision failed for site id 466 "
                    "(StructuredResponseError)"
                ),
                "Pass-one extraction is still pending for site ids: 264",
            ],
        }
    )


def test_the_profile_never_tells_the_app_to_use_a_command_line_flag():
    """"run with --verbose" names a flag nobody inside a running app can type.

    The count is still worth showing -- two notes exist and that is true on
    every surface -- so what changes is the route to reading them, which is the
    caller's business rather than the renderer's.
    """
    from sherlock_project.tui.results_pane import render_profile_text

    rendered = render_profile_text(_profile_with_warnings()).plain
    assert "2 notes about how this was built" in rendered
    assert "--verbose" not in rendered
    assert "press v" in rendered


def test_the_anchor_caveat_is_stated_once_not_twice():
    """It was printed by the renderer AND stored as the first warning.

    So pressing v put the same statement on two consecutive lines in slightly
    different words. The renderer's line is the one that stays: it shows whether
    or not anyone asks for diagnostics, and it is the only note that changes how
    the VALUES should be read rather than how they were gathered.
    """
    from sherlock_project.profile_synthesis import (
        AGGREGATE_ANCHOR_WARNING,
        ProfileSynthesis,
    )
    from sherlock_project.tui.results_pane import render_profile_text

    profile = ProfileSynthesis.model_validate(
        {
            "username": "0day",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "warnings": [
                AGGREGATE_ANCHOR_WARNING,
                "Pass-one extraction is still pending for site ids: 1726",
            ],
        }
    )

    expanded = render_profile_text(profile, show_notes=True).plain
    assert expanded.count("No anchors used") == 1
    assert "No anchor was supplied" not in expanded
    # The other note is diagnostics and still shows.
    assert "1726" in expanded

    # And the count offered beforehand matches what expanding reveals -- one,
    # not the two that are stored.
    collapsed = render_profile_text(profile).plain
    assert "1 note about how this was built" in collapsed


def test_the_caveat_leads_and_the_diagnostics_trail():
    """Position by purpose, not by type.

    The anchor caveat changes how the values should be READ, so it has to come
    before them -- placed after, it arrives once they have already been read as
    fact. Everything else describes how the profile was BUILT, and stacked on
    top it pushed the values down the panel behind a wall of site ids.
    """
    from sherlock_project.profile_synthesis import (
        AGGREGATE_ANCHOR_WARNING,
        ProfileSynthesis,
    )
    from sherlock_project.tui.results_pane import render_profile_text

    profile = ProfileSynthesis.model_validate(
        {
            "username": "0day",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "warnings": [
                AGGREGATE_ANCHOR_WARNING,
                "Pass-one extraction is still pending for site ids: 1726",
            ],
        }
    )

    for rendered in (
        render_profile_text(profile, show_notes=True).plain,
        render_profile_text(profile).plain,
    ):
        caveat = rendered.index("No anchors used")
        value = rendered.index("Avery Stone")
        built = rendered.index("about how this was built") if (
            "about how this was built" in rendered
        ) else rendered.index("1726")

        assert caveat < value, "the caveat must precede the values it qualifies"
        assert value < built, "provenance must not push the values down the panel"


def test_pressing_v_shows_the_notes_instead_of_the_count():
    """The hint has to be true: the key it names must actually reveal them."""
    from sherlock_project.tui.results_pane import render_profile_text

    shown = render_profile_text(_profile_with_warnings(), show_notes=True).plain
    assert "StructuredResponseError" in shown
    assert "2 notes about how this was built" not in shown


def test_the_cli_keeps_its_own_hint():
    """The command line CAN act on --verbose, so it still says so.

    This is the check that the fix was a parameter rather than a deletion.
    """
    from io import StringIO

    from rich.console import Console

    from sherlock_project.notify import TerminalReporter

    buffer = StringIO()
    console = Console(file=buffer, width=100, no_color=True)
    TerminalReporter(console=console, error_console=console).render_profile(
        _profile_with_warnings()
    )
    assert "--verbose" in buffer.getvalue()


def test_changing_model_says_how_to_redo_extractions_in_the_app():
    """Why switching model looks like it did nothing.

    The Pass 1 cache is keyed on the PROMPT CONTRACT, not the model, so an
    extraction made by another model is still valid and is kept. The new model
    therefore only touches sites that have no extraction yet -- which is correct,
    and indistinguishable from being ignored unless something says so.

    That line is the only thing standing between the user and the wrong
    conclusion, so it has to name a control they have. It named `--fresh`.
    """
    reporter = TuiReporter()
    reporter.ai_extractions_from_other_models(
        username="0day",
        configured_model="qwen/qwen3-4b",
        counts={"qwen/qwen3-8b": 98},
    )
    log = " ".join(line.plain for line in reporter.snapshot_log())

    assert "did not come from qwen/qwen3-4b" in log
    assert "Re-scan all" in log
    assert "--fresh" not in log
    assert "sherlock 0day" not in log


def test_the_cli_still_names_the_flag_for_redoing_extractions():
    """The command line has a command; only the wording is surface-specific."""
    from io import StringIO

    from rich.console import Console

    from sherlock_project.notify import TerminalReporter

    buffer = StringIO()
    console = Console(file=buffer, width=120, no_color=True)
    TerminalReporter(
        console=console, error_console=console
    ).ai_extractions_from_other_models(
        username="0day",
        configured_model="qwen/qwen3-4b",
        counts={"qwen/qwen3-8b": 98},
    )
    assert "--fresh" in buffer.getvalue()


def test_the_scan_log_points_at_the_results_tab_not_at_a_flag():
    """The end-of-synthesis panel is drawn for the CLI's scrollback.

    In the app it arrived clipped -- laid out against the reporter's fixed
    200-column sink, then written into an activity log a third that wide -- and
    carrying flag advice. One line, pointing where the profile really is.
    """
    reporter = TuiReporter()
    reporter.render_profile(_profile_with_warnings())

    log = " ".join(line.plain for line in reporter.snapshot_log())
    assert "--verbose" not in log
    assert "RESULTS tab" in log


# -- the design system ------------------------------------------------------


def test_every_status_has_a_glyph_as_well_as_a_colour():
    """Colour is never the only signal.

    The found/absent/inconclusive/blocked distinction is the one the whole tool
    exists to make, and it is exactly the one a colour-blind operator loses.
    """
    for status in QueryStatus:
        style = status_style(status)
        assert style.glyph.strip(), f"{status} has no glyph"
        assert style.label.strip()

    glyphs = {style.glyph for style in STATUS_STYLES.values()}
    assert len(glyphs) == len(STATUS_STYLES), "two statuses share a glyph"


def test_absent_is_called_absent_not_available():
    """The enum answers a different question than the operator is asking.

    "Available" is true and means "free to register"; the operator asked "did I
    find them here", and the answer to that is "absent".
    """
    assert status_style(QueryStatus.AVAILABLE).label == "absent"
    assert status_style(QueryStatus.WAF).label == "blocked"


def test_counter_rows_are_a_fixed_width_so_digits_stack():
    """Right-aligned numbers in a fixed column are what makes the panel
    comparable at a glance rather than readable only line by line."""
    rows = [
        stat_row("found", 7, "green"),
        stat_row("absent", 400, "dim"),
        stat_row("inconclusive", 42, "yellow"),
    ]
    widths = {row.cell_len for row in rows}
    assert len(widths) == 1, f"counter rows are ragged: {widths}"
    # Every line ends with its digits, so the units column lines up.
    for row in rows:
        assert row.plain.rstrip() == row.plain


def test_a_glyph_nobody_has_been_taught_is_not_a_signal_either():
    """The key says, in words, what each symbol means.

    Every status carries a glyph so colour is never the only signal -- but the
    feed is the only place that teaches them, because `status_cell` prints the
    word beside the symbol on every row. A two-cell column in a table cannot do
    that, and an unexplained ▲ is a colour-only signal by another route.
    """
    for status in QueryStatus:
        style = status_style(status)
        rendered = status_key([status]).plain
        assert style.glyph in rendered
        assert style.label in rendered


def test_the_key_cannot_drift_from_the_cells_it_explains():
    """Built from `STATUS_STYLES`, not written out beside the table.

    A key that can disagree with the column above it is worse than no key: it
    is a wrong answer given confidently. Rename a status in the one table and
    the key renames with it, or this fails.
    """
    for status, style in STATUS_STYLES.items():
        assert f"{style.glyph} {style.label}" in status_key([status]).plain


def test_the_key_reads_the_same_way_every_time():
    """`STATUS_ORDER`, whatever order the rows happened to arrive in, and each
    symbol named once however many rows wear it."""
    scrambled = status_key(
        [QueryStatus.WAF, QueryStatus.CLAIMED, QueryStatus.WAF]
    ).plain
    assert scrambled == status_key([QueryStatus.CLAIMED, QueryStatus.WAF]).plain
    positions = [
        scrambled.index(status_style(s).label)
        for s in STATUS_ORDER
        if status_style(s).label in scrambled
    ]
    assert positions == sorted(positions)
    assert scrambled.count("blocked") == 1


def test_an_unknown_stored_status_does_not_take_down_the_pane():
    """A row written by an older version can hold a spelling this one dropped.

    Reading one is a display problem, and a display problem must not crash the
    pane reporting it -- the same rule `status_style` follows.
    """
    assert status_from_name("Claimed") is QueryStatus.CLAIMED
    assert status_from_name("Illegal") is QueryStatus.ILLEGAL
    assert status_from_name("Whatever") is QueryStatus.UNKNOWN
    assert status_from_name(None) is QueryStatus.UNKNOWN


def test_status_cell_is_text_not_markup():
    """Site data reaches these cells and Rich parses markup in them."""
    cell = status_cell(QueryStatus.CLAIMED)
    assert "found" in cell.plain
    assert cell.markup != cell.plain or "[" not in cell.plain


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (9.4, "9s"), (59, "59s"), (60, "1m00s"), (132, "2m12s"), (3700, "1h01m")],
)
def test_elapsed_reads_as_a_duration(seconds, expected):
    assert elapsed_label(seconds) == expected


# -- settings resolution ----------------------------------------------------


def test_absence_is_tested_with_is_none_not_falsiness():
    """A stored `False` is a choice, not an absence.

    The same trap the CLI resolver documents: testing falsiness makes
    "the user turned this off" indistinguishable from "nobody chose", and the
    default then overrides the user's own setting.
    """
    assert resolved({"scan.webbrowser": False}, "scan.webbrowser") is False
    # Nothing stored falls back to what the build ships, which is on.
    assert resolved({}, "scan.webbrowser") is True


def test_analysis_runs_only_when_a_model_is_configured():
    assert ai_is_configured({"ai.model": "vendor/m"}) is True
    assert ai_is_configured({"ai.model": None}) is False
    assert ai_is_configured({}) is False


def test_the_config_line_agrees_with_what_the_scan_will_do():
    """One predicate for "is a model available", used by both surfaces."""
    from sherlock_project.tui.reporter import analysis_is_on

    for values in ({"ai.model": "vendor/m"}, {"ai.model": None}, {}):
        assert analysis_is_on(values) == ai_is_configured(values)


def test_the_config_line_reports_a_missing_model_but_not_the_run_choice():
    """Whether analysis runs is a per-run toggle with its own visible state.

    Saying it here too would give one answer two places to be read from, and
    one of them would go stale. A missing model is different -- that is stored
    state, and it is the reason the toggle cannot help.
    """
    assert "no model configured" in describe_settings({})
    assert "no model configured" not in describe_settings({"ai.model": "vendor/m"})
    assert "AI analysis" not in describe_settings({"ai.model": "vendor/m"})


def test_the_config_line_names_the_setting_that_changes_what_a_result_means():
    """A browserless run can report a real account as absent. Someone reading a
    finding needs that visible where the finding was made."""
    assert "no browser" in describe_settings({"scan.webbrowser": False})
    assert "no browser" not in describe_settings({"scan.webbrowser": True})


def test_the_runner_reuses_the_cli_scan_lifecycle():
    """The rule the whole TUI is built on, made mechanical.

    `runner.py` imports these from `sherlock.py` INSIDE the scan function, so a
    rename would not surface until someone actually pressed SCAN -- by which
    point the failure is an ImportError in a worker, mid-run. This asserts every
    name still resolves and still takes the arguments the runner passes.

    It is also the guard on the rule itself: if one of these stops being
    importable, the tempting fix is to copy the logic into the runner, and that
    is exactly the drift this test exists to make loud.
    """
    import inspect

    from sherlock_project.ai_engine import pass_one_contract_hash
    from sherlock_project.sherlock import (
        _cancel_ai_pipeline_now,
        _close_ai_service_and_db,
        is_resumable,
        report_cached_ai_evidence,
        restore_saved_results,
        run_ai_pipeline,
        sherlock,
        synthesize_profiles,
    )

    assert callable(pass_one_contract_hash)

    # The keyword the runner actually passes, per function. A signature change
    # that dropped one of these would break the scan and nothing else would say
    # so until it ran.
    expected = {
        _cancel_ai_pipeline_now: {"ai_queue", "ai_pipeline_task"},
        _close_ai_service_and_db: {"ai_service", "db"},
        is_resumable: {"row", "using_browser"},
        report_cached_ai_evidence: {"db", "usernames", "contract_hash", "reporter"},
        restore_saved_results: {"username", "saved_rows", "site_data_all"},
        run_ai_pipeline: {"ai_queue", "sherlock_db", "reporter", "ai_settings"},
        synthesize_profiles: {"db", "ai_service", "usernames", "force", "reporter"},
        sherlock: {
            "username", "engine", "db", "site_data", "query_notify",
            "enqueue_ai", "proxy", "timeout", "on_cancel",
        },
    }
    for function, names in expected.items():
        parameters = set(inspect.signature(function).parameters)
        missing = names - parameters
        assert not missing, f"{function.__name__} no longer takes {missing}"


def test_the_tui_reporter_satisfies_everything_the_scan_calls():
    """Why this subclasses `TerminalReporter` and not the `QueryNotify` stub.

    The bare base is a no-op with none of these methods, so a reporter built on
    it dies with AttributeError partway through a run -- at whichever call site
    the scan happens to reach first, which varies by which sites answer.
    """
    reporter = TuiReporter()
    for name in (
        "info", "warning", "failure", "debug", "hint", "success",
        "browser_status", "browserless_transport", "restored_results",
        "ai_model_starting", "ai_scheduled", "ai_draining", "ai_pass_finished",
        "ai_extractions_from_other_models", "processing_interrupted", "close",
    ):
        assert callable(getattr(reporter, name)), f"reporter has no {name}"


# -- the database listing ---------------------------------------------------


async def test_listing_orders_by_when_it_was_actually_scanned(db: SherlockDB):
    """`usernames.last_scanned_at` records first-seen and is never updated.

    Sorting by it would put a username scanned this morning below one first seen
    last year and never touched since -- the same trap `get_username_overview`
    documents.
    """
    for username, site, status in (
        ("older", "A", QueryStatus.CLAIMED),
        ("newer", "B", QueryStatus.CLAIMED),
        ("newer", "C", QueryStatus.AVAILABLE),
    ):
        await db.save_result(
            username=username,
            site_name=site,
            site_url=f"https://{site}",
            status=str(status),
            status_code=200,
            query_time_ms=1.0,
            error_context=None,
            response_text=None,
        )

    listings = await db.list_usernames()
    names = [item.username for item in listings]
    assert set(names) == {"older", "newer"}

    by_name = {item.username: item for item in listings}
    assert by_name["newer"].total_sites == 2
    assert by_name["newer"].claimed_sites == 1
    assert by_name["older"].claimed_sites == 1
    assert by_name["older"].has_profile is False


async def test_a_username_with_no_results_is_not_offered(db: SherlockDB):
    """An interrupted scan can leave one, and its detail view would be empty."""
    await db.get_or_create_username_id("ghost")
    assert await db.list_usernames() == []


# -- the runner's lifecycle -------------------------------------------------


class _FakeEngine:
    """Stands in for a real engine so the lifecycle can be tested offline."""

    instances: ClassVar[list[_FakeEngine]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        _FakeEngine.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class _FakeSite:
    def __init__(self, name):
        self.name = name
        self.information = {"urlMain": f"https://{name}.test"}


async def _run_with_fakes(
    monkeypatch,
    settings_values,
    *,
    scan_spy,
    sites=(),
    fresh=False,
    use_ai=False,
    anchors=(),
):
    """Drive `run_scan_session` with the network and the browser removed.

    `sites` is empty by default, which is itself a case worth exercising -- it
    is what a fully resumed username looks like.
    """
    import sherlock_project.sherlock as sherlock_module
    from sherlock_project.tui import runner as runner_module

    _FakeEngine.instances.clear()
    monkeypatch.setattr(runner_module, "HttpEngine", _FakeEngine)
    monkeypatch.setattr(runner_module, "PlaywrightEngine", _FakeEngine)

    class _Sites:
        def __iter__(self):
            return iter([_FakeSite(name) for name in sites])

        def remove_nsfw_sites(self, do_not_remove):
            return None

    monkeypatch.setattr(runner_module, "SitesInformation", lambda **kw: _Sites())
    monkeypatch.setattr(sherlock_module, "sherlock", scan_spy)

    reporter = TuiReporter()
    await runner_module.run_scan_session(
        username="someone",
        reporter=reporter,
        settings_values=settings_values,
        fresh=fresh,
        use_ai=use_ai,
        anchors=anchors,
    )
    return reporter


async def test_a_browserless_setting_really_starts_no_browser(monkeypatch):
    """The setting that changes whether a result can be trusted.

    Not starting Chromium is the whole win of the fast path, so this asserts the
    choice reaches engine construction rather than only the label on screen.
    """
    called = []

    async def scan_spy(**kwargs):
        called.append(kwargs)
        return {}

    await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": False, "scan.concurrency": 10, "scan.timeout": 30},
        scan_spy=scan_spy,
        sites=("GitHub",),
    )

    assert len(_FakeEngine.instances) == 1
    engine = _FakeEngine.instances[0]
    # HttpEngine takes a plain proxy string; PlaywrightEngine takes a dict and a
    # status callback. The absence of those is what says which one was built.
    assert "status_callback" not in engine.kwargs
    assert engine.entered and engine.exited


async def test_the_engine_is_closed_even_when_the_scan_raises(monkeypatch):
    """A scan that fails must not leave the engine or the database open."""

    async def exploding_scan(**kwargs):
        raise RuntimeError("site loop fell over")

    with pytest.raises(RuntimeError):
        await _run_with_fakes(
            monkeypatch,
            {"scan.webbrowser": False},
            scan_spy=exploding_scan,
            sites=("GitHub",),
        )

    # `async with` unwinds on the way out, so the engine is released even though
    # the exception propagates to the worker that reports it.
    assert _FakeEngine.instances[0].exited is True


def test_restored_results_reach_the_counters_and_the_feed():
    """The pane must describe the USERNAME, not just this run.

    A resumed scan of a username with stored accounts showed `found 0` and an
    empty findings table, because only sites checked in this run reached the
    counters. That reads as "checked everything, found nothing" -- the exact
    opposite of the truth, and it is why a fully resumed run looked like it had
    re-scanned from scratch.
    """
    restored = {
        "GitHub": {"status": _result("GitHub", QueryStatus.CLAIMED)},
        "Nowhere": {"status": _result("Nowhere", QueryStatus.AVAILABLE)},
    }
    reporter = TuiReporter()
    reporter.restored_results(username="someone", results=restored, to_scan=1)

    assert reporter.counts[QueryStatus.CLAIMED] == 1
    # Both are retained -- the filter decides which get drawn.
    assert sorted(f.site_name for f in reporter.findings) == ["GitHub", "Nowhere"]
    # Marked, because evidence recalled and evidence just found are different
    # claims about the world.
    assert all("stored" in f.detail for f in reporter.findings)

    # `start` zeroes the parent's counters, so they have to survive it.
    reporter.start("someone", total=1)
    assert reporter.counts[QueryStatus.CLAIMED] == 1
    # 2 restored + 1 still to check: the bar spans the username, not the run.
    assert reporter.total == 3, "the bar must count restored sites too"

    # A live hit lands on top of the restored ones rather than replacing them.
    reporter.update(_result("Reddit", QueryStatus.CLAIMED))
    assert reporter.counts[QueryStatus.CLAIMED] == 2


def test_replaying_restored_results_does_not_flood_the_activity_log():
    """The parent logs a line per hit; several hundred would bury the run's own
    narration under a transcript of a scan that already happened."""
    restored = {
        f"Site{i}": {"status": _result(f"Site{i}", QueryStatus.CLAIMED)}
        for i in range(50)
    }
    reporter = TuiReporter()
    reporter.restored_results(username="someone", results=restored, to_scan=0)

    assert len(reporter.snapshot_log()) <= 2
    assert reporter.counts[QueryStatus.CLAIMED] == 50


async def test_no_browser_is_started_when_there_is_nothing_to_scan(monkeypatch):
    """A fully resumed username used to start Chromium, report it ready, then
    scan nothing -- seconds of browser startup for a run with no work in it,
    and on screen indistinguishable from a real scan."""
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.save_result(
            username="someone",
            site_name="GitHub",
            site_url="https://github.com/someone",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            query_time_ms=1.0,
            error_context=None,
            response_text=None,
            transport="browser",
        )
    finally:
        await db.close()

    async def scan_spy(**kwargs):
        return {}

    reporter = await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": True},
        scan_spy=scan_spy,
        sites=("GitHub",),
    )

    assert _FakeEngine.instances == [], "an engine was built with nothing to fetch"
    assert any(
        "already stored" in line.plain for line in reporter.snapshot_log()
    )


async def test_nothing_left_to_check_says_so_rather_than_scanning_zero_sites(
    monkeypatch,
):
    """Announcing "0 sites" under a list of restored results is the confusing
    shape the CLI avoids, so the runner says what actually happened."""
    calls = []

    async def scan_spy(**kwargs):
        calls.append(kwargs)
        return {}

    reporter = await _run_with_fakes(
        monkeypatch, {"scan.webbrowser": False}, scan_spy=scan_spy
    )

    # The fake manifest is empty and nothing is stored, so this is the one case
    # `restored_results` cannot report -- there is nothing to restore. Said
    # here instead, rather than leaving a run that did nothing unexplained.
    assert calls == []
    assert any("No sites to check" in line.plain for line in reporter.snapshot_log())


# -- the controls actually do something -------------------------------------


async def test_widget_state_never_shadows_a_textual_internal():
    """The bug this file did not catch until a human pressed the button.

    A widget subclass shares an attribute namespace with the framework.
    `ScanPane._running` collided with `MessagePump._running`, which Textual sets
    to True the moment the widget mounts -- so `action_run`'s "is a scan already
    going?" guard was True before anything had run, and SCAN and Enter did
    nothing at all, silently, forever.

    Checked against a bare container rather than by listing known names, so a
    field added to any pane later is covered without anyone remembering to.

    The names a pane assigns are read off its `__init__` bytecode rather than
    from a live instance: by the time a widget is mounted its `__dict__` is full
    of the framework's own attributes, so comparing instances would flag every
    one of them and prove nothing.
    """
    from textual.containers import Vertical

    framework = set(dir(Vertical()))
    for cls in (ScanPane, ResultsPane, SettingsPane):
        assigned = {
            name
            for name in cls.__init__.__code__.co_names
            if name.startswith("_") and not name.startswith("__")
        }
        clashes = assigned & framework
        assert not clashes, (
            f"{cls.__name__}.__init__ assigns {clashes}, which Textual already "
            f"uses. Pick a name of your own -- see ScanPane._scan_running."
        )


def _keys_claimed_by_widgets() -> set[str]:
    """Every key the widgets on screen already answer to."""
    from textual.app import App
    from textual.screen import Screen
    from textual.widgets import Button, DataTable, Input, RichLog, Tabs

    claimed: set[str] = set()
    for cls in (App, Screen, Input, DataTable, Tabs, Button, RichLog):
        for binding in getattr(cls, "BINDINGS", []):
            key = str(getattr(binding, "key", binding))
            claimed.update(part.strip() for part in key.split(","))
    return claimed


def test_navigation_keys_avoid_flow_control_and_function_keys():
    """Why the tabs are alt+digit rather than F1..F3 or ctrl+letter.

    - ctrl+s and ctrl+q are XON/XOFF. On a terminal with flow control still on
      they never reach the application, and the symptom is a key that silently
      does nothing.
    - Function keys need Fn held on most laptops, and F1 is commonly grabbed
      for help by the terminal or the desktop before the app sees it.

    `sherlock settings` keeps ctrl+s to save because it shipped that way and
    changing a key underneath people is worse than the caveat; this asserts
    nothing NEW picks one of these.
    """
    hostile = {"ctrl+s", "ctrl+q", "f1", "f2", "f3", "f4"}
    for owner in (SherlockUI, ScanPane, ResultsPane):
        for binding in owner.BINDINGS:
            for key in str(binding.key).split(","):
                assert key.strip() not in hostile, (
                    f"{owner.__name__} binds {key!r}, which is either terminal "
                    f"flow control or a function key -- see the note on "
                    f"SherlockUI.BINDINGS."
                )


def test_app_level_keys_are_not_already_taken_by_a_widget():
    """An app binding that a focused widget also claims never fires.

    The widget wins, so the key looks broken rather than conflicting -- which
    is exactly how STOP on ctrl+x came to do nothing at all.
    """
    claimed = _keys_claimed_by_widgets()
    for binding in SherlockUI.BINDINGS:
        for key in str(binding.key).split(","):
            assert key.strip() not in claimed, (
                f"SherlockUI binds {key!r}, which a widget on screen already "
                f"claims; the widget would win."
            )


def test_scan_bindings_are_keys_the_username_field_does_not_claim():
    """The other half of the STOP bug, generalised.

    The app opens with the username field focused, so any binding on this pane
    that `Input` also binds is swallowed before it arrives -- silently, with the
    field performing an edit instead. STOP was ctrl+x, which `Input` uses for
    cut, so it did nothing while a scan ran.
    """
    from textual.widgets import Input

    claimed: set[str] = set()
    for binding in Input.BINDINGS:
        claimed.update(str(getattr(binding, "key", binding)).split(","))

    for binding in ScanPane.BINDINGS:
        for key in str(binding.key).split(","):
            assert key not in claimed, (
                f"ScanPane binds {key!r}, which Input already uses -- it will "
                f"never fire while the username field has focus."
            )


async def _start_scan_and_capture(monkeypatch, press):
    """Drive a real control and report whether a scan was actually launched."""
    from sherlock_project.tui import runner as runner_module

    launched: list[str] = []

    async def fake_session(*, username, reporter, settings_values, **options):
        launched.append(username)
        reporter.start(username, total=1)

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"alice")
        await press(pilot, app)
        await pilot.pause()
        await pilot.pause()
    return launched


async def test_pressing_enter_starts_a_scan(monkeypatch):
    async def press(pilot, app):
        await pilot.press("enter")

    assert await _start_scan_and_capture(monkeypatch, press) == ["alice"]


async def test_clicking_the_scan_button_starts_a_scan(monkeypatch):
    async def press(pilot, app):
        await pilot.click("#scan-button")

    assert await _start_scan_and_capture(monkeypatch, press) == ["alice"]


async def test_the_run_binding_starts_a_scan(monkeypatch):
    async def press(pilot, app):
        await pilot.press("ctrl+r")

    assert await _start_scan_and_capture(monkeypatch, press) == ["alice"]


async def test_analysis_is_off_unless_asked_for(monkeypatch):
    """Matches `--ai`, which is opt-in on the command line.

    The UI used to run a local model automatically whenever one was configured,
    which meant the same investigation cost a GPU pass in the app and not on the
    CLI. A screen must not quietly do more than the command it stands for.
    """
    from sherlock_project.tui import runner as runner_module

    captured: dict[str, bool] = {}

    async def fake_session(*, username, reporter, settings_values, **options):
        captured["use_ai"] = options["use_ai"]
        captured["fresh"] = options["fresh"]

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"alice")
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
        # Nothing stored for this username, so no prompt and no re-scan.
        assert captured["use_ai"] is False
        assert captured["fresh"] is False

        # The analysis toggle reaches the scan.
        await pilot.click("#toggle-ai")
        await pilot.press("ctrl+r")
        for _ in range(10):
            await pilot.pause()
        assert captured["use_ai"] is True


async def test_the_toggles_show_their_own_state():
    """They report as much as they invite a press -- a control whose state is
    invisible is one you have to press to find out about."""
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        ai = app.query_one("#toggle-ai", Button)
        assert "off" in str(ai.label)
        await pilot.click("#toggle-ai")
        assert "on" in str(app.query_one("#toggle-ai", Button).label)


async def test_re_scan_defeats_the_resume_filter(monkeypatch, tmp_path):
    """Without this the UI can scan a username exactly once, ever.

    Every later run finds everything already stored and reports that there is
    nothing to do -- which is the dead end `--fresh` exists to open, and the
    most ordinary thing anyone wants is to check again later.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.save_result(
            username="someone",
            site_name="GitHub",
            site_url="https://github.com/someone",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            query_time_ms=1.0,
            error_context=None,
            response_text=None,
            transport="browser",
        )
    finally:
        await db.close()

    scanned: list[dict] = []

    async def scan_spy(**kwargs):
        scanned.append(kwargs)
        return {}

    # Resuming: the stored row satisfies the filter, so nothing is scanned.
    await _run_with_fakes(
        monkeypatch, {"scan.webbrowser": True}, scan_spy=scan_spy,
        sites=("GitHub",),
    )
    assert scanned == []

    # Re-scanning: the same site is checked again, and stored extractions are
    # redone rather than reused.
    await _run_with_fakes(
        monkeypatch, {"scan.webbrowser": True}, scan_spy=scan_spy,
        sites=("GitHub",), fresh=True,
    )
    assert len(scanned) == 1
    assert list(scanned[0]["site_data"]) == ["GitHub"]
    assert scanned[0]["force_ai_extraction"] is True


async def test_a_configured_model_going_unused_is_reported(monkeypatch):
    """The gap that made changing model look broken.

    Analysis is off by default on every run, matching `--ai`. Someone who has
    just been to SETTINGS to choose a model has signalled they want it, then
    scans, and nothing runs -- no extraction, and not even the stale-extraction
    warning, because that is only reported when analysis is ON. Silence, and the
    old model still shown on the profile.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.save_result(
            username="someone",
            site_name="GitHub",
            site_url="https://github.com/someone",
            status=str(QueryStatus.CLAIMED),
            status_code=200,
            query_time_ms=1.0,
            error_context=None,
            # Page text stored, so this row is genuinely eligible.
            response_text="<html>hello</html>",
            transport="browser",
        )
    finally:
        await db.close()

    async def scan_spy(**kwargs):
        return {}

    reporter = await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": True, "ai.model": "vendor/m"},
        scan_spy=scan_spy,
        sites=("GitHub",),
        use_ai=False,
    )
    log = " ".join(line.plain for line in reporter.snapshot_log())
    assert "Analysis is off" in log
    assert "1 sites with stored pages" in log
    assert "OPTIONS" in log


async def test_no_such_warning_when_there_is_nothing_to_analyse(monkeypatch):
    """Otherwise it becomes noise on every fast scan, and stops being read."""
    async def scan_spy(**kwargs):
        return {}

    reporter = await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": False, "ai.model": "vendor/m"},
        scan_spy=scan_spy,
        use_ai=False,
    )
    assert not any(
        "Analysis is off" in line.plain for line in reporter.snapshot_log()
    )


async def test_asking_for_analysis_without_a_model_says_so(monkeypatch):
    """A scan that silently produced no profile would look like the model
    failed; the actual problem is one unset setting."""
    async def scan_spy(**kwargs):
        return {}

    reporter = await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": False, "ai.model": None},
        scan_spy=scan_spy,
        use_ai=True,
    )
    assert any(
        "no model is configured" in line.plain.lower()
        for line in reporter.snapshot_log()
    )


async def test_an_empty_username_is_refused_rather_than_scanned(monkeypatch):
    async def press(pilot, app):
        await pilot.press("enter")

    from sherlock_project.tui import runner as runner_module

    launched: list[str] = []

    async def fake_session(*, username, reporter, settings_values, **options):
        launched.append(username)

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)
    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
    assert launched == []


async def test_stopping_a_scan_leaves_other_work_alone(monkeypatch):
    """STOP cancels the scan, not everything running in the app.

    `self.workers` is the app's manager, not the widget's, so cancelling all of
    it would take the results pane's database reads down with the scan.
    """
    import asyncio

    from textual.worker import WorkerState

    from sherlock_project.tui import runner as runner_module

    async def slow_session(*, username, reporter, settings_values, **options):
        await asyncio.sleep(60)

    monkeypatch.setattr(runner_module, "run_scan_session", slow_session)

    app = SherlockUI()
    async with app.run_test() as pilot:
        bystander = app.run_worker(asyncio.sleep(60), group="bystander")
        await pilot.press(*"alice")
        await pilot.press("enter")
        await pilot.pause()

        pane = app.query_one(ScanPane)
        assert pane._scan_running is True

        await pilot.press("escape")
        # Cancellation is not instant: `cancel()` requests it, and the
        # CANCELLED state change only arrives once the task has actually
        # unwound. A single pause races that.
        for _ in range(10):
            if not pane._scan_running:
                break
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert pane._scan_running is False
        assert bystander.state is not WorkerState.CANCELLED
        bystander.cancel()


# -- the findings filter ----------------------------------------------------


async def _pane_with_results():
    """An app whose scan pane holds one result of every kind."""
    app = SherlockUI()
    pane = None
    async with app.run_test(size=(110, 34)) as pilot:
        pane = app.query_one(ScanPane)
        reporter = TuiReporter()
        reporter.start("someone", total=5)
        for site, status in (
            ("Hit", QueryStatus.CLAIMED),
            ("Nowhere", QueryStatus.AVAILABLE),
            ("Slow", QueryStatus.UNKNOWN),
            ("Walled", QueryStatus.WAF),
            ("Bad", QueryStatus.ILLEGAL),
        ):
            reporter.update(_result(site, status))
        pane._reporter = reporter
        pane._flush()
        await pilot.pause()
        yield app, pane, pilot


async def test_the_filter_works_before_any_scan_has_run():
    """The state the app opens in, and the one every earlier test skipped.

    `_rebuild_feed` redrew the title unconditionally but guarded the counters
    behind "is there a reporter", so before the first scan the title updated and
    the strikethrough did not -- the filter visibly worked in one place and
    looked broken in the other. Every other filter test seeded a reporter first,
    which is exactly why none of them caught it.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        pane = app.query_one(ScanPane)
        assert pane._reporter is None, "this test is about the pre-scan state"

        blocked = app.query_one("#count-waf", CounterRow)
        assert any("strike" in str(span.style) for span in blocked.render().spans)

        await pilot.click("#count-waf")
        await pilot.pause()

        assert QueryStatus.WAF in pane._visible
        after = app.query_one("#count-waf", CounterRow).render()
        assert not any("strike" in str(span.style) for span in after.spans), (
            "the row is in the filter but still drawn as excluded"
        )

        # And back off again, still with no reporter.
        await pilot.click("#count-waf")
        await pilot.pause()
        again = app.query_one("#count-waf", CounterRow).render()
        assert any("strike" in str(span.style) for span in again.spans)


async def test_clicking_a_counter_shows_that_kind_in_the_feed():
    """The counters are the filter -- one place answers "how many" and "which"."""
    from textual.widgets import DataTable

    async for app, pane, pilot in _pane_with_results():
        table = app.query_one("#feed", DataTable)
        assert table.row_count == 1, "only the hit should be drawn by default"

        await pilot.click("#count-waf")
        await pilot.pause()
        assert table.row_count == 2
        assert QueryStatus.WAF in pane._visible

        # Clicking again hides it, so the control is a real toggle.
        await pilot.click("#count-waf")
        await pilot.pause()
        assert table.row_count == 1


async def test_a_hidden_kind_is_struck_through_not_removed():
    """The count has to stay exact and readable while its rows are hidden.

    What the panel counts and what the feed displays are two different facts,
    and only the second one is being toggled -- so the number never changes,
    it is just struck through.
    """
    async for app, pane, pilot in _pane_with_results():
        absent = app.query_one("#count-available", CounterRow)
        hit = app.query_one("#count-claimed", CounterRow)

        struck = absent.render()
        shown = hit.render()
        assert any("strike" in str(span.style) for span in struck.spans)
        assert not any("strike" in str(span.style) for span in shown.spans)
        # The number is still there and still right.
        assert "1" in struck.plain

        await pilot.click("#count-available")
        await pilot.pause()
        after = app.query_one("#count-available", CounterRow).render()
        assert not any("strike" in str(span.style) for span in after.spans)


async def test_the_filter_cannot_be_emptied():
    """Turning the last visible kind off leaves a blank table with no reason
    given, and "hide everything" is never what one click means."""
    from textual.widgets import DataTable

    async for app, pane, pilot in _pane_with_results():
        await pilot.click("#count-claimed")
        await pilot.pause()

        assert pane._visible == {QueryStatus.CLAIMED}
        assert app.query_one("#feed", DataTable).row_count == 1


async def test_one_key_flips_between_everything_and_hits_only():
    """The two ends of the filter are what anyone wants in a hurry, and
    clicking four rows is four chances to lose track of which are on."""
    from textual.widgets import DataTable

    async for app, pane, pilot in _pane_with_results():
        table = app.query_one("#feed", DataTable)

        await pilot.press("alt+f")
        await pilot.pause()
        assert pane._visible == set(QueryStatus)
        assert table.row_count == 5

        await pilot.press("alt+f")
        await pilot.pause()
        assert pane._visible == {QueryStatus.CLAIMED}
        assert table.row_count == 1


async def test_a_filtered_feed_says_it_is_filtered():
    """A filtered table that does not say so is how someone concludes a scan
    found nothing."""
    from textual.widgets import Static

    async for app, pane, pilot in _pane_with_results():
        title = app.query_one("#feed-title", Static).render().plain
        assert "showing found" in title

        await pilot.press("alt+f")
        await pilot.pause()
        # Nothing is being filtered out, so there is nothing to qualify.
        assert "showing" not in (
            app.query_one("#feed-title", Static).render().plain
        )


# -- the resume prompt ------------------------------------------------------


async def _run_with_prompt(monkeypatch, username: str):
    """Press SCAN and hand back the app plus what the scan was told."""
    from sherlock_project.tui import runner as runner_module

    captured: dict = {}

    async def fake_session(*, username, reporter, settings_values, **options):
        captured.update(options)
        captured["username"] = username

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)
    return captured


async def test_an_unknown_username_scans_without_asking(monkeypatch):
    """A prompt that appears every time gets dismissed without reading, and
    there is nothing to warn about on a first scan."""
    captured = await _run_with_prompt(monkeypatch, "nobody")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"nobody")
        await pilot.press("enter")
        for _ in range(15):
            await pilot.pause()

        assert not isinstance(app.screen, ModalScreen)
        assert captured.get("username") == "nobody"
        assert captured.get("fresh") is False


async def test_a_known_username_asks_before_scanning_again(monkeypatch):
    """The resume state used to be invisible until after the run.

    A scan of a fully-stored username skipped every site, said so only in the
    activity log, and otherwise looked exactly like a scan that found nothing.
    """
    from sherlock_project.tui.resume_screen import ResumeScreen

    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])
    captured = await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"marcus")
        await pilot.press("enter")
        for _ in range(15):
            await pilot.pause()

        assert isinstance(app.screen, ResumeScreen)
        # The counts are the content -- "scanned before, continue?" is a
        # question nobody can answer.
        detail = app.screen.query_one("#resume-detail").render().plain
        assert "1 result already stored" in detail
        # Nothing has run yet.
        assert captured == {}

        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
        assert captured == {}, "cancelling must start nothing"


async def test_choosing_re_scan_passes_fresh(monkeypatch):
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])
    captured = await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"marcus")
        await pilot.press("enter")
        for _ in range(15):
            await pilot.pause()

        await pilot.click("#resume-fresh")
        for _ in range(15):
            await pilot.pause()

    assert captured.get("fresh") is True


async def test_nothing_left_to_check_offers_the_results_instead(monkeypatch):
    """Resuming would check zero sites, so offering "Resume" would be a button
    that does not do what its label says."""
    from textual.widgets import Button

    from sherlock_project.sites import SitesInformation
    from sherlock_project.tui import runner as runner_module

    # Every site in the manifest already answered for this username.
    sites = SitesInformation(honor_exclusions=False)
    sites.remove_nsfw_sites(do_not_remove=[])
    stored = [(site.name, QueryStatus.AVAILABLE) for site in sites]
    await _seed(marcus=stored)

    captured: dict = {}

    async def fake_session(**options):
        captured.update(options)

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"marcus")
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()

        detail = app.screen.query_one("#resume-detail").render().plain
        assert "Nothing is left to check" in detail
        # No "Resume" button at all; the useful action is to go and look.
        assert not app.screen.query("#resume-continue")
        assert app.screen.query_one("#resume-view", Button)

        await pilot.click("#resume-view")
        for _ in range(20):
            await pilot.pause()

        from textual.widgets import TabbedContent

        assert app.query_one(TabbedContent).active == "tab-results"
        assert captured == {}, "viewing results must not start a scan"


def test_the_dialog_and_the_scan_share_one_resume_calculation():
    """The numbers offered and the work done must come from one function.

    Two implementations of the resume filter is how a dialog promises "39 left"
    and the scan then checks a different number.
    """
    import inspect

    from sherlock_project.tui import runner as runner_module

    session = inspect.getsource(runner_module.run_scan_session)
    assert "build_scan_plan(" in session
    assert "is_resumable(" not in session, (
        "run_scan_session re-implements the resume filter instead of using "
        "build_scan_plan"
    )
    peek = inspect.getsource(runner_module.peek_stored)
    assert "build_scan_plan(" in peek


# -- anchors ----------------------------------------------------------------


def test_anchor_lines_stack_into_one_column():
    """Values must start in the same cell on every row.

    Sizing the field column to each field's own length is the obvious thing and
    it is wrong -- `full_name`, `location` and `email` then began their values
    at three different columns, which is the ragged edge this design keeps
    fixed widths to avoid.
    """
    from sherlock_project.profile_synthesis import IdentityAnchor
    from sherlock_project.tui.theme import ANCHOR_FIELD_WIDTH, anchor_line

    lines = [
        anchor_line(
            IdentityAnchor(field=field, value=value, trust=trust), width=24
        )
        for field, value, trust in (
            ("full_name", "Avery Stone", "strong"),
            ("location", "Berlin", "context"),
            ("a_very_long_field_name", "x", "verified"),
        )
    ]
    # The value starts at the same offset on every row, whatever the field.
    starts = {2 + ANCHOR_FIELD_WIDTH + 1 for _ in lines}
    assert len(starts) == 1
    for line in lines:
        assert line.cell_len <= 24, f"anchor line overflows its column: {line.plain!r}"
    # An over-long field is cut rather than allowed to shove its value along.
    assert "…" in lines[2].plain


async def test_the_anchor_section_appears_only_with_analysis_on():
    """It feeds the AI's second pass and nothing else, so with analysis off it
    is a control that cannot do anything."""
    from textual.containers import Vertical

    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        block = app.query_one("#anchors-block", Vertical)
        assert block.display is False

        await pilot.click("#toggle-ai")
        await pilot.pause()
        assert block.display is True

        # The empty state teaches rather than sitting blank -- this is the only
        # place the app says why anchors matter.
        from textual.widgets import Static

        assert "mix different people" in (
            app.query_one("#anchors-list", Static).render().plain
        )
        assert pane._use_ai is True


async def test_hidden_anchors_cannot_reach_a_scan(monkeypatch):
    """State you cannot see must not change what a run does.

    Anchors survive analysis being toggled off, so they are still there when it
    comes back on -- but while it is off they are not passed at all, rather than
    passed and merely ignored downstream.
    """
    from sherlock_project.profile_synthesis import IdentityAnchor
    from sherlock_project.tui import runner as runner_module

    captured: dict = {}

    async def fake_session(*, username, reporter, settings_values, **options):
        captured.update(options)

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)

    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._anchors = [IdentityAnchor(field="full_name", value="Avery Stone")]

        await pilot.press(*"alice")
        await pilot.press("enter")
        await pilot.pause()
        assert captured["anchors"] == [], "hidden anchors reached the scan"

        # Turned on, the same anchors are used rather than needing retyping.
        await pilot.click("#toggle-ai")
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert [a.field for a in captured["anchors"]] == ["full_name"]


async def test_anchors_can_be_added_and_removed():
    """The feature that turns "accounts sharing a username" into "a person"."""
    from textual.widgets import Input

    from sherlock_project.tui.anchor_screen import AnchorScreen

    app = SherlockUI()
    async with app.run_test() as pilot:
        collected: list = []
        app.push_screen(AnchorScreen([]), collected.append)
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, AnchorScreen)

        screen.query_one("#anchor-field", Input).value = "full_name"
        screen.query_one("#anchor-value", Input).value = "Avery Stone"
        screen.action_add()
        await pilot.pause()

        assert [(a.field, a.value) for a in screen._anchors] == [
            ("full_name", "Avery Stone")
        ]
        # The boxes clear, so a second anchor is typed rather than edited over.
        assert screen.query_one("#anchor-field", Input).value == ""

        screen.action_remove()
        await pilot.pause()
        assert screen._anchors == []


async def test_an_incomplete_anchor_says_which_box_is_empty():
    """Repeating pydantic at someone is not an error message."""
    from textual.widgets import Input, Static

    from sherlock_project.tui.anchor_screen import AnchorScreen

    app = SherlockUI()
    async with app.run_test() as pilot:
        app.push_screen(AnchorScreen([]))
        await pilot.pause()
        screen = app.screen

        screen.query_one("#anchor-field", Input).value = "full_name"
        screen.action_add()
        await pilot.pause()

        assert "value" in screen.query_one("#anchor-status", Static).render().plain
        assert screen._anchors == []


def test_the_editor_does_not_ask_for_a_trust_level():
    """It was a three-way choice that changed nothing observable.

    Trust never reaches the model -- `_format_anchors` reduces anchors to
    `{field: [values]}` before the prompt -- and nothing branches on it:
    `has_trusted_anchor` is the only code separating the levels and has no
    callers. Two of the three words were treated as identical by that very
    function. `AnchorTrust` stays in the model so stored profiles and
    `--anchor verified:f=v` keep working; the screen just stops asking.
    """
    import inspect

    from sherlock_project.tui import anchor_screen

    source = inspect.getsource(anchor_screen.AnchorScreen)
    assert "anchor-trust" not in source
    assert not hasattr(anchor_screen, "TRUST_LEVELS")

    # Anchors it makes carry the model's own default, unchosen and unprinted.
    from sherlock_project.notify import DEFAULT_ANCHOR_TRUST
    from sherlock_project.profile_synthesis import IdentityAnchor

    assert str(IdentityAnchor(field="f", value="v").trust) == DEFAULT_ANCHOR_TRUST


async def test_leaving_the_editor_untouched_keeps_the_existing_anchors():
    """Dismissing with an empty list and dismissing with nothing are different.

    One clears the anchors, the other leaves them alone -- and Escape has to
    mean the second, or opening the editor to look would wipe the run's setup.
    """
    from sherlock_project.profile_synthesis import IdentityAnchor
    from sherlock_project.tui.anchor_screen import AnchorScreen

    existing = [IdentityAnchor(field="full_name", value="Avery Stone")]
    app = SherlockUI()
    async with app.run_test() as pilot:
        result: list = []
        app.push_screen(AnchorScreen(existing), result.append)
        await pilot.pause()
        app.screen.action_close()
        await pilot.pause()

    assert result == [None]
    # And the caller's own list was never edited in place.
    assert [a.field for a in existing] == ["full_name"]


async def test_anchors_set_without_analysis_are_reported(monkeypatch):
    """Typed, stored for the run, and never used is the silent failure here.

    The profile would come back carrying the very caveat the anchors were meant
    to remove, with nothing on screen explaining why.
    """
    from sherlock_project.profile_synthesis import IdentityAnchor

    async def scan_spy(**kwargs):
        return {}

    reporter = await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": False},
        scan_spy=scan_spy,
        anchors=[IdentityAnchor(field="full_name", value="Avery Stone")],
    )
    assert any(
        "anchor" in line.plain.lower() and "off" in line.plain.lower()
        for line in reporter.snapshot_log()
    )


# -- the app shell ----------------------------------------------------------


async def test_the_app_opens_on_scan_with_three_tabs():
    """Tab order is the order of a session: scan, then read, then adjust."""
    app = SherlockUI()
    async with app.run_test() as pilot:
        from textual.widgets import TabbedContent, TabPane

        tabs = app.query_one(TabbedContent)
        assert tabs.active == "tab-scan"

        # TabPane, not ".-content-tab". The latter is an internal Textual class
        # that matches nothing in 8.x, and it sat behind an `or [...]` fallback
        # naming the three ids by hand -- so an empty query silently substituted
        # the literal and the length check compared it to itself. It asserted
        # the tabs existed while being unable to observe whether they did.
        assert [pane.id for pane in tabs.query(TabPane)] == [
            "tab-scan",
            "tab-results",
            "tab-settings",
        ]

        # pause() after each press, because press() only awaits `_wait_for_screen`
        # and the binding's action can still be sitting on the app's queue when
        # the next line runs. Only pause() adds the `wait_for_idle` that drains
        # it. This is the suite's convention everywhere else; the two presses
        # below were the exception, and macOS CI is where the race finally
        # showed -- the session-scoped browser fixture from the Playwright tests
        # is still alive on the same session-scoped event loop, so the pump has
        # company.
        await pilot.press("alt+2")
        await pilot.pause()
        assert tabs.active == "tab-results"
        await pilot.press("alt+3")
        await pilot.pause()
        assert tabs.active == "tab-settings"


async def test_function_keys_switch_tabs_because_digits_would_not():
    """The scan pane has a text field. A binding on "1" would mean a username
    with a digit in it changes tab instead of typing.

    Also asserts the app opens focused on that field: no `.focus()` here, the
    keys are simply typed. A screen that opens with focus somewhere unhelpful
    makes the first keystroke a guess -- and a pane grabbing focus for itself on
    mount used to swallow it outright.
    """
    app = SherlockUI()
    async with app.run_test() as pilot:
        from textual.widgets import Input, TabbedContent

        await pilot.press("u", "1", "2")
        # Same pause, and it matters more here than it looks: this asserts a tab
        # switch did NOT happen, so without settling first the test would pass
        # by observing the app too early rather than by the digits being typed
        # into the field.
        await pilot.pause()
        assert app.query_one("#target-input", Input).value == "u12"
        assert app.query_one(TabbedContent).active == "tab-scan"


async def test_the_feed_draws_what_the_reporter_buffered():
    """The other half of the buffering argument: a tick draws everything that
    arrived since the last one, in one pass."""
    app = SherlockUI()
    async with app.run_test():
        from textual.widgets import DataTable

        pane = app.query_one(ScanPane)
        reporter = TuiReporter()
        reporter.start("someone", total=3)
        pane._reporter = reporter

        reporter.update(_result("GitHub", QueryStatus.CLAIMED))
        reporter.update(_result("Nowhere", QueryStatus.AVAILABLE))
        reporter.update(_result("Reddit", QueryStatus.WAF))

        pane._flush()

        table = app.query_one("#feed", DataTable)
        # One row, not three: all three were counted and kept, but the default
        # filter draws only hits.
        assert table.row_count == 1


async def test_the_title_stops_saying_scanning_when_the_scan_stops():
    """A live-looking label must not outlive the activity it describes.

    The verb was set when the scan started and never cleared, so a finished run
    went on claiming to be in progress for as long as the app stayed open.
    """
    from textual.widgets import Static

    app = SherlockUI()
    async with app.run_test():
        pane = app.query_one(ScanPane)
        pane._reset_for("0day")

        title = app.query_one("#feed-title", Static)
        assert "scanning 0day" in title.render().plain

        pane._scan_running = False
        pane._redraw_feed_title()

        rendered = app.query_one("#feed-title", Static).render().plain
        assert "scanning" not in rendered
        # The username stays -- these are still 0day's results.
        assert "0day" in rendered


async def test_findings_and_activity_are_separated_by_a_rule():
    """They share a column with nothing between them, which is the one boundary
    on this screen a heading alone did not make obvious."""
    app = SherlockUI()
    async with app.run_test(size=(110, 34)):
        title = app.query_one("#activity-title")
        assert title.styles.border_top[0], "no rule above ACTIVITY"


async def test_an_empty_scan_pane_says_idle_rather_than_pretending_to_work():
    """Nothing started must not look like something in progress."""
    app = SherlockUI()
    async with app.run_test():
        from textual.widgets import Static

        assert "idle" in app.query_one("#progress-strip", Static).render().plain


def test_the_progress_bar_is_exactly_as_wide_as_it_is_told():
    """The reason it is drawn rather than delegated to a widget.

    A compound progress widget sizes its own parts and collapsed to zero inside
    a one-row strip, so the bar simply was not there. Fixed cells also mean the
    line does not grow by one every time the completed figure gains a digit.
    """
    for completed, total in ((0, 680), (412, 680), (680, 680), (5, 0)):
        assert progress_bar(completed, total, 40).cell_len == 40

    # Filled and unfilled are different CHARACTERS, not one character in two
    # colours -- colour alone would make every bar look finished on a monochrome
    # terminal, in a screenshot, or to a colour-blind reader.
    assert progress_bar(0, 680, 40).plain == "─" * 40
    assert progress_bar(680, 680, 40).plain == "━" * 40
    half = progress_bar(340, 680, 40)
    assert half.plain.count("━") == 20
    assert half.plain.count("─") == 20
    # Nothing started, and no division by a zero total.
    assert progress_bar(5, 0, 10).plain == "─" * 10


async def test_saving_settings_updates_what_the_scan_pane_claims_it_will_do():
    """A concurrency change that only took effect after a restart would be a
    silent lie about what the next scan does."""
    app = SherlockUI()
    async with app.run_test() as pilot:
        from textual.widgets import Static

        await pilot.press("alt+3")
        pane = app.query_one("#scan-config", Static)
        before = pane.render().plain

        app._settings_values["scan.webbrowser"] = False
        app.query_one(ScanPane).refresh_settings(app._settings_values)

        after = app.query_one("#scan-config", Static).render().plain
        assert "no browser" in after
        assert after != before


async def test_the_results_tab_reloads_when_you_switch_to_it():
    """Every tab's content mounts at app start, not on first view.

    So a list loaded in `on_mount` is loaded before the scan that fills it. The
    username you just finished scanning was missing from the tab whose whole job
    is showing what has been scanned, and only a manual refresh brought it back.
    """
    from textual.widgets import DataTable

    from sherlock_project.database import SherlockDB, default_database_path

    app = SherlockUI()
    async with app.run_test() as pilot:
        table = app.query_one("#username-list", DataTable)
        assert table.row_count == 0

        # Something lands in the database while the app is already open, which
        # is exactly what finishing a scan on the first tab does.
        db = await SherlockDB.create(str(default_database_path()))
        try:
            await db.save_result(
                username="freshly-scanned",
                site_name="GitHub",
                site_url="https://github.com/freshly-scanned",
                status=str(QueryStatus.CLAIMED),
                status_code=200,
                query_time_ms=1.0,
                error_context=None,
                response_text=None,
            )
        finally:
            await db.close()

        await pilot.press("alt+2")
        for _ in range(10):
            if table.row_count:
                break
            await pilot.pause()

        assert table.row_count == 1


async def _seed(**rows) -> None:
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for username, entries in rows.items():
            for site, status in entries:
                await db.save_result(
                    username=username,
                    site_name=site,
                    site_url=f"https://{site.lower()}.example.com/{username}",
                    status=str(status),
                    status_code=200,
                    query_time_ms=1.0,
                    error_context="timed out" if status is QueryStatus.UNKNOWN else None,
                    response_text=None,
                    transport="browser",
                )
    finally:
        await db.close()


async def test_a_section_tab_and_its_pane_have_different_ids():
    """`query_one` returns whichever the walk reaches first, not what you meant.

    The section tabs and the panes they select started out sharing one id each,
    which made `query_one("#sec-accounts")` ambiguous -- it happened to work, by
    walk order, until a test asked for the other one and got a `DataTable` where
    it wanted a `Tab`.

    Framework-internal ids are not checked: Textual reuses `tabs-scroll` inside
    every `Tabs` and scopes ids per parent, so policing those would fail on the
    library's own structure rather than on anything this code decides.
    """
    from textual.widgets import Tab

    from sherlock_project.tui.results_pane import SECTION_FOR_TAB

    app = SherlockUI()
    async with app.run_test():
        for tab_id, section_id in SECTION_FOR_TAB.items():
            assert tab_id != section_id
            # Both resolve, each to the type the code expects of it.
            assert isinstance(app.query_one(f"#{tab_id}", Tab), Tab)
            app.query_one(f"#{section_id}")


async def test_focus_does_not_restyle_the_section_tabs():
    """They must look like tabs, not like a selected list row.

    Textual's default paints the active tab with the block cursor once the
    strip has focus -- reversed colours across the whole label -- so the strip
    changed appearance depending on where focus happened to be, and the active
    section read as a highlighted selection rather than an open tab.

    Compares the rendered strip with and without focus: identical styling is
    the property, whatever the theme's colours happen to be.
    """
    from textual.widgets import Tabs

    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])

    def strip_styles(app):
        rendered = []
        for strip in app.screen._compositor.render_strips():
            text = "".join(segment.text for segment in strip)
            if "ACCOUNTS" not in text:
                continue
            # Only the tab labels. The row also carries the border between the
            # master list and the detail, and that cell legitimately changes
            # shade when the list beside it loses focus -- a different pane's
            # focus ring, not this strip's styling.
            for segment in strip:
                if any(
                    word in segment.text
                    for word in ("ACCOUNTS", "UNRESOLVED", "PROFILE")
                ):
                    rendered.append(
                        (
                            segment.text,
                            str(segment.style.color),
                            str(segment.style.bgcolor),
                        )
                    )
            break
        return rendered

    app = SherlockUI()
    async with app.run_test(size=(100, 26)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()
        blurred = strip_styles(app)

        app.query_one("#detail-tabs", Tabs).focus()
        for _ in range(6):
            await pilot.pause()
        focused = strip_styles(app)

    assert blurred, "did not find the section strip"
    assert blurred == focused, (
        "focusing the section strip changed how it is drawn -- Textual's block "
        "cursor is painting the active tab like a selected row"
    )


async def test_the_detail_pane_has_exactly_one_scroll_region():
    """Two scrollbars side by side is what made this pane look unfinished.

    A `DataTable` is itself a scrolling viewport, so nesting one inside a
    scrolling column produced two vertical bars and, once URLs got long, a
    horizontal one as well. Sections are switched now, so only one scrolls.
    """
    await _seed(marcus=[(f"Site{i:02d}", QueryStatus.CLAIMED) for i in range(40)])

    app = SherlockUI()
    async with app.run_test(size=(100, 26)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()

        detail = app.query_one("#result-detail")
        scrolling = [
            widget
            for widget in detail.query("*")
            if widget.show_vertical_scrollbar or widget.show_horizontal_scrollbar
        ]
        assert len(scrolling) <= 1, (
            f"detail pane has {len(scrolling)} scroll regions: "
            f"{[type(w).__name__ + '#' + str(w.id) for w in scrolling]}"
        )
        # And never a horizontal one -- long URLs ellipsize instead.
        assert not any(w.show_horizontal_scrollbar for w in scrolling)


async def test_the_section_labels_carry_their_counts():
    """The one thing switching would otherwise hide.

    An account list means something different once six sites gave no answer,
    and behind a bare tab that number is invisible until someone thinks to
    look. So it goes on the tab.
    """
    from textual.widgets import Tab

    await _seed(
        marcus=[
            ("GitHub", QueryStatus.CLAIMED),
            ("Reddit", QueryStatus.CLAIMED),
            ("Slow", QueryStatus.UNKNOWN),
        ]
    )
    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()

        assert "2" in str(app.query_one("#tab-accounts", Tab).label)
        assert "1" in str(app.query_one("#tab-unresolved", Tab).label)


async def test_unresolved_sites_are_listed_not_only_counted():
    """There was no in-app equivalent of `show --unresolved`, and a UI needs one
    more than the CLI does -- there is no pipe to fall back on."""
    from textual.widgets import DataTable

    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)]
    )
    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()

        await pilot.press("alt+right")
        for _ in range(5):
            await pilot.pause()

        from textual.widgets import ContentSwitcher

        assert app.query_one("#detail-switch", ContentSwitcher).current == "sec-unresolved"
        assert app.query_one("#sec-unresolved", DataTable).row_count == 1


def _legend_text(app) -> str:
    """What the key is actually SHOWING.

    Its `display`, not just its content: a key that is correct and not on
    screen explains exactly as much as no key at all, and reading only the
    renderable would let it be hidden without a single test noticing.
    """
    from textual.widgets import Static

    legend = app.query_one("#detail-legend", Static)
    return legend.render().plain if legend.display else ""


async def test_the_symbol_column_comes_with_a_key():
    """Nothing on this screen said what ▲ meant.

    The mark column is two cells wide and its header is blank, so the
    distinction the whole tool exists to make -- "the site blocked us" is not
    "the rules did not decide" -- was being drawn in symbols the operator had
    never been shown a glossary for. The feed on the scan pane teaches its own
    because it prints the word beside every symbol; a table has no room for
    that, so the sentence goes above the column instead.
    """
    await _seed(
        marcus=[
            ("GitHub", QueryStatus.CLAIMED),
            ("Slow", QueryStatus.UNKNOWN),
            ("Cloudflared", QueryStatus.WAF),
        ]
    )
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        key = _legend_text(app)
        for glyph, word in (("?", "inconclusive"), ("▲", "blocked")):
            assert f"{glyph} {word}" in key, f"the key does not explain {glyph}"


async def test_the_key_names_what_is_on_screen_and_nothing_else():
    """A key offering `rejected` when nothing was rejected is noise in a line
    whose whole job is to be short enough to read in passing."""
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)])
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        key = _legend_text(app)
        assert "inconclusive" in key
        assert "blocked" not in key
        assert "rejected" not in key


async def test_the_key_costs_a_row_only_where_the_symbols_contend():
    """One row of this pane is a finding not shown, so the key has to earn it.

    UNRESOLVED earns it: three symbols contend there, and telling them apart is
    the tool's central claim. ACCOUNTS does not -- every row is a hit, so its
    mark column is one symbol repeated down a list the tab already calls
    ACCOUNTS, and a key there would spend a row to disambiguate nothing. Nor
    does PROFILE, which draws no symbols at all.

    This is also what keeps the line off the tab row, where the spare width
    looks like it is: `Tabs` is a scrolling strip, so a key sharing that row
    drops tabs rather than wrapping -- silently, and from the left.
    """
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)])
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()

        assert _legend_text(app) == "", (
            "ACCOUNTS is one symbol repeated and is still paying a row for a "
            "key that disambiguates nothing"
        )

        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()
        assert "inconclusive" in _legend_text(app)

        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()
        assert _legend_text(app) == "", (
            "the profile section has no symbols and is still reserving a row "
            "for a key"
        )


async def test_a_rejected_username_is_not_drawn_as_inconclusive():
    """The symbol came from a prefix match on its own explanation.

    Anything whose reason did not start with "blocked" was drawn with the
    inconclusive `?`, so a username the site's own rules reject -- which `show`
    reports as "username format rejected" and which has its own ✕ -- arrived
    wearing the symbol for "we could not tell". That is exactly the conflation
    the unresolved list exists to prevent, and a key naming the symbols would
    have printed the wrong word beside it just as confidently.
    """
    from textual.widgets import DataTable

    await _seed(marcus=[("StrictSite", QueryStatus.ILLEGAL)])
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        row = app.query_one("#sec-unresolved", DataTable).get_row_at(0)
        mark = str(row[0])
        assert mark == status_style(QueryStatus.ILLEGAL).glyph
        assert mark != status_style(QueryStatus.UNKNOWN).glyph

        key = _legend_text(app)
        assert "rejected" in key
        assert "inconclusive" not in key


async def test_deleting_a_username_removes_both_tables(db: SherlockDB):
    """Leaving the `usernames` row would keep the name, its first-seen date and
    its stored profile on disk. For a tool whose subject is people, "most of
    the record" is not what removing a record means."""
    await db.save_result(
        username="wrongperson",
        site_name="GitHub",
        site_url="https://github.com/wrongperson",
        status=str(QueryStatus.CLAIMED),
        status_code=200,
        query_time_ms=1.0,
        error_context=None,
        response_text=None,
    )
    await db.update_username_profile_summary(
        username="wrongperson",
        profile_summary='{"username": "wrongperson"}',
        input_hash="h",
    )

    assert await db.delete_username("wrongperson") == 1

    assert await db.list_usernames() == []
    assert await db.get_username_overview("wrongperson") is None
    assert await db.get_saved_results("wrongperson") == {}
    # The username row itself is gone, not just emptied.
    assert await db.get_profile_summary_cache("wrongperson") is None
    # Deleting something absent is not an error.
    assert await db.delete_username("wrongperson") == 0


async def _settle(app, pilot, predicate=None, tries: int = 30) -> None:
    """Wait for the app's workers, not just for the event loop to go quiet.

    `pilot.pause()` ends at `wait_for_idle`, which is satisfied the moment no
    callback is READY to run -- and a `@work` worker awaiting SQLite through
    aiosqlite's thread executor leaves the loop exactly that idle while its I/O
    is outstanding. So a `for _ in range(20): await pilot.pause()` loop can spin
    through all twenty iterations in microseconds without the worker advancing a
    step. On a fast disk the I/O lands between iterations and the loop looks
    like it works; on a slower runner it does not, which is how macOS CI failed
    `test_confirming_delete_erases_and_refreshes_the_list` with `assert 2 == 1`
    despite the test already pausing twenty times.

    `workers.wait_for_complete()` is the actual barrier. It stays inside a
    bounded loop because these flows CHAIN workers -- the delete finishes, and
    only then does the reload it triggers start -- so one barrier does not
    always cover the whole sequence.
    """
    for _ in range(tries):
        await app.workers.wait_for_complete()
        await pilot.pause()
        if predicate is None or predicate():
            return


async def test_delete_asks_first_and_cancelling_keeps_everything():
    """The only irreversible action in the app, and the only one that asks.

    Cancel must be the default: a dialog whose default is the destructive
    option gets dismissed into data loss by muscle memory.
    """
    from textual.widgets import Button

    from sherlock_project.database import SherlockDB, default_database_path
    from sherlock_project.tui.confirm_screen import ConfirmScreen

    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(10):
            await pilot.pause()

        await pilot.press("delete")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen)

        # Cancel holds focus, not the destructive control.
        assert app.focused is screen.query_one("#confirm-no", Button)
        # And the question names what will go, with figures.
        detail = screen.query_one("#confirm-detail").render().plain
        assert "marcus" in detail
        assert "cannot be undone" in detail

        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert [item.username for item in await db.list_usernames()] == ["marcus"]
    finally:
        await db.close()


async def test_confirming_delete_erases_and_refreshes_the_list():
    from textual.widgets import DataTable

    from sherlock_project.database import SherlockDB, default_database_path
    from sherlock_project.tui.confirm_screen import ConfirmScreen

    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
    )

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        await _settle(app, pilot)

        table = app.query_one("#username-list", DataTable)
        await _settle(app, pilot, lambda: table.row_count == 2)
        assert table.row_count == 2
        selected = app.query_one(ResultsPane)._selected

        await pilot.press("delete")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await _settle(app, pilot, lambda: table.row_count == 1)

        assert table.row_count == 1

    db = await SherlockDB.create(str(default_database_path()))
    try:
        remaining = [item.username for item in await db.list_usernames()]
    finally:
        await db.close()
    assert selected not in remaining
    assert len(remaining) == 1


def _delete_marks(table) -> dict[str, str]:
    """Which rows are currently drawing a delete control, by username."""
    from sherlock_project.tui.results_pane import UsernameList

    return {
        str(key.value): table.get_cell(key, UsernameList.DELETE_COLUMN).plain.strip()
        for key in table.rows
    }


def _delete_cell(table, username: str) -> tuple[int, int]:
    """Where `username`'s delete control is, in the list's own coordinates.

    Measured off the table rather than written down: the column widths are
    tuned to fit a fixed pane and a test that hard-codes an offset starts
    clicking the wrong cell the first time one of them moves, which would look
    like the control had stopped working.
    """
    x = sum(column.get_render_width(table) for column in table.ordered_columns[:-1])
    return x + 1, table.header_height + table.get_row_index(username)


@asynccontextmanager
async def _list_with(**rows):
    """The results tab, open, with its list loaded.

    A context manager rather than the one-shot async generator the filter tests
    use: these tests wait on the list with `_settle`, and a predicate that
    closes over a variable bound by `async for` is a closure over a loop
    variable -- which is a lint error, and a real trap the moment a helper
    yields twice.
    """
    from sherlock_project.tui.results_pane import UsernameList

    await _seed(**rows)
    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.press("alt+2")
        table = app.query_one("#username-list", UsernameList)
        await _settle(app, pilot, lambda: table.row_count == len(rows))
        assert table.row_count == len(rows)
        yield app, table, pilot


async def test_a_username_offers_a_delete_control_while_it_is_pointed_at():
    """Erasing a username was a key with nothing on screen to say it existed.

    `delete` worked on the selected row and the footer named the key, but the
    list drew no control at all -- so the one action in the app that destroys a
    dossier was the one you had to already know about.

    Drawn on the pointed-at row ONLY. A ✕ on every row reads as a list of names
    queued for deletion, and it parks an irreversible control one misclick from
    each of them.
    """
    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
    ) as (_app, table, pilot):
        assert _delete_marks(table) == {"marcus": "", "keeper": ""}

        await pilot.hover("#username-list", offset=_delete_cell(table, "keeper"))
        await pilot.pause()
        assert _delete_marks(table) == {"marcus": "", "keeper": "✕"}

        # One row at a time: the control follows the pointer rather than
        # accumulating behind it.
        await pilot.hover("#username-list", offset=_delete_cell(table, "marcus"))
        await pilot.pause()
        assert _delete_marks(table) == {"marcus": "✕", "keeper": ""}

        # Below the last row there is no row, whatever the row cursor does with
        # that space -- so there is nothing to offer either.
        x, _ = _delete_cell(table, "marcus")
        await pilot.hover("#username-list", offset=(x, table.header_height + 12))
        await pilot.pause()
        assert _delete_marks(table) == {"marcus": "", "keeper": ""}

        # And a control still drawn after the pointer has gone belongs to no
        # row at all.
        await pilot.hover("#username-list", offset=_delete_cell(table, "keeper"))
        await pilot.pause()
        await pilot.hover("#detail-tabs")
        await pilot.pause()
        assert _delete_marks(table) == {"marcus": "", "keeper": ""}


async def test_the_delete_control_asks_about_its_own_row_and_selects_nothing():
    """The ✕ acts on the row under the pointer, not on the open one.

    That is the whole point of having it: removing a username you can see
    should not mean opening it first, which is what the key makes you do.

    And pressing it must not move the selection. Cancelling has to leave the
    pane exactly as it was -- a dialog that swapped the detail on the right for
    a username you then decided not to delete has already done something you
    did not ask for.
    """
    from sherlock_project.database import SherlockDB, default_database_path
    from sherlock_project.tui.confirm_screen import ConfirmScreen

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Reddit", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
    ) as (app, table, pilot):
        pane = app.query_one(ResultsPane)
        opened = pane._selected
        other = next(
            str(key.value) for key in table.rows if str(key.value) != opened
        )
        cursor = table.cursor_row

        await pilot.click("#username-list", offset=_delete_cell(table, other))
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ConfirmScreen)
        detail = screen.query_one("#confirm-detail").render().plain
        # The row that was pressed, counted from its own listing.
        assert other in detail
        assert opened not in detail
        assert "cannot be undone" in detail

        assert pane._selected == opened, "pressing ✕ moved the selection"
        assert table.cursor_row == cursor

        await pilot.press("escape")
        await _settle(app, pilot)

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert len(await db.list_usernames()) == 2, "cancelling deleted something"
    finally:
        await db.close()


async def test_deleting_a_pointed_at_row_leaves_the_reader_where_they_were():
    """Erasing the row the pointer was on must not move the reader off the row
    they were reading.

    The list reopens the FIRST username after a reload unless it is told
    otherwise, so deleting a third party would otherwise swap the detail pane
    for someone else's record as a side effect.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
        third=[("Forum", QueryStatus.CLAIMED)],
    ) as (app, table, pilot):
        pane = app.query_one(ResultsPane)
        opened = pane._selected
        doomed = next(
            str(key.value) for key in table.rows if str(key.value) != opened
        )

        await pilot.click("#username-list", offset=_delete_cell(table, doomed))
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await _settle(app, pilot, lambda: table.row_count == 2)

        assert table.row_count == 2
        assert doomed not in _delete_marks(table)
        assert pane._selected == opened

    db = await SherlockDB.create(str(default_database_path()))
    try:
        remaining = [item.username for item in await db.list_usernames()]
    finally:
        await db.close()
    assert doomed not in remaining
    assert opened in remaining


async def test_the_progress_strip_sits_with_the_findings():
    """At the bottom of the pane it read as chrome next to the keybindings and
    was missed entirely by the person watching the scan it reported on."""
    from textual.widgets import DataTable, Static

    app = SherlockUI()
    async with app.run_test(size=(110, 34)):
        pane = app.query_one(ScanPane)
        strip = app.query_one("#progress-strip", Static)

        # Nothing to report yet, so no row of dashes labelled "idle".
        assert strip.display is False

        reporter = TuiReporter()
        reporter.start("someone", total=3)
        reporter.update(_result("GitHub", QueryStatus.CLAIMED))
        pane._reporter = reporter
        pane._flush()

        assert strip.display is True
        # Between the heading and the table, inside the findings column.
        column = list(app.query_one("#feed-col").children)
        assert [w.id for w in column][:3] == ["feed-title", "progress-strip", "feed"]
        assert isinstance(app.query_one("#feed", DataTable), DataTable)


async def _results_with(username: str, *, extractions: int = 0):
    """Seed a username, optionally with stored Pass 1 evidence."""
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for index in range(max(1, extractions)):
            site = f"Site{index:02d}"
            await db.save_result(
                username=username,
                site_name=site,
                site_url=f"https://{site.lower()}.com/{username}",
                status=str(QueryStatus.CLAIMED),
                status_code=200,
                query_time_ms=1.0,
                error_context=None,
                # Page text is what extraction runs on -- a row without it is
                # never eligible, which is why seeding it matters here.
                response_text="<html>Avery Stone</html>" if extractions else None,
                transport="browser",
            )
        if extractions:
            pending = await db.get_pending_ai_extraction_ids(
                username, contract_hash="hash"
            )
            for site_id in pending[:extractions]:
                await db.update_result_ai_extraction(
                    site_id,
                    '{"full_name": ["Avery Stone"]}',
                    contract_hash="hash",
                    model_key="vendor/m",
                )
    finally:
        await db.close()


async def test_no_evidence_offers_a_scan_rather_than_an_empty_build():
    """Pass 2 merges stored extractions; it does not read pages.

    A username scanned without analysis has nothing to synthesise, so a Build
    button there would produce an empty profile and look broken rather than say
    why.
    """
    from textual.widgets import Button, Static

    await _results_with("noevidence")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()
        await pilot.press("alt+right")
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "scanned without analysis" in hint
        assert "Scan with analysis" in str(
            app.query_one("#profile-build", Button).label
        )
        # No point offering anchors for a build that cannot happen.
        assert app.query_one("#profile-anchors", Button).display is False


async def test_stored_evidence_offers_a_build_and_says_it_is_instant():
    """The unanchored path calls no model at all -- `run_synthesis_only` loads
    one only when anchors make it necessary. Worth saying, because "build a
    profile" otherwise reads as a slow operation."""
    from textual.widgets import Button, Static

    await _results_with("hasevidence", extractions=3)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()
        await pilot.press("alt+right")
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "Evidence from 3 sites is ready" in hint
        assert "builds instantly" in hint
        assert "Build profile" in str(
            app.query_one("#profile-build", Button).label
        )


async def test_an_existing_profile_offers_a_rebuild():
    """Changing the anchors and re-running pass 2 is the other half of this.

    It belongs here, beside the profile it replaces -- not on the scan tab,
    which scans nothing to do it. The moment you want different anchors is the
    moment you are looking at a profile that mixed two people together.
    """
    from textual.widgets import Button

    from sherlock_project.database import SherlockDB, default_database_path

    await _results_with("built", extractions=1)
    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.update_username_profile_summary(
            username="built",
            profile_summary=(
                '{"username": "built", "input_hash": "h", "mode": "aggregate",'
                ' "resolution_status": "aggregated", "completeness": "partial",'
                ' "strong_profile": {"full_name": ["Avery"]}}'
            ),
            input_hash="h",
        )
    finally:
        await db.close()

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()
        assert "Rebuild" in str(app.query_one("#profile-build", Button).label)


async def test_a_rebuild_keeps_the_profiles_own_anchors():
    """The CLI defect this pane does not inherit.

    `--ai-synthesize-only` takes anchors from the command line and never from
    the profile it overwrites, so rebuilding without re-typing `--anchor`
    discards them: `has_anchors` goes false, synthesis falls through to the
    aggregate path, and the identity resolution is abandoned rather than
    recomputed. Measured on real data as 4 fields / 7 values / 2 anchors
    becoming 11 fields / 73 values / 0 -- bigger, and worse.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    await _results_with("anchored", extractions=2)
    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.update_username_profile_summary(
            username="anchored",
            profile_summary=(
                '{"username": "anchored", "input_hash": "h", "mode": "anchored",'
                ' "resolution_status": "resolved", "completeness": "partial",'
                ' "strong_profile": {"full_name": ["Avery"]},'
                ' "anchors": [{"field": "name", "value": "Avery Stone"}]}'
            ),
            input_hash="h",
        )
    finally:
        await db.close()

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()

        pane = app.query_one(ResultsPane)
        assert [a.field for a in pane._build_anchors] == ["name"], (
            "the stored anchors were not carried into the rebuild"
        )


async def test_rebuilding_an_anchored_profile_with_no_anchors_is_warned_about():
    """Not a recompute -- an abandonment. The result looks bigger while being
    worse, so the warning has to arrive before the button, not after."""
    from textual.widgets import Static

    from sherlock_project.database import SherlockDB, default_database_path

    await _results_with("anchored2", extractions=2)
    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.update_username_profile_summary(
            username="anchored2",
            profile_summary=(
                '{"username": "anchored2", "input_hash": "h", "mode": "anchored",'
                ' "resolution_status": "resolved", "completeness": "partial",'
                ' "strong_profile": {"full_name": ["Avery"]},'
                ' "anchors": [{"field": "name", "value": "Avery Stone"}]}'
            ),
            input_hash="h",
        )
    finally:
        await db.close()

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()

        pane = app.query_one(ResultsPane)
        # Someone clears them in the editor.
        pane._build_anchors = []
        pane._redraw_profile_actions(pane._record)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "unresolved merge" in hint


async def test_building_reports_progress_where_you_are_standing(monkeypatch):
    """Silence was the whole complaint, and the toast had already vanished.

    Reported here rather than in the scan tab's ACTIVITY log: that is a
    different tab, and being told to go and watch somewhere else is not
    feedback. An anchored rebuild loads the model -- measured at 187s cold --
    so the one thing this must not do is nothing.
    """
    import asyncio

    from textual.widgets import Static

    from sherlock_project import sherlock as sherlock_module

    await _results_with("slowbuild", extractions=2)

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_synthesis(*, usernames, force, inline_anchors, reporter=None):
        started.set()
        if reporter is not None:
            reporter.ai_model_starting()
        await release.wait()

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", slow_synthesis)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()
        await pilot.press("alt+right")
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        await pilot.click("#profile-build")
        for _ in range(30):
            await pilot.pause()
            if started.is_set():
                break
        assert started.is_set()

        for _ in range(6):
            await pilot.pause()

        status = app.query_one("#profile-status", Static)
        assert status.display is True
        assert "model" in status.render().plain
        # The buttons are gone, so a second press cannot start a competing
        # synthesis over the same rows.
        assert app.query_one("#profile-buttons").display is False

        release.set()
        for _ in range(20):
            await pilot.pause()


async def test_a_failed_build_leaves_the_reason_on_screen(monkeypatch):
    """A failure that disappears after five seconds is one nobody can act on."""
    from textual.widgets import Static

    from sherlock_project import sherlock as sherlock_module

    await _results_with("badbuild", extractions=1)

    async def failing(*, usernames, force, inline_anchors, reporter=None):
        raise RuntimeError("LM Studio is not running")

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", failing)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        for _ in range(12):
            await pilot.pause()
        await pilot.press("alt+right")
        await pilot.press("alt+right")
        for _ in range(6):
            await pilot.pause()

        await pilot.click("#profile-build")
        for _ in range(25):
            await pilot.pause()

        shown = app.query_one("#profile-status", Static).render().plain
        assert "LM Studio is not running" in shown
        # And the controls come back, so it can be tried again.
        assert app.query_one("#profile-buttons").display is True


def test_force_follows_the_button_not_a_constant():
    """A first BUILD must not force -- there is nothing to replace, and a
    matching cache is correct to reuse. A REBUILD must, or unchanged anchors
    would hit the cache and the button would appear to do nothing at all."""
    import inspect

    from sherlock_project.tui.results_pane import ResultsPane

    source = inspect.getsource(ResultsPane._synthesize)
    assert "force=rebuild" in source
    # And it reuses the CLI's own synthesis-only path rather than a second one.
    assert "run_synthesis_only" in source

    pressed = inspect.getsource(ResultsPane._build_profile)
    assert 'rebuild=record.get("profile") is not None' in pressed


# -- the no-terminal guard --------------------------------------------------


async def test_the_ui_refuses_to_draw_without_a_terminal(capsys):
    """A full-screen app that takes over a CI log is worse than no app."""
    console = Console(force_terminal=False, no_color=True)
    assert await run_ui([], interactive=False, console=console) == 0
    assert "sherlock <username>" in capsys.readouterr().out


async def test_unknown_arguments_are_reported_rather_than_ignored(capsys):
    """Someone typing `sherlock ui --fresh` has an expectation about that run
    which this cannot meet, so silently dropping the flag would be worse."""
    console = Console(force_terminal=False, no_color=True)
    assert await run_ui(["--fresh"], console=console) == 2
    assert "takes no arguments" in capsys.readouterr().out
