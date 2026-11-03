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
from types import SimpleNamespace
from typing import ClassVar

import pytest
from rich.console import Console
from textual.screen import ModalScreen
from textual.worker import WorkerCancelled

from sherlock_project.database import SherlockDB
from sherlock_project.result import QueryResult, QueryStatus
from sherlock_project.tui.app import SherlockUI, run_ui
from sherlock_project.tui.reporter import (
    TuiReporter,
    describe_options,
    describe_settings,
)
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
    to recommend `sherlock-rm show --unresolved`, a command nobody in the app
    can run -- the third instance of that bug. The same trap catches any
    rename: this must name a control the app actually HAS, so pointing at the
    retired UNRESOLVED tab is as broken as pointing at the flag was.
    """
    log = " ".join(line.plain for line in _scanned_reporter().snapshot_log())

    assert "gave no answer" in log
    assert "inconclusive" in log and "blocked by bot protection" in log
    assert 'Not the same as "not found"' in log
    assert "found only" in log and "SITES" in log
    assert "--unresolved" not in log
    assert "UNRESOLVED tab" not in log


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
        assert str(app.query_one("#toggle-verbose", Button).label).startswith("☐")

        await pilot.click("#toggle-verbose")
        await pilot.pause()
        assert pane._verbose is True
        assert str(app.query_one("#toggle-verbose", Button).label).startswith("☑")


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


# -- the model block --------------------------------------------------------


def test_the_model_name_keeps_the_half_that_identifies_it():
    """Truncating a model key from the left throws away the answer.

    Keys are paths, and everything someone downloaded from one place shares the
    vendor and the repo -- so plain truncation renders three different models as
    three identical lines, while cutting the size and the quantisation that
    tell them apart.
    """
    from sherlock_project.tui.theme import model_name

    assert (
        model_name("unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf", 24)
        == "Qwen3-30B-A3B-Q4_K_M"
    )
    # Windows separators reach this from a models folder on this platform.
    assert model_name(r"C:\models\Qwen3-4B-GGUF\Qwen3-4B-Q8_0.gguf", 24) == (
        "Qwen3-4B-Q8_0"
    )
    # A plain key is left alone.
    assert model_name("qwen/qwen3-4b", 24) == "qwen3-4b"
    # And what still does not fit is elided rather than overflowing the column.
    narrow = model_name("Qwen3-30B-A3B-Q4_K_M.gguf", 10)
    assert len(narrow) == 10
    assert narrow.endswith("…")


def test_the_readout_rows_all_keep_one_right_edge():
    """Every block in the left column is read against the same ruler.

    A value column that starts wherever the label ended makes stacked blocks
    read as panels assembled by accident. `38 tok/s` is wider than the six cells
    a count needs, so `value_row` measures the value first and gives the label
    what is left -- the opposite split from `stat_row`, to the same right edge.
    """
    from sherlock_project.tui.theme import value_row

    widths = {
        value_row("speed", "38 tok/s").cell_len,
        value_row("· waiting", "").cell_len,
        stat_row("inconclusive", 135, "yellow").cell_len,
        stat_row("extracted", 149, "cyan").cell_len,
    }
    assert len(widths) == 1, f"the left column does not line up: {widths}"


def test_an_indented_sub_row_is_exactly_as_wide_as_a_top_level_one():
    """ANALYSIS indents the three outcomes that decompose `extracted`.

    The indent comes out of the LABEL column rather than being prefixed to the
    line, so the numbers go on stacking. Prefixed, every sub-row's value would
    sit two cells right of its own total, which is the ragged edge this whole
    file exists to prevent.
    """
    top = stat_row("extracted", 149, "cyan")
    sub = stat_row("with facts", 8, "green", indent=2)
    assert top.cell_len == sub.cell_len
    assert sub.plain.startswith("  ")

    # A pathological indent eats the label, never the alignment.
    assert stat_row("x", 1, "cyan", indent=99).cell_len == top.cell_len


def test_speed_reads_as_whole_tokens_per_second():
    from sherlock_project.tui.theme import throughput_label

    assert throughput_label(38.4) == "38 tok/s"
    # Absent, not "0 tok/s" -- see `TuiReporter.throughput`.
    assert throughput_label(None) == ""
    assert throughput_label(0) == ""


def _trace(
    *,
    output_tokens,
    elapsed,
    generation_seconds=None,
    phase="pass_one",
):
    from sherlock_project.ai_engine import AIRequestTrace
    from sherlock_project.ai_provider import AIGenerationStats

    return AIRequestTrace(
        phase=phase,
        username="someone",
        site_name="GitHub",
        site_id=1,
        attempt=1,
        provider="llama.cpp",
        model_key="qwen/qwen3-4b",
        temperature=0.2,
        context_length=32768,
        max_tokens=1024,
        elapsed_seconds=elapsed,
        stats=AIGenerationStats(
            output_tokens=output_tokens,
            generation_seconds=generation_seconds,
        ),
        native_reasoning="",
        final_text="{}",
        structured_reasoning="",
        validated_output={},
        validation_error=None,
    )


def test_speed_is_measured_by_the_model_not_by_the_screen():
    """Averaged over finished requests, from each request's own timings.

    A live rate is not available and is not faked: requests are sent with
    streaming off, so between dispatch and the complete reply there is nothing
    to observe. What the server reports afterwards is a real measurement, and
    the average of those is the only honest speed this screen can show.
    """
    reporter = TuiReporter()
    assert reporter.throughput is None

    reporter.ai_trace(
        _trace(output_tokens=300, elapsed=10.0, generation_seconds=5.0)
    )
    reporter.ai_trace(
        _trace(output_tokens=300, elapsed=10.0, generation_seconds=5.0)
    )
    assert reporter.throughput == pytest.approx(60.0)

    # A server that reports no token usage leaves this absent rather than zero:
    # `0 tok/s` beside a model that is visibly working reads as a fault.
    silent = TuiReporter()
    silent.ai_trace(_trace(output_tokens=None, elapsed=10.0))
    assert silent.throughput is None


def test_speed_excludes_the_time_the_model_spent_reading_the_prompt():
    """The denominator is generation time, not the round trip.

    Pass 1 sends a whole scraped page, so prompt processing is a large and
    VARIABLE share of each request -- large enough that dividing by the round
    trip reported a model generating at 60 tok/s as doing 30, and variable
    enough that the error moved with page size rather than with the model. The
    number beside `speed` has to be the one llama.cpp would print, or it cannot
    be compared against a benchmark, a driver change, or another machine.
    """
    reporter = TuiReporter()
    # 300 tokens in 5s of generation; the other 5s went on the prompt.
    reporter.ai_trace(
        _trace(output_tokens=300, elapsed=10.0, generation_seconds=5.0)
    )
    assert reporter.throughput == pytest.approx(60.0)

    # Weighted by tokens, which is what summing each side separately gives: a
    # 30-token reply does not get an equal say with a 300-token one.
    reporter.ai_trace(
        _trace(output_tokens=30, elapsed=8.0, generation_seconds=1.0)
    )
    assert reporter.throughput == pytest.approx(330 / 6.0)


def test_a_failed_request_does_not_drag_the_speed_down():
    """A request that never produced tokens took no generation time either.

    A timeout or a dropped connection still emits a trace, with its full
    round trip and empty stats. Counted, one 120s failure would halve the
    displayed speed for the rest of a scan and keep it there -- a metric that
    reports the scan's bad luck as the model's slowness.
    """
    reporter = TuiReporter()
    reporter.ai_trace(
        _trace(output_tokens=300, elapsed=10.0, generation_seconds=5.0)
    )
    reporter.ai_trace(
        _trace(output_tokens=None, elapsed=120.0, generation_seconds=None)
    )
    assert reporter.throughput == pytest.approx(60.0)


async def test_the_extraction_in_flight_is_a_stopwatch_not_a_snapshot():
    """Which site, and for how long, recomputed on every read.

    The site name is the parent's own -- the same field the CLI progress bar
    captions itself with, so the two surfaces cannot name different sites. The
    only thing added here is the start time, because a Rich task times itself
    and the parent therefore keeps none.
    """
    import asyncio

    reporter = TuiReporter()
    assert reporter.extraction is None

    reporter.ai_job_started("Reddit")
    live = reporter.extraction
    assert live is not None
    assert live.site_name == "Reddit" == reporter._ai_current_site

    await asyncio.sleep(0.05)
    assert reporter.extraction.elapsed > live.elapsed

    reporter.ai_job_finished("with_facts")
    assert reporter.extraction is None


def test_a_job_that_never_finishes_does_not_leave_a_spinner_turning():
    """A model that dies mid-request leaves a job started and never finished.

    A live indicator outliving the activity it describes is the one failure it
    must not have, so the end of the pass clears it whatever happened to the
    job -- as does an interrupted run.
    """
    ended = TuiReporter()
    ended.ai_job_started("Reddit")
    ended.ai_pass_finished()
    assert ended.extraction is None

    stopped = TuiReporter()
    stopped.ai_job_started("Reddit")
    stopped.processing_interrupted()
    assert stopped.extraction is None


def _model_settings() -> dict:
    return {
        "ai.model": "unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
        "ai.context_length": 32768,
        "ai.temperature": 0.2,
    }


def _model_block(app) -> str:
    from textual.widgets import Static

    return app.query_one("#model-lines", Static).render().plain


def _analysis_block(app) -> str:
    from textual.widgets import Static

    return app.query_one("#ai-counters", Static).render().plain


def _synthesis_line(app) -> str:
    from textual.widgets import Static

    return app.query_one("#synthesis-line", Static).render().plain


def _profile():
    """The minimum `synthesis_finished` needs: a name and a resolution status.

    Both are read on the way past -- the base reporter details its success line
    with the status, and `TuiReporter.render_profile` names the username when it
    points at the RESULTS tab.
    """
    from sherlock_project.profile_synthesis import ProfileSynthesis

    return ProfileSynthesis.model_validate(
        {
            "username": "someone",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "resolved",
            "completeness": "partial",
        }
    )


def _rows(block: str) -> dict[str, str]:
    """A readout block as {label: value}, so tests assert facts not padding.

    The column widths are deliberate and are covered by their own tests; a
    behaviour test that hard-codes the spacing fails whenever the layout is
    tuned, which teaches everyone to update assertions without reading them.
    """
    rows: dict[str, str] = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows[" ".join(parts[:-1])] = parts[-1]
    return rows


async def test_the_model_is_named_before_a_scan_rather_than_after():
    """The point of stating the model is to be read BEFORE committing to a run.

    Drawn from stored settings, so it answers the moment analysis is turned on
    -- a block that only filled in once a scan started would arrive after the
    decision it informs.
    """
    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())

        # Hidden with analysis off, like the anchors block: a model this run
        # will not load is not state worth a panel.
        assert app.query_one("#model-block").display is False

        await pilot.click("#toggle-ai")
        await pilot.pause()
        assert app.query_one("#model-block").display is True

        rendered = _model_block(app)
        assert "Qwen3-30B-A3B-Q4_K_M" in rendered
        assert "32768" in rendered
        assert "0.2" in rendered
        # The vendor and the repo are the half that was dropped.
        assert "unsloth" not in rendered


async def test_analysis_without_a_model_says_where_to_set_one():
    """The empty state teaches, exactly like the empty anchors list. This is
    precisely the moment someone needs telling: analysis is on and cannot run."""
    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values["ai.model"] = None

        await pilot.click("#toggle-ai")
        await pilot.pause()
        assert "SETTINGS" in _model_block(app)


async def test_the_live_rows_sit_with_the_analysis_they_describe():
    """Which site, for how long -- the finest grain there is with streaming off.

    IN THE ANALYSIS BLOCK, not the MODEL block. Speed and the site in flight are
    progress of the extraction, not properties of the model: they exist only
    while a run is going, and under MODEL they sat beside a name, a window and a
    temperature that cannot change once it starts. MODEL is a nameplate now and
    everything that moves is counted in one place.

    And it stops when the scan does: `_flush` returns early once a finished scan
    has nothing left to drain, which is exactly the tick that would otherwise
    leave the last spinner frame on screen forever.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 40)) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())
        pane._use_ai = True

        reporter = TuiReporter()
        reporter.start("someone", total=1)
        reporter.ai_pass_started()
        reporter.ai_scheduled()
        reporter.ai_job_started("Reddit")
        reporter.ai_trace(
            _trace(output_tokens=300, elapsed=10.0, generation_seconds=10.0)
        )
        pane._reporter = reporter
        pane._scan_running = True

        pane._flush()
        await pilot.pause()
        rendered = _analysis_block(app)
        assert "Reddit" in rendered
        assert "30 tok/s" in rendered
        # The nameplate carries none of it.
        assert "Reddit" not in _model_block(app)

        reporter.ai_job_finished("with_facts")
        reporter.ai_pass_finished()
        pane._scan_running = False
        pane._redraw_ai(reporter)
        await pilot.pause()
        finished = _analysis_block(app)
        assert "Reddit" not in finished
        # The measurement of the run survives; only the live part goes.
        assert "30 tok/s" in finished


async def test_analysis_says_what_became_of_every_extraction():
    """`extracted` decomposes into outcomes, and the outcomes are drawn.

    The scan computes six figures and the panel used to draw three, so
    `extracted 149 / with facts 8` left 141 sites unaccounted for -- covering
    three unrelated situations, one of which is OUTRIGHT FAILURE. A model
    quietly failing forty extractions looked exactly like a model succeeding on
    forty empty pages.

    That is the error the SITES block one column up refuses to make, and it is
    the distinction the whole tool is built on.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 40)) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())
        pane._use_ai = True

        reporter = TuiReporter()
        reporter.start("someone", total=3)
        reporter.ai_pass_started()
        for _ in range(3):
            reporter.ai_scheduled()
        reporter.ai_job_finished("with_facts")
        reporter.ai_job_finished("no_facts")
        # How the pipeline records a failure: the log line, then the outcome.
        reporter.ai_failed("Bandcamp", RuntimeError("out of tokens"))
        reporter.ai_job_finished("pending")

        pane._reporter = reporter
        pane._scan_running = True
        pane._redraw_ai(reporter)
        await pilot.pause()

        rows = _rows(_analysis_block(app))
        assert rows["extracted"] == "3"
        assert rows["with facts"] == "1"
        assert rows["no facts"] == "1"
        # `pending` is the failure bucket. The panel calls it what it is.
        assert rows["failed"] == "1"
        assert rows["queued"] == "0"

        # The breakdown is a real decomposition, not an assortment: every
        # finished job lands in exactly one bucket, so they sum to the total.
        assert (
            int(rows["with facts"]) + int(rows["no facts"]) + int(rows["failed"])
            == int(rows["extracted"])
        )


async def test_pass_two_gets_its_own_line_and_stops_the_waiting_row_lying():
    """The bug this restructure exists for.

    Synthesis is awaited inside the scan session, so while it runs the scan is
    still going, no extraction is in flight, and the held row concluded the only
    thing it could: `· waiting`. The panel reported idle through the single
    heaviest model call of the run -- every stored extraction merged in one
    request, and a model loaded cold for it on an anchored rebuild.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 40)) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())
        pane._use_ai = True

        reporter = TuiReporter()
        reporter.start("someone", total=1)
        reporter.ai_pass_started()
        reporter.ai_scheduled()
        reporter.ai_job_finished("with_facts")
        reporter.ai_pass_finished()
        pane._reporter = reporter
        pane._scan_running = True

        # Between the passes the held row is still correct, and still drawn --
        # it is only wrong once Pass 2 owns the screen.
        pane._redraw_ai(reporter)
        await pilot.pause()
        assert app.query_one("#synthesis-line").display is False
        assert "waiting" in _analysis_block(app)

        reporter.synthesis_started("someone")
        pane._redraw_ai(reporter)
        await pilot.pause()
        assert app.query_one("#synthesis-line").display is True
        assert "building" in _synthesis_line(app)
        assert "waiting" not in _analysis_block(app)

        reporter.synthesis_finished("someone", _profile(), cache_hit=False)
        pane._scan_running = False
        pane._redraw_ai(reporter)
        await pilot.pause()
        assert "ready" in _synthesis_line(app)


async def test_pass_two_is_reported_even_when_this_run_extracted_nothing():
    """A resumed username whose hits all have stored extractions.

    Nothing is scheduled for Pass 1, so the tallies say "not running" -- and
    Pass 2 still runs, over the evidence already on disk. Gated on `scheduled`
    the synthesis row would have been invisible on exactly the runs where it is
    the only thing happening.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 40)) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())
        pane._use_ai = True

        reporter = TuiReporter()
        reporter.start("someone", total=0)
        reporter.synthesis_started("someone")
        pane._reporter = reporter
        pane._scan_running = True
        pane._redraw_ai(reporter)
        await pilot.pause()

        assert "not running" in _analysis_block(app)
        assert app.query_one("#synthesis-line").display is True
        assert "building" in _synthesis_line(app)


async def test_a_stopped_scan_leaves_no_synthesis_spinner_turning():
    """A cancelled worker never reports an outcome for Pass 2.

    It did not fail and it did not land, so the line claims neither -- but a
    spinner still turning beside a scan that has stopped is the one thing a live
    indicator must never do.
    """
    app = SherlockUI()
    async with app.run_test(size=(110, 40)) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.update(_model_settings())
        pane._use_ai = True

        reporter = TuiReporter()
        reporter.start("someone", total=1)
        reporter.synthesis_started("someone")
        pane._reporter = reporter

        pane._scan_running = False
        pane._redraw_ai(reporter)
        await pilot.pause()
        assert "building" not in _synthesis_line(app)
        assert "stopped" in _synthesis_line(app)


def test_an_interrupted_synthesis_reports_no_outcome_at_all():
    """Abandoned is neither `ready` nor `failed`, so the row goes entirely."""
    reporter = TuiReporter()
    reporter.synthesis_started("someone")
    assert reporter.synthesis is not None

    reporter.processing_interrupted()
    assert reporter.synthesis is None


def test_a_cached_profile_is_distinguished_from_a_rebuilt_one():
    """The answer to "I changed the model and nothing happened".

    A cache hit is a landed profile the model was never asked to build, which is
    a different fact from one it just produced -- and it is the fact behind the
    commonest confusion about this tool.
    """
    rebuilt = TuiReporter()
    rebuilt.synthesis_started("someone")
    rebuilt.synthesis_finished("someone", _profile(), cache_hit=False)
    assert rebuilt.synthesis.state == "ready"

    cached = TuiReporter()
    cached.synthesis_started("someone")
    cached.synthesis_finished("someone", _profile(), cache_hit=True)
    assert cached.synthesis.state == "cached"


def test_a_landed_synthesis_stops_counting():
    """The bug the startup phases already paid for, not repeated here.

    `TerminalReporter` keeps only a start time, so a duration recomputed on each
    repaint makes a finished step climb forever -- `ready 1s`, `ready 2s`, and
    eventually minutes for something that took under a second.
    """
    reporter = TuiReporter()
    reporter.synthesis_started("someone")
    reporter.synthesis_finished("someone", _profile(), cache_hit=False)

    settled = reporter.synthesis.elapsed
    assert reporter.synthesis.elapsed == settled


# -- no command-line advice inside the app ----------------------------------


def _profile_with_warnings():
    from sherlock_project.profile_synthesis import ProfileSynthesis

    return ProfileSynthesis.model_validate(
        {
            "username": "7ghost",
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
            "username": "7ghost",
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
    # The other note is diagnostics and still shows -- as a count of pages,
    # not the site id the stored legacy wording listed.
    assert "1 stored page has not been analysed" in expanded
    assert "1726" not in expanded

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
            "username": "7ghost",
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
        ) else rendered.index("not been analysed")

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
        username="7ghost",
        configured_model="qwen/qwen3-4b",
        counts={"qwen/qwen3-8b": 98},
    )
    log = " ".join(line.plain for line in reporter.snapshot_log())

    assert "did not come from qwen/qwen3-4b" in log
    assert "Re-scan all" in log
    assert "--fresh" not in log
    assert "sherlock 7ghost" not in log


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
        username="7ghost",
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


def test_each_toggle_tooltip_names_the_toggle_it_belongs_to():
    """A floating box beside the pointer carries no other clue about which of
    two adjacent controls it is describing."""
    analysis, verbose = describe_options(
        {"ai.model": "vendor/m"}, use_ai=False, verbose=False
    )
    assert analysis.startswith("AI analysis — OFF")
    assert verbose.startswith("Verbose — OFF")


def test_the_toggle_tooltips_give_the_consequence_not_just_the_state():
    """The toggle already shows `off`. What it cannot show is what turning it
    off costs -- the results tab builds its profile from what analysis stores,
    so a scan run without it can never produce one."""
    off, _ = describe_options({"ai.model": "vendor/m"}, use_ai=False, verbose=False)
    assert "cannot build a profile" in off

    on, _ = describe_options({"ai.model": "vendor/m"}, use_ai=True, verbose=False)
    assert "reads every hit" in on


def test_the_analysis_tooltip_says_so_when_no_model_is_configured():
    """The one state the toggle cannot honour on its own -- and the reason is
    two tabs away, so the text has to name where to go."""
    warned, _ = describe_options({}, use_ai=True, verbose=False)
    assert "no model is configured" in warned
    assert "SETTINGS" in warned

    # Nothing is asking for a model, so its absence is not worth raising here.
    # The stored-settings line beside the toggles still reports it.
    quiet, _ = describe_options({}, use_ai=False, verbose=False)
    assert "no model" not in quiet
    assert "no model configured" in describe_settings({})


def test_a_toggle_tooltip_defers_a_toggle_flipped_during_a_scan():
    """Both toggles are read once, when the scan starts, so flipping either
    mid-run changes the NEXT scan and nothing about this one.

    Left unsaid the toggle reads as a live control: analysis switched on at site
    40 of 680 extracts nothing for the remaining 640, and the screen offers no
    reason why.
    """
    analysis, verbose = describe_options(
        {"ai.model": "vendor/m"},
        use_ai=True,
        verbose=False,
        running=(False, False),
    )

    assert "from the NEXT scan" in analysis
    assert "started without it" in analysis
    # Verbose is what the run is actually using, so it is described rather than
    # deferred. Deferring both would say the wrong thing about one of them.
    assert "NEXT scan" not in verbose


def test_the_toggle_tooltips_defer_nothing_when_they_match_the_run():
    """A scan started WITH analysis is already applying it -- telling that
    operator it takes effect next time is simply false."""
    analysis, verbose = describe_options(
        {"ai.model": "vendor/m"}, use_ai=True, verbose=True, running=(True, True)
    )
    assert "NEXT scan" not in analysis
    assert "NEXT scan" not in verbose
    assert "reads every hit" in analysis


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


async def test_a_fully_stored_username_analyses_without_fetching_anything(
    monkeypatch,
):
    """The pass the resume dialog now offers, asserted where it happens.

    Two halves have to hold at once for "analyse without scanning" to be real:
    nothing may be fetched, and the model must still be given every stored page
    it has not read. The engine is skipped because the plan has nothing to
    fetch, and the backfill after it enqueues from the database -- so this is
    the guarantee the dialog's wording depends on, and the one that would break
    silently if the backfill ever moved inside the `if site_data:` branch.
    """
    from sherlock_project.database import SherlockDB, default_database_path
    from sherlock_project.tui import runner as runner_module

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for site in ("GitHub", "Reddit"):
            await db.save_result(
                username="someone",
                site_name=site,
                site_url=f"https://{site.lower()}.com/someone",
                status=str(QueryStatus.CLAIMED),
                status_code=200,
                query_time_ms=1.0,
                error_context=None,
                response_text="<html>Avery Stone</html>",
                transport="browser",
            )
    finally:
        await db.close()

    enqueued: list[int] = []

    async def fake_pipeline(*, ai_queue, sherlock_db, reporter, ai_settings):
        while True:
            try:
                site_id = await ai_queue.get()
            except Exception:
                return
            enqueued.append(site_id)
            ai_queue.task_done()

    async def fake_synthesize(**kwargs):
        return None

    import sherlock_project.sherlock as sherlock_module

    monkeypatch.setattr(sherlock_module, "run_ai_pipeline", fake_pipeline)
    monkeypatch.setattr(sherlock_module, "synthesize_profiles", fake_synthesize)

    class _Stored:
        ai = SimpleNamespace(model="vendor/m")

    monkeypatch.setattr(runner_module, "try_load_settings", lambda: _Stored())

    scanned: list[dict] = []

    async def scan_spy(**kwargs):
        scanned.append(kwargs)
        return {}

    await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": True, "ai.model": "vendor/m"},
        scan_spy=scan_spy,
        sites=("GitHub", "Reddit"),
        use_ai=True,
    )

    assert _FakeEngine.instances == [], "an engine was built with nothing to fetch"
    assert scanned == [], "a fetch happened for pages already stored"
    # And the model was still given both stored pages.
    expected = await _pending_ids("someone")
    assert enqueued == expected != []


async def _pending_ids(username: str) -> list[int]:
    from sherlock_project.ai_engine import pass_one_contract_hash
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        return await db.get_pending_ai_extraction_ids(
            username, contract_hash=pass_one_contract_hash()
        )
    finally:
        await db.close()


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

    `sherlock-rm settings` keeps ctrl+s to save because it shipped that way and
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
        await _settle(app, pilot, lambda: bool(launched))
    return launched


async def test_pressing_enter_starts_a_scan(monkeypatch):
    async def press(pilot, app):
        await pilot.press("enter")

    assert await _start_scan_and_capture(monkeypatch, press) == ["alice"]


async def test_clicking_the_scan_button_starts_a_scan(monkeypatch):
    async def press(pilot, app):
        await pilot.click("#scan-button")

    assert await _start_scan_and_capture(monkeypatch, press) == ["alice"]


async def test_ctrl_r_does_not_start_a_scan(monkeypatch):
    """ctrl+r refreshes RESULTS. Here it used to start a 680-site run.

    One chord with two meanings, one of them a network operation, is a scan
    started by a hand still in the habit of the other tab. Enter and the SCAN
    button are the ways to start one, and both are tested above.
    """
    async def press(pilot, app):
        await pilot.press("ctrl+r")

    assert await _start_scan_and_capture(monkeypatch, press) == []


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
        await _settle(app, pilot, lambda: "use_ai" in captured)
        # Nothing stored for this username, so no prompt and no re-scan.
        assert captured["use_ai"] is False
        assert captured["fresh"] is False

        # The analysis toggle reaches the scan.
        await pilot.click("#toggle-ai")
        await pilot.click("#scan-button")
        # `captured` carries the previous run's value, so wait for the CHANGE.
        await _settle(app, pilot, lambda: captured.get("use_ai") is True)
        assert captured["use_ai"] is True


async def test_the_toggles_show_their_own_state():
    """They report as much as they invite a press -- a control whose state is
    invisible is one you have to press to find out about."""
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        ai = app.query_one("#toggle-ai", Button)
        assert str(ai.label) == "☐ analysis"
        await pilot.click("#toggle-ai")
        assert str(app.query_one("#toggle-ai", Button).label) == "☑ analysis"


async def test_the_toggles_carry_hover_text_that_follows_their_state():
    """Hung on every redraw, not once at mount.

    The text is a function of state, and a tooltip describing the state the app
    booted in is worse than none -- nothing about a stale one looks stale.
    """
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        ai = app.query_one("#toggle-ai", Button)
        verbose = app.query_one("#toggle-verbose", Button)
        assert "AI analysis — OFF" in ai.tooltip
        assert "Verbose — OFF" in verbose.tooltip

        await pilot.click("#toggle-ai")
        await pilot.pause()
        assert "AI analysis — ON" in app.query_one("#toggle-ai", Button).tooltip

        await pilot.click("#toggle-verbose")
        await pilot.pause()
        assert "Verbose — ON" in app.query_one("#toggle-verbose", Button).tooltip


async def test_nothing_is_drawn_under_the_options_row():
    """The reason this is a tooltip and not a pair of lines.

    Two permanent sentences explaining controls that do not change during a scan
    cost three rows the counters and the feed want back, every run, forever
    after the one time they are read.
    """
    from textual.widgets import Static

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app.query("#options-help")
        # The stored-settings summary still rides in the row itself.
        assert "browser" in app.query_one("#scan-config", Static).render().plain


async def test_hovering_a_toggle_actually_shows_the_tooltip():
    """The attribute is not the feature -- the popup is.

    `run_test` disables tooltips unless asked, so this is the one test that
    proves the wiring rather than the text: hover the control, let
    TOOLTIP_DELAY elapse, and find the box on screen carrying its words.
    """
    app = SherlockUI()
    async with app.run_test(tooltips=True) as pilot:
        from textual.widgets import Tooltip

        tooltip = app.screen.get_child_by_type(Tooltip)
        assert tooltip.display is False

        await pilot.hover("#toggle-ai")
        await pilot.pause(app.TOOLTIP_DELAY + 0.1)
        await pilot.pause()

        assert tooltip.display is True
        assert "AI analysis" in tooltip.render().plain


async def test_the_profile_button_carries_the_hover_text_too():
    """What the button IS, beside a line saying what pressing it does NOW.

    The split is what keeps the two from being one answer written twice: the
    mechanism never changes, the state changes on every redraw.
    """
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_results(app, pilot)
        anchors = app.query_one("#profile-anchors", Button)
        assert "Anchors" in anchors.tooltip
        assert "merging every name every site showed" in anchors.tooltip


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
        await _settle(app, pilot)
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
        pane = app.query_one(ScanPane)
        # NOT `_settle` here: its barrier is `workers.wait_for_complete()`, and
        # this scan sleeps for a minute on purpose -- waiting for it to finish
        # is waiting for the thing the test is about to cancel. What is needed
        # is the opposite: wait for the chain to have STARTED it. The sleep is
        # load-bearing rather than padding, because `peek_stored` runs on
        # aiosqlite's thread executor and only a real timer yields to it --
        # `pause()` alone returns while that read is still outstanding.
        for _ in range(50):
            if pane._scan_running:
                break
            await pilot.pause()
            await asyncio.sleep(0.02)
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
        assert blocked.render().plain.startswith("☐")

        await pilot.click("#count-waf")
        await pilot.pause()

        assert QueryStatus.WAF in pane._visible
        after = app.query_one("#count-waf", CounterRow).render()
        assert after.plain.startswith("☑"), (
            "the row is in the filter but still drawn as excluded"
        )

        # And back off again, still with no reporter.
        await pilot.click("#count-waf")
        await pilot.pause()
        again = app.query_one("#count-waf", CounterRow).render()
        assert again.plain.startswith("☐")


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


async def test_a_hidden_kind_is_unticked_not_removed():
    """The count has to stay exact and readable while its rows are hidden.

    What the panel counts and what the feed displays are two different facts,
    and only the second one is being toggled -- so the number never changes,
    only the box beside it. It was strikethrough, which read as "crossed off":
    on first launch four of five counters looked void before anything had run.
    """
    async for app, pane, pilot in _pane_with_results():
        absent = app.query_one("#count-available", CounterRow)
        hit = app.query_one("#count-claimed", CounterRow)

        hidden = absent.render()
        shown = hit.render()
        assert hidden.plain.startswith("☐")
        assert shown.plain.startswith("☑")
        # Nothing struck through: the number reads the same either way.
        assert not any("strike" in str(span.style) for span in hidden.spans)
        # The number is still there and still right.
        assert "1" in hidden.plain

        await pilot.click("#count-available")
        await pilot.pause()
        after = app.query_one("#count-available", CounterRow).render()
        assert after.plain.startswith("☑")


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
        await _settle(app, pilot, lambda: captured.get("username") == "nobody")

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
        await _settle(app, pilot, lambda: isinstance(app.screen, ResumeScreen))

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
        await _settle(app, pilot, lambda: app.screen.query("#resume-fresh"))

        await _click_when_drawn(app, pilot, "#resume-fresh")
        await _settle(app, pilot, lambda: captured.get("fresh") is True)

    assert captured.get("fresh") is True


async def _fully_stored_with_unread_pages(username: str, *, pages: int) -> None:
    """Every site answered, `pages` of them confirmed hits never analysed.

    The exact state the resume dialog used to have no honest answer for: no
    site left to fetch, and a whole AI pass still waiting on evidence already
    on disk.
    """
    from sherlock_project.database import SherlockDB, default_database_path
    from sherlock_project.sites import SitesInformation

    sites = SitesInformation(honor_exclusions=False)
    sites.remove_nsfw_sites(do_not_remove=[])
    names = [site.name for site in sites]

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for index, name in enumerate(names):
            claimed = index < pages
            await db.save_result(
                username=username,
                site_name=name,
                site_url=f"https://example.com/{username}",
                status=str(
                    QueryStatus.CLAIMED if claimed else QueryStatus.AVAILABLE
                ),
                status_code=200,
                query_time_ms=1.0,
                error_context=None,
                response_text="<html>Avery Stone</html>" if claimed else None,
                transport="browser",
            )
    finally:
        await db.close()


async def test_stored_pages_offer_an_analysis_pass_that_fetches_nothing(
    monkeypatch,
):
    """The dead end this change exists to remove.

    With every site answered, the dialog offered `View results` (which returns
    to the pane that sent you) and `Re-scan all` (which re-fetches 680 pages to
    reach work that needs none). The pass over stored pages already existed in
    the runner -- a resumed run with nothing to fetch builds no engine and
    analyses what is on disk -- and nothing on screen had ever said so.
    """
    from textual.widgets import Button

    from sherlock_project.tui.scan_pane import ScanPane

    await _fully_stored_with_unread_pages("marcus", pages=2)
    captured = await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values["ai.model"] = "vendor/m"
        pane._use_ai = True

        await pilot.press(*"marcus")
        await pilot.press("enter")
        await _settle(app, pilot, lambda: app.screen.query("#resume-detail"))

        detail = app.screen.query_one("#resume-detail").render().plain
        assert "have not been analysed" in detail
        assert "re-fetches nothing" in detail
        button = app.screen.query_one("#resume-analyse", Button)
        assert "Analyse 2" in str(button.label)
        # The cheap action holds focus, so Enter cannot start the expensive one.
        assert app.screen.focused is button

        await _click_when_drawn(app, pilot, "#resume-analyse")
        await _settle(app, pilot, lambda: captured.get("username") == "marcus")

    # `fresh=False` is the whole mechanism: the plan resumes every stored row,
    # leaves nothing to fetch, and the runner never builds an engine -- so the
    # model runs over pages already on disk.
    assert captured.get("fresh") is False
    assert captured.get("use_ai") is True


@pytest.mark.parametrize("size", [(80, 30), (110, 34), (140, 40)])
async def test_the_analysis_button_draws_its_whole_label(monkeypatch, size):
    """A control that cannot say what it does is not a control.

    The same failure this file already guards on the profile pane, and it was
    real here: the row is a grid whose first column is `1fr`, sized from what
    the fixed columns leave over. At 80 cells they left ten, so the primary
    button rendered as "Analyse" -- no count, no noun -- while `Button.label`
    read back in full. The count is the part that makes the offer legible
    against `Re-scan all 680` beside it, so losing it is losing the choice.
    """
    from textual.geometry import Region
    from textual.widgets import Button

    from sherlock_project.tui.scan_pane import ScanPane

    await _fully_stored_with_unread_pages("marcus", pages=3)
    await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test(size=size) as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values["ai.model"] = "vendor/m"
        pane._use_ai = True

        await pilot.press(*"marcus")
        await pilot.press("enter")
        await _settle(app, pilot, lambda: app.screen.query("#resume-analyse"))

        # Every button in the row, not only the new one: it takes the cell the
        # spacer used to hold, so adding it re-sizes the three that were there.
        for button_id in (
            "#resume-analyse",
            "#resume-view",
            "#resume-fresh",
            "#resume-cancel",
        ):
            button = app.screen.query_one(button_id, Button)
            drawn = " ".join(
                strip.text
                for strip in button.render_lines(
                    Region(0, 0, button.region.width, button.region.height)
                )
            )
            for word in str(button.label).split():
                assert word in drawn, (
                    f"{word!r} clipped out of {button_id} at {size}: {drawn!r}"
                )


async def test_stored_pages_are_not_offered_when_analysis_is_off(monkeypatch):
    """Both halves are required, or the button promises a pass that will not run.

    With analysis off the run skips those pages exactly as the last one did,
    so offering to analyse them would be the same broken promise in reverse.
    """
    await _fully_stored_with_unread_pages("marcus", pages=2)
    await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press(*"marcus")
        await pilot.press("enter")
        await _settle(app, pilot, lambda: app.screen.query("#resume-detail"))

        assert not app.screen.query("#resume-analyse")
        detail = app.screen.query_one("#resume-detail").render().plain
        assert "Nothing is left to check" in detail


async def test_stored_pages_are_not_offered_without_a_configured_model(
    monkeypatch,
):
    """`run_scan_session` turns analysis off with a warning when no model is
    set, so a dialog offering the pass would be contradicted by the run."""
    from sherlock_project.tui.scan_pane import ScanPane

    await _fully_stored_with_unread_pages("marcus", pages=2)
    await _run_with_prompt(monkeypatch, "marcus")

    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._settings_values.pop("ai.model", None)
        pane._use_ai = True

        await pilot.press(*"marcus")
        await pilot.press("enter")
        await _settle(app, pilot, lambda: app.screen.query("#resume-detail"))

        assert not app.screen.query("#resume-analyse")


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
        await _settle(app, pilot, lambda: app.screen.query("#resume-detail"))

        detail = app.screen.query_one("#resume-detail").render().plain
        assert "Nothing is left to check" in detail
        # No "Resume" button at all; the useful action is to go and look.
        assert not app.screen.query("#resume-continue")
        assert app.screen.query_one("#resume-view", Button)

        await _click_when_drawn(app, pilot, "#resume-view")

        from textual.widgets import TabbedContent

        tabs = app.query_one(TabbedContent)
        # Wait on the tab actually switching, not on a fixed number of pauses:
        # this click dismisses a screen and then activates RESULTS, whose
        # handler reloads the pane off a SQLite read, and `pause()` returns
        # while that read is still outstanding. macOS CI asserted here and got
        # 'tab-scan'.
        await _settle(app, pilot, lambda: tabs.active == "tab-results")

        assert tabs.active == "tab-results"
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
        # `_settle`, not `pause()`: starting a scan hands off to a @work
        # worker, and `wait_for_idle` is satisfied while that worker's I/O is
        # still outstanding -- so the assert could read `captured` before the
        # session had been called at all. That is a KeyError rather than a
        # wrong value, which is how it failed on CI while passing locally.
        await _settle(app, pilot, lambda: "anchors" in captured)
        assert captured["anchors"] == [], "hidden anchors reached the scan"

        # Turned on, the same anchors are used rather than needing retyping.
        await pilot.click("#toggle-ai")
        await pilot.click("#scan-button")
        # Same race, and `captured` is reused across both runs -- so the wait
        # is for the value to CHANGE, not merely for the key to exist.
        await _settle(app, pilot, lambda: bool(captured.get("anchors")))
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
        # Not before the tab bar has applied its own first activation, which
        # lands a few frames after mount and would undo an earlier switch.
        await _settle(app, pilot, lambda: app.tabs_ready)
        await pilot.press("alt+2")
        await _settle(app, pilot, lambda: tabs.active == "tab-results")
        assert tabs.active == "tab-results"
        await pilot.press("alt+3")
        await _settle(app, pilot, lambda: tabs.active == "tab-settings")
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
        pane._reset_for("7ghost")

        title = app.query_one("#feed-title", Static)
        assert "scanning 7ghost" in title.render().plain

        pane._scan_running = False
        pane._redraw_feed_title()

        rendered = app.query_one("#feed-title", Static).render().plain
        assert "scanning" not in rendered
        # The username stays -- these are still 7ghost's results.
        assert "7ghost" in rendered


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
    silent lie about what the next scan does.

    Calls `refresh_settings` directly, so it covers the REDRAW and nothing else.
    What actually reaches that method is a separate question and has its own
    test below -- this one passed throughout the whole time the wiring was
    broken, which is exactly the shape of test that lets a bug ship.
    """
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


def _save_settings_to_disk(**overrides):
    """Write settings exactly the way ^S on the settings pane writes them."""
    from sherlock_project.ai_config import save_settings, try_load_settings
    from sherlock_project.settings import apply_values, field_values

    stored = try_load_settings()
    values = field_values(stored)
    values.update(overrides)
    save_settings(apply_values(stored, values))


async def test_settings_reach_the_scan_tab_by_every_route_not_just_escape():
    """Reported as "I change the model and MODEL does not refresh".

    Settings reach DISK on ^S; the scan pane was told on `SettingsPane.Closed`,
    which only Escape posts. Two different keystrokes, nothing joining them --
    so saving a model and then leaving with alt+1 or a click on the SCAN tab
    left the pane drawing whatever it was built with.

    NOT a display bug, though that is how it surfaces. The same dict is what
    `run_scan_session` reads its transport, concurrency and timeout from, and
    what `ai_is_configured` is asked about -- so a model chosen, saved, and left
    behind with alt+1 gave a scan that warned "no model is configured" and ran
    with analysis off while the model sat on disk. Hence the second assertion:
    the line describing how the scan will RUN has to move too.
    """
    app = SherlockUI()
    async with app.run_test() as pilot:
        from textual.widgets import Static

        pane = app.query_one(ScanPane)
        pane._use_ai = True
        pane._redraw_options()
        await pilot.pause()
        assert "Qwen3-4B-Q4_K_M" not in _model_block(app)

        # On the settings tab, saving as ^S does -- and then leaving by the
        # route that is NOT escape.
        await pilot.press("alt+3")
        _save_settings_to_disk(
            **{
                "ai.base_url": "http://localhost:8080",
                "ai.model": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
                "scan.webbrowser": False,
            }
        )
        await pilot.press("alt+1")
        await pilot.pause()

        assert "Qwen3-4B-Q4_K_M" in _model_block(app)
        # The half that changes what a result MEANS, not just what it says.
        assert "no browser" in app.query_one("#scan-config", Static).render().plain
        assert app._settings_values["ai.model"].endswith("Qwen3-4B-Q4_K_M.gguf")


async def test_unsaved_settings_edits_do_not_reach_the_scan_pane():
    """The reload reads DISK, not the settings pane's working copy.

    An edit that has not been saved is not a setting yet, and the scan pane's
    one job here is describing what the next run will actually do. Showing a
    typed-but-unsaved model would promise analysis that `run_scan_session` --
    which re-reads the stored config itself -- would not deliver.
    """
    app = SherlockUI()
    async with app.run_test() as pilot:
        pane = app.query_one(ScanPane)
        pane._use_ai = True

        await pilot.press("alt+3")
        settings = app.query_one(SettingsPane)
        settings._values["ai.model"] = "typed/but-never-saved.gguf"

        await pilot.press("alt+1")
        await pilot.pause()
        assert "but-never-saved" not in _model_block(app)


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
        # The reload this tab triggers is a worker doing a SQLite read, so a
        # loop of pause() can return with the read still outstanding. This is
        # the right barrier for that, but it was NOT what made this test flaky:
        # the 4-in-12 failure rate measured here was `SherlockDB.connect`
        # racing itself on a database that did not exist yet, and it is fixed
        # in database.py. Kept because waiting on the worker is still the
        # correct thing to do, not because it is load-bearing.
        await _settle(app, pilot, lambda: table.row_count == 1)

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


async def _seed_extraction(username, site, **kwargs) -> None:
    """Attach a stored Pass 1 extraction to an already-seeded site."""
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        rows = await db.get_site_extractions(username)
        site_id = next(row.site_id for row in rows if row.site_name == site)
        await db.update_result_ai_extraction(site_id=site_id, **kwargs)
    finally:
        await db.close()


def _accounts_rows(app) -> dict[str, list[str]]:
    """The SITES table as {site name: cells}.

    Keyed off the site cell rather than the first one -- column 0 is the status
    glyph, which no longer distinguishes rows now that one table carries both
    the hits and the unresolved.
    """
    from textual.widgets import DataTable

    from sherlock_project.tui.results_pane import SEC_SITES

    table = app.query_one(f"#{SEC_SITES}", DataTable)
    rows: dict[str, list[str]] = {}
    for index in range(table.row_count):
        cells = [str(cell) for cell in table.get_row_at(index)]
        rows[cells[1]] = cells
    return rows


async def test_the_accounts_list_says_what_pass_one_made_of_each_site():
    """The judgement that makes someone change model, without opening anything.

    THREE STATES, and the middle one is the point. A blank means nobody has
    asked the model about that page; `0` means it was asked and the page held
    nothing. Collapsing them would turn "not analysed" into "nothing there",
    which is the absence-of-evidence error the SITES counters refuse to make.
    """
    await _seed(
        marcus=[
            ("GitHub", QueryStatus.CLAIMED),
            ("Bandcamp", QueryStatus.CLAIMED),
            ("Zulip", QueryStatus.CLAIMED),
        ]
    )
    await _seed_extraction(
        "marcus",
        "GitHub",
        ai_extraction='{"full_name": ["Marcus Vale"], "emails": ["a@b.c"]}',
        contract_hash="hash",
        model_key="qwen/qwen3-4b",
        reasoning="include Marcus Vale as full_name",
    )
    await _seed_extraction(
        "marcus",
        "Bandcamp",
        ai_extraction="{}",
        contract_hash="hash",
        model_key="qwen/qwen3-4b",
    )

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_results(app, pilot)
        rows = _accounts_rows(app)

        # Two values under two keys is two facts, not two keys.
        assert rows["GitHub"][2] == "2"
        # Asked, found nothing. A result.
        assert rows["Bandcamp"][2] == "0"
        # Never asked. Not the same claim.
        assert rows["Zulip"][2] == "—"


def _extraction_list(app) -> list[list[str]]:
    from textual.widgets import DataTable

    table = app.query_one("#extraction-list", DataTable)
    return [
        [str(cell) for cell in table.get_row_at(index)]
        for index in range(table.row_count)
    ]


def _extraction_panel(app) -> tuple[str, str]:
    from textual.widgets import Static

    return (
        app.query_one("#extraction-detail", Static).render().plain,
        app.query_one("#extraction-summary", Static).render().plain,
    )


async def _open_extractions(pilot):
    """Reach the EXTRACTIONS section by the real binding."""
    from textual.widgets import ContentSwitcher

    from sherlock_project.tui.results_pane import SEC_EXTRACTIONS, SECTION_FOR_TAB

    await _open_results(pilot.app, pilot)
    for _ in range(len(SECTION_FOR_TAB)):
        switcher = pilot.app.query_one("#detail-switch", ContentSwitcher)
        if switcher.current == SEC_EXTRACTIONS:
            break
        await pilot.press("alt+right")
        await _settle(pilot.app, pilot)
    await _settle(pilot.app, pilot)


async def test_the_extraction_panel_ranks_by_yield_not_alphabetically():
    """Both ends of this list are the interesting ones.

    The best extractions are what a prompt edit is judged on; the empty tail is
    what "should I change model" is judged on. Alphabetical order buries both in
    the middle of 149 rows, which is the whole reason this list is not just
    ACCOUNTS again.
    """
    await _seed(
        marcus=[
            ("Alpha", QueryStatus.CLAIMED),
            ("Beta", QueryStatus.CLAIMED),
            ("Gamma", QueryStatus.CLAIMED),
            ("Delta", QueryStatus.CLAIMED),
        ]
    )
    # Alphabetically first, and deliberately the least productive.
    await _seed_extraction(
        "marcus", "Alpha", ai_extraction="{}",
        contract_hash="hash", model_key="qwen/qwen3-4b",
    )
    await _seed_extraction(
        "marcus", "Beta", ai_extraction='{"full_name": ["Marcus"]}',
        contract_hash="hash", model_key="qwen/qwen3-4b",
    )
    await _seed_extraction(
        "marcus",
        "Gamma",
        ai_extraction='{"emails": ["a@b.c", "d@e.f"], "location": ["Lisbon"]}',
        contract_hash="hash",
        model_key="qwen/qwen3-4b",
    )
    # Delta is never analysed at all.

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_extractions(pilot)

        assert [row[1] for row in _extraction_list(app)] == [
            "Gamma",   # 3 facts
            "Beta",    # 1 fact
            "Alpha",   # asked, 0 facts
            "Delta",   # never asked -- last, below the real zero
        ]
        # And the distribution, which is the judgement about the MODEL rather
        # than about any one page.
        _, summary = _extraction_panel(app)
        assert "1 with facts" not in summary
        assert "2 with facts" in summary
        assert "1 empty" in summary
        assert "1 never analysed" in summary


async def test_the_panel_detail_follows_the_cursor():
    """The reason this replaced a dialog: reviewing extractions is a sweep.

    Arrowing down the list changes the reading beside it. Requiring Enter on each
    site would make the panel a dialog with extra steps, and 149 sites would be
    149 open-and-close cycles.
    """
    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Mastodon", QueryStatus.CLAIMED)]
    )
    await _seed_extraction(
        "marcus",
        "GitHub",
        ai_extraction='{"full_name": ["Marcus Vale"]}',
        contract_hash="hash",
        model_key="unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        reasoning="include Marcus Vale as full_name; skip: nav link",
    )
    await _seed_extraction(
        "marcus",
        "Mastodon",
        ai_extraction='{"location": ["Lisbon"]}',
        contract_hash="hash",
        model_key="unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        reasoning="include Lisbon as location",
    )

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_extractions(pilot)

        # Opens on the first row rather than blank: a panel that asks the reader
        # to act before it says anything has wasted the switch.
        detail, _ = _extraction_panel(app)
        assert "Marcus Vale" in detail
        assert "full name" in detail
        # The developer's half -- what moves when the prompt changes.
        assert "skip: nav link" in detail
        # Provenance, with the vendor and repo trimmed off the model key.
        assert "Qwen3-4B-Q4_K_M" in detail
        assert "unsloth" not in detail

        # No Enter. Just the cursor.
        await pilot.press("down")
        for _ in range(4):
            await pilot.pause()
        detail, _ = _extraction_panel(app)
        assert "Lisbon" in detail
        assert "Marcus Vale" not in detail


async def test_a_username_scanned_without_analysis_says_so_once():
    """Rather than 149 rows each reporting "not analysed" individually."""
    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Mastodon", QueryStatus.CLAIMED)]
    )

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_extractions(pilot)
        detail, _ = _extraction_panel(app)
        assert "None of these sites has been analysed" in detail
        assert "Scan this username again with analysis on" in detail


async def test_the_extractions_tab_counts_yield_not_rows():
    """`EXTRACTIONS 8` beside `ACCOUNTS 149` is the quality signal.

    The row count is the account count, which the tab next to it already
    carries. Repeating it would spend the label on a number already on screen.
    """
    from textual.widgets import Tab

    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Mastodon", QueryStatus.CLAIMED)]
    )
    await _seed_extraction(
        "marcus", "GitHub", ai_extraction='{"full_name": ["Marcus"]}',
        contract_hash="hash", model_key="qwen/qwen3-4b",
    )
    await _seed_extraction(
        "marcus", "Mastodon", ai_extraction="{}",
        contract_hash="hash", model_key="qwen/qwen3-4b",
    )

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_results(app, pilot)
        label = str(app.query_one("#tab-extractions", Tab).label)
        assert label == "EXTRACTIONS 1"


async def test_the_viewer_separates_never_analysed_from_found_nothing():
    """Two empty states that mean different things, said differently.

    A page nobody asked the model about is a gap in the analysis. A page the
    model read and found nothing on is a result -- and most pages are that, so
    saying so plainly is what stops it reading as a failure.
    """
    from sherlock_project.database import SiteExtractionRecord
    from sherlock_project.tui.extraction_view import (
        extraction_facts,
        extraction_reasoning,
    )

    def rendered(record):
        return (
            f"{extraction_facts(record).plain}\n"
            f"{extraction_reasoning(record).plain}"
        )

    never = SiteExtractionRecord(
        site_id=1,
        site_name="Zulip",
        site_url=None,
        scanned_at=None,
        ai_extraction=None,
        ai_extraction_contract_hash=None,
        ai_extraction_model=None,
        ai_extraction_reasoning=None,
    )
    assert "has not been analysed" in rendered(never)

    asked = SiteExtractionRecord(
        site_id=2,
        site_name="Bandcamp",
        site_url=None,
        scanned_at=None,
        ai_extraction="{}",
        ai_extraction_contract_hash="hash",
        ai_extraction_model="qwen/qwen3-4b",
        ai_extraction_reasoning=None,
    )
    text = rendered(asked)
    assert "found nothing about the owner" in text
    # And it says WHY there is no reasoning rather than leaving a blank panel:
    # a native-reasoning model is never sent the field at all.
    assert "reasons natively" in text


async def test_the_viewer_compares_the_contract_rather_than_printing_it():
    """64 hex characters answer nothing. "Will a re-scan redo this" does."""
    from sherlock_project.database import SiteExtractionRecord
    from sherlock_project.tui.extraction_view import extraction_provenance

    def record(stored_hash):
        return SiteExtractionRecord(
            site_id=1,
            site_name="GitHub",
            site_url=None,
            scanned_at=None,
            ai_extraction='{"full_name": ["Blue"]}',
            ai_extraction_contract_hash=stored_hash,
            ai_extraction_model="qwen/qwen3-4b",
            ai_extraction_reasoning="include Blue as full_name",
        )

    fresh = extraction_provenance(record("abc"), "abc").plain
    assert "current" in fresh
    assert "abc" not in fresh

    stale = extraction_provenance(record("old"), "abc").plain
    assert "superseded" in stale

    # No current hash available means the line is omitted, never guessed at:
    # there is no honest fallback for "is this extraction current".
    unknown = extraction_provenance(record("old")).plain
    assert "superseded" not in unknown
    assert "current" not in unknown


async def test_switching_username_does_not_carry_extractions_across():
    """Rows are keyed by site name, and two usernames routinely share sites.

    Left behind, `x` on one username's GitHub row would open another's
    extraction -- which is exactly the kind of cross-contamination an evidence
    tool cannot have.
    """
    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
        avery=[("GitHub", QueryStatus.CLAIMED)],
    )
    await _seed_extraction(
        "marcus",
        "GitHub",
        ai_extraction='{"full_name": ["Marcus Vale"]}',
        contract_hash="hash",
        model_key="qwen/qwen3-4b",
        reasoning="include Marcus Vale as full_name",
    )

    app = SherlockUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)

        pane.select_username("marcus")
        await _settle(app, pilot)
        assert pane._extractions["GitHub"].fact_count == 1

        pane.select_username("avery")
        await _settle(app, pilot)
        # avery's GitHub was never analysed, and must not inherit marcus's.
        assert pane._extractions["GitHub"].analysed is False


async def test_a_section_tab_and_its_pane_have_different_ids():
    """`query_one` returns whichever the walk reaches first, not what you meant.

    The section tabs and the panes they select started out sharing one id each,
    which made `query_one("#sec-sites")` ambiguous -- it happened to work, by
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
            if "SITES" not in text:
                continue
            # Only the tab labels. The row also carries the border between the
            # master list and the detail, and that cell legitimately changes
            # shade when the list beside it loses focus -- a different pane's
            # focus ring, not this strip's styling.
            for segment in strip:
                if any(
                    word in segment.text
                    for word in ("SITES", "PROFILE")
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
        await _open_results(app, pilot)
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


async def test_no_scroll_region_is_nested_inside_another():
    """The real lesson from the double-scrollbar defect, in every section.

    THE INVARIANT IS NESTING, not counting, and the earlier version of this test
    asserted the wrong one -- "at most one scroller in the detail pane". That was
    true of the layout at the time and it was never the property that mattered:
    the actual bug was a `DataTable`, which is itself a scrolling viewport, put
    INSIDE a scrolling column, so two vertical bars appeared in one column with
    one wrapping the other.

    Two scrollers side by side in separate grid cells are fine and always were --
    the username list and this detail pane have done exactly that from the start,
    and the EXTRACTIONS section now does it one level down. Counting would have
    forbidden a master-detail layout for no reason; nesting is what looks broken.

    Checked with EVERY section activated, because the old test only ever saw the
    default one and so said nothing about the other three.
    """
    from textual.widgets import ContentSwitcher, Tabs

    from sherlock_project.tui.results_pane import SECTION_FOR_TAB

    await _seed(marcus=[(f"Site{i:02d}", QueryStatus.CLAIMED) for i in range(40)])

    app = SherlockUI()
    async with app.run_test(size=(100, 26)) as pilot:
        await _open_results(app, pilot)

        def scrollers(root):
            return [
                widget
                for widget in root.query("*")
                if widget.show_vertical_scrollbar or widget.show_horizontal_scrollbar
            ]

        for tab_id in SECTION_FOR_TAB:
            app.query_one("#detail-tabs", Tabs).active = tab_id
            for _ in range(6):
                await pilot.pause()
            switcher = app.query_one("#detail-switch", ContentSwitcher)
            assert switcher.current == SECTION_FOR_TAB[tab_id]

            found = scrollers(app.query_one("#result-detail"))
            for widget in found:
                # Nothing that scrolls may contain anything else that scrolls.
                inner = [other for other in scrollers(widget) if other is not widget]
                assert not inner, (
                    f"{type(widget).__name__}#{widget.id} in section {tab_id} "
                    f"wraps {[type(w).__name__ + '#' + str(w.id) for w in inner]}"
                )
            # And never a horizontal one -- cells ellipsize, prose wraps.
            assert not any(w.show_horizontal_scrollbar for w in found), (
                f"section {tab_id} scrolls horizontally: "
                f"{[str(w.id) for w in found if w.show_horizontal_scrollbar]}"
            )


async def test_the_counts_survive_the_merge_into_one_table():
    """The number the two-tab layout existed to keep visible.

    An account list means something different once a site gave no answer, and
    behind a bare tab that number was invisible until someone thought to look
    -- so it used to live on the tab labels. One table makes that arrangement
    unnecessary, and the line above the table carries both numbers instead,
    next to the control that acts on them.
    """
    from textual.widgets import Static

    await _seed(
        marcus=[
            ("GitHub", QueryStatus.CLAIMED),
            ("Reddit", QueryStatus.CLAIMED),
            ("Slow", QueryStatus.UNKNOWN),
        ]
    )
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await _open_results(app, pilot)

        counts = app.query_one("#sites-counts", Static).render().plain
        assert "2 found" in counts
        assert "1 unresolved" in counts


async def test_filtering_to_hits_says_what_it_is_hiding():
    """The whole risk of merging the two lists, closed in one line.

    `show` refuses to let "we could not tell" read as "nobody was home". A
    filter that silently dropped the unanswered rows would undo that in the one
    place someone is most likely to conclude a scan found nothing -- so with
    the filter on, the count of what it is holding back is stated, not dropped.
    """
    from textual.widgets import DataTable, Static

    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)]
    )
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await _open_results(app, pilot)

        table = app.query_one("#sec-sites", DataTable)
        assert table.row_count == 1, "found only should open showing just hits"
        counts = app.query_one("#sites-counts", Static).render().plain
        assert "1 unresolved" in counts and "hidden" in counts

        await pilot.press("f")
        await _settle(app, pilot)

        assert table.row_count == 2, "f did not reveal the unresolved rows"
        assert "hidden" not in app.query_one("#sites-counts", Static).render().plain


async def test_the_filter_is_the_same_control_as_the_scan_toggles():
    """One visual language for "press to change this", across the whole app.

    A ticked or empty box, as on `analysis`, `verbose` and the scan counters.
    The `‹ on ›` brackets it used to wear mean "step through values" on the
    settings editor, where the arrows really do step. The button also reports
    its own state, which is the part a keybinding cannot do.
    """
    from textual.widgets import Button

    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await _open_results(app, pilot)

        toggle = app.query_one("#toggle-found-only", Button)
        assert "chip" in toggle.classes, "not the app's flat chip styling"
        assert str(toggle.label) == "☑ found only"

        await pilot.press("f")
        await _settle(app, pilot)
        # The whole label, not a clipped one.
        assert str(toggle.label) == "☐ found only"
        assert toggle.size.width >= len(str(toggle.label))


async def test_unresolved_sites_are_listed_not_only_counted():
    """There was no in-app equivalent of `show --unresolved`, and a UI needs one
    more than the CLI does -- there is no pipe to fall back on."""
    from textual.widgets import ContentSwitcher, DataTable

    await _seed(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)]
    )
    app = SherlockUI()
    async with app.run_test(size=(110, 30)) as pilot:
        await _open_results(app, pilot)
        await pilot.press("f")
        await _settle(app, pilot)

        assert app.query_one("#detail-switch", ContentSwitcher).current == "sec-sites"
        rows = app.query_one("#sec-sites", DataTable)
        details = " ".join(
            str(cell) for i in range(rows.row_count) for cell in rows.get_row_at(i)
        )
        assert "Slow" in details and "timed out" in details


def _key_text(app) -> str:
    """What the key is actually SHOWING.

    Its `display`, not just its content: a key that is correct and not on
    screen explains exactly as much as no key at all, and reading only the
    renderable would let it be hidden without a single test noticing.
    """
    from textual.widgets import Static

    keys = app.query_one("#sites-key", Static)
    shown = keys.display and app.query_one("#sites-controls").display
    return keys.render().plain if shown else ""


async def test_the_symbol_column_comes_with_a_key():
    """Nothing on this screen said what ▲ meant.

    The mark column is two cells wide and its header is blank, so the
    distinction the whole tool exists to make -- "the site blocked us" is not
    "the rules did not decide" -- was being drawn in symbols the operator had
    never been shown a glossary for. The feed on the scan pane teaches its own
    because it prints the word beside every symbol; a table has no room for
    that, so the glossary goes in the header band instead.
    """
    await _seed(
        marcus=[
            ("GitHub", QueryStatus.CLAIMED),
            ("Slow", QueryStatus.UNKNOWN),
            ("Cloudflared", QueryStatus.WAF),
        ]
    )
    app = SherlockUI()
    async with app.run_test(size=(120, 30)) as pilot:
        await _open_results(app, pilot)

        key = _key_text(app)
        for glyph, word in (
            ("●", "found"),
            ("?", "inconclusive"),
            ("▲", "blocked"),
            ("✕", "rejected"),
        ):
            assert f"{glyph} {word}" in key, f"the key does not explain {glyph}"


async def test_the_key_is_a_glossary_not_a_summary_of_one_record():
    """It names the app's vocabulary, and stays the same size doing it.

    Data-aware, it would have to be rebuilt per username -- and a bordered
    block that changes height between records moves the table underneath it.
    The statuses it names are fixed, and `absent` is not among them: `show`
    does not list absent sites at all, so a key offering `· absent` would name
    a symbol that cannot appear in this table.
    """
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)], quiet=[])
    app = SherlockUI()
    async with app.run_test(size=(120, 30)) as pilot:
        await _open_results(app, pilot)

        key = _key_text(app)
        # Only hits stored, yet the key still explains the rest.
        assert "inconclusive" in key and "rejected" in key
        assert "absent" not in key


async def test_the_key_is_on_the_section_that_draws_the_symbols():
    """On SITES, beside the filter; not on the sections with no symbol column.

    It was a bordered box in the header, on every section -- including
    EXTRACTIONS and PROFILE, where no status symbol is drawn, so it was a
    glossary for nothing on screen that also cost the header four rows.
    """
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED), ("Slow", QueryStatus.UNKNOWN)])
    app = SherlockUI()
    async with app.run_test(size=(120, 30)) as pilot:
        await _open_results(app, pilot)
        assert "found" in _key_text(app)

        await pilot.press("alt+right")
        await _settle(app, pilot)
        assert _key_text(app) == ""

        await pilot.press("alt+left")
        await _settle(app, pilot)
        assert "found" in _key_text(app)


async def test_the_key_moves_before_the_counts_clip():
    """Beside the counts when there is room, under them when there is not.

    The counts line carries the unresolved count, which must never be the
    thing that disappears. So on a short line the key drops to a row of its
    own, and only on a very narrow pane does it go entirely.
    """
    await _seed(marcus=[("GitHub", QueryStatus.CLAIMED)])

    app = SherlockUI()
    async with app.run_test(size=(150, 30)) as pilot:
        await _open_results(app, pilot)
        controls = app.query_one("#sites-controls")
        assert "found" in _key_text(app), "wide enough, and the key is missing"
        assert not controls.has_class("-stacked")

        await pilot.resize_terminal(80, 30)
        await _settle(app, pilot)
        assert "found" in _key_text(app), "the key vanished instead of moving"
        assert controls.has_class("-stacked")

        await pilot.resize_terminal(150, 30)
        await _settle(app, pilot)
        assert not controls.has_class("-stacked"), "the key did not move back"


async def test_a_rejected_username_is_not_drawn_as_inconclusive():
    """The symbol came from a prefix match on its own explanation.

    Anything whose reason did not start with "blocked" was drawn with the
    inconclusive `?`, so a username the site's own rules reject -- which `show`
    reports as "username format rejected" and which has its own ✕ -- arrived
    wearing the symbol for "we could not tell". That is exactly the conflation
    keeping these rows honest exists to prevent, and the key would have printed
    the wrong word beside it just as confidently.
    """
    from textual.widgets import DataTable

    await _seed(marcus=[("StrictSite", QueryStatus.ILLEGAL)])
    app = SherlockUI()
    async with app.run_test(size=(120, 30)) as pilot:
        await _open_results(app, pilot)
        await pilot.press("f")
        await _settle(app, pilot)

        row = app.query_one("#sec-sites", DataTable).get_row_at(0)
        mark = str(row[0])
        assert mark == status_style(QueryStatus.ILLEGAL).glyph
        assert mark != status_style(QueryStatus.UNKNOWN).glyph


async def test_hits_come_before_the_rows_that_answered_nothing():
    """Interleaved by name, a handful of findings scatters through hundreds of
    rows that are not findings."""
    from textual.widgets import DataTable

    await _seed(
        marcus=[
            ("Zulip", QueryStatus.CLAIMED),
            ("Aardvark", QueryStatus.UNKNOWN),
            ("Basecamp", QueryStatus.WAF),
        ]
    )
    app = SherlockUI()
    async with app.run_test(size=(120, 30)) as pilot:
        await _open_results(app, pilot)
        await pilot.press("f")
        await _settle(app, pilot)

        table = app.query_one("#sec-sites", DataTable)
        marks = [str(table.get_row_at(i)[0]) for i in range(table.row_count)]
        assert marks[0] == status_style(QueryStatus.CLAIMED).glyph, (
            f"a hit is not the first row: {marks}"
        )
        assert status_style(QueryStatus.CLAIMED).glyph not in marks[1:]


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

    WorkerCancelled is swallowed because in these flows it is the NORMAL case,
    not a failure. `ResultsPane._load` is `@work(exclusive=True)`, so a second
    reload cancels the first by design, and `Worker.wait()` re-raises that as
    WorkerCancelled -- `wait_for_complete` only absorbs `asyncio.CancelledError`,
    which is a different exception. Treating a superseded worker as an error
    made this helper fail on Windows CI with "Worker was cancelled, and did not
    complete" on a delete that had worked perfectly; the reload simply replaced
    a reload still in flight. WorkerFailed is deliberately NOT caught: that one
    means a worker raised, and hiding it would turn a real error into a silent
    timeout here.
    """
    for _ in range(tries):
        try:
            await app.workers.wait_for_complete()
        except WorkerCancelled:
            pass
        await pilot.pause()
        if predicate is None or predicate():
            return


async def _click_when_drawn(app, pilot, selector: str) -> None:
    """Click a control once it is laid out on screen, not merely composed.

    A dialog's widgets are queryable as soon as it is composed, a refresh before
    they have a region. A click in that gap lands on nothing, silently -- which
    is how "View results" left macOS and Ubuntu CI sitting on the SCAN tab.
    """
    def drawn() -> bool:
        found = app.screen.query(selector)
        return bool(found) and found.first().region.area > 0

    await _settle(app, pilot, drawn)
    assert drawn(), f"{selector} never appeared on screen"
    await pilot.click(selector)


async def _open_results(app, pilot) -> None:
    """Switch to RESULTS without racing the app's own mount.

    `alt+2` activates a tab whose handler reaches for ResultsPane to reload its
    list. Pressed as a test's FIRST action, that can arrive before the DOM has
    finished mounting, and the failure surfaces inside the app rather than the
    test: `NoMatches: No nodes match 'ResultsPane'`.

    A `for _ in range(10): await pilot.pause()` preamble never prevented it --
    `wait_for_idle` is satisfied while the results loader's SQLite read is
    outstanding on a thread executor, so every iteration can pass in
    microseconds with nothing mounted. adf2088 established this and fixed the
    seven sites that open the PROFILE section; these are the rest, found when
    two of them failed on Windows for the same reason.
    """
    from textual.widgets import TabbedContent

    await _settle(app, pilot, lambda: bool(app.query(ResultsPane)) and app.tabs_ready)
    tabs = app.query_one(TabbedContent)
    # Pressed until it ARRIVES. Even with the pane mounted, a key sent in the
    # app's first ticks can be dropped, and every later assertion then ran
    # against the SCAN tab -- focus in the username field, the results pane
    # zero-sized, clicks landing on nothing.
    for _ in range(5):
        if tabs.active == "tab-results":
            break
        await pilot.press("alt+2")
        await _settle(app, pilot, lambda: tabs.active == "tab-results", tries=10)
    assert tabs.active == "tab-results", "RESULTS never opened"
    await _settle(app, pilot)


async def _open_profile_section(app, pilot) -> None:
    """Open RESULTS, then its PROFILE section, waiting on state not on counts.

    Every pane is composed at app start, so `alt+2` is normally safe. But a key
    pressed before the DOM has finished mounting reaches an app that cannot
    answer for it: activating RESULTS runs a handler that reaches for
    ResultsPane to reload its list, and on the slowest runner that raised
    `NoMatches: No nodes match 'ResultsPane'` -- inside the app, not the test.

    So the wait is for the pane to EXIST before driving it, and then for the
    section to have actually switched. A `for _ in range(12): await
    pilot.pause()` preamble never guaranteed either: `wait_for_idle` is
    satisfied while the results loader's SQLite read is outstanding on a thread
    executor, so all twelve iterations can pass in microseconds with nothing
    mounted.
    """
    from sherlock_project.tui.results_pane import SEC_PROFILE, SECTION_FOR_TAB

    await _settle(app, pilot, lambda: bool(app.query(ResultsPane)))
    await pilot.press("alt+2")
    await _settle(app, pilot, lambda: bool(app.query("#profile-actions")))
    # Pressed until it ARRIVES, never a counted number of times. The count is
    # the section count, and that has changed twice already: ACCOUNTS and
    # UNRESOLVED became one SITES section, then EXTRACTIONS went in between
    # SITES and PROFILE. Each time, a counted walk landed a section short and
    # the tests then asserted against widgets belonging to a section that was
    # not showing -- failing somewhere unrelated to what they were testing.
    for _ in range(len(SECTION_FOR_TAB)):
        if app.query_one("#detail-switch").current == SEC_PROFILE:
            break
        await pilot.press("alt+right")
        await _settle(app, pilot)
    await _settle(
        app,
        pilot,
        lambda: app.query_one("#detail-switch").current == SEC_PROFILE,
    )


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
        await _open_results(app, pilot)

        await pilot.press("delete")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfirmScreen)

        # Cancel holds focus, not the destructive control.
        assert app.focused is screen.query_one("#confirm-no", Button)
        # It names what will go -- in the title, rather than asking the
        # title's question again in the body -- and counts it.
        title = str(screen.query_one(".dialog-title").render())
        assert "marcus" in title
        # The button is the verb alone, and the pair is one fixed size: a
        # username in the label made the destructive button as wide as the
        # name.
        yes = screen.query_one("#confirm-yes", Button)
        no = screen.query_one("#confirm-no", Button)
        assert str(yes.label) == "Delete"
        assert yes.size.width == no.size.width
        detail = screen.query_one("#confirm-detail").render().plain
        assert "1 found account" in detail
        assert "cannot be undone" in detail
        # And it looks like what it is: the danger frame, not the amber one
        # the harmless dialogs wear.
        assert screen.has_class("-danger")

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
        await _open_results(app, pilot)

        table = app.query_one("#username-list", DataTable)
        await _settle(app, pilot, lambda: table.row_count == 2)
        assert table.row_count == 2
        selected = app.query_one(ResultsPane)._selected

        await pilot.press("delete")
        await _settle(app, pilot, lambda: isinstance(app.screen, ConfirmScreen))
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


@asynccontextmanager
async def _list_with(**rows):
    """The results tab, open, with its list loaded.

    A context manager rather than the one-shot async generator the filter tests
    use: these tests wait on the list with `_settle`, and a predicate that
    closes over a variable bound by `async for` is a closure over a loop
    variable, which is a lint error.
    """
    from textual.widgets import DataTable

    await _seed(**rows)
    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        # Through `_open_results`, not a bare first-tick `alt+2`, which was
        # sometimes dropped: the list still loaded behind the SCAN tab, so the
        # wait below passed and every test using this ran on the wrong tab.
        await _open_results(app, pilot)
        table = app.query_one("#username-list", DataTable)
        await _settle(app, pilot, lambda: table.row_count == len(rows))
        assert table.row_count == len(rows)
        yield app, table, pilot


async def _open_actions(app, pilot):
    from sherlock_project.tui.record_actions import RecordActionsScreen

    # The chip appears with the RECORD, a second read after the list. Clicking
    # before it arrives clicks nothing.
    pane = app.query_one(ResultsPane)
    await _settle(
        app,
        pilot,
        lambda: pane._record is not None
        and app.query_one("#record-actions").display
        and app.query_one("#record-actions").region.area > 0,
    )
    await _settle(app, pilot)
    await pilot.click("#record-actions")
    await _settle(app, pilot, lambda: isinstance(app.screen, RecordActionsScreen))
    assert isinstance(app.screen, RecordActionsScreen)
    return app.screen


async def _choose(app, pilot, option_id: str) -> None:
    from textual.widgets import OptionList

    menu_screen = app.screen
    menu = menu_screen.query_one(OptionList)
    menu.highlighted = menu.get_option_index(option_id)
    await pilot.press("enter")
    if option_id == "delete":
        # Dismissing the menu and pushing the confirmation are two steps; wait
        # for the second, not just the first.
        from sherlock_project.tui.confirm_screen import ConfirmScreen

        await _settle(app, pilot, lambda: isinstance(app.screen, ConfirmScreen))
    else:
        await _settle(app, pilot, lambda: app.screen is not menu_screen)


async def test_the_actions_menu_appears_with_a_record_and_not_before():
    """Everything that can be done to a record belongs to a record.

    Under a "Nothing scanned yet" sentence it would be a menu with nothing to
    act on, and its last item is the irreversible kind.
    """
    from textual.widgets import Button

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.press("alt+2")
        await _settle(app, pilot)
        assert app.query_one("#record-actions", Button).display is False

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
    ) as (app, _table, pilot):
        # The menu arrives with the RECORD, which is a second read after the
        # list -- waiting on the list alone raced it on a slow runner.
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        await _settle(app, pilot)
        assert app.query_one("#record-actions", Button).display is True


async def test_delete_is_never_a_tab_stop():
    """It used to be the first stop after the username list.

    A permanent red "Delete this username" button sat directly under the name,
    so one press of Tab from the list armed the only irreversible action in the
    app. It now lives last in the Actions menu, and neither the menu's chip nor
    anything on the way to the table can be reached by Tab.
    """
    from textual.widgets import Button

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
    ) as (app, table, pilot):
        table.focus()
        await pilot.pause()
        seen = []
        for _ in range(4):
            await pilot.press("tab")
            await pilot.pause()
            seen.append(app.focused)
        assert app.query_one("#record-actions", Button) not in seen
        assert all("delete" not in str(getattr(w, "id", "")) for w in seen)


async def test_delete_is_last_in_the_menu_and_not_where_it_opens():
    """Safest first, Delete last and set apart -- and the menu opens on the
    first row, so Enter straight after opening it cannot erase anything."""
    from textual.widgets import OptionList

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
    ) as (app, _table, pilot):
        await _open_actions(app, pilot)
        menu = app.screen.query_one(OptionList)
        ids = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        assert ids[-1] == "delete"
        assert menu.highlighted == 0
        assert ids[0] != "delete"


async def test_the_delete_action_asks_about_the_record_on_screen():
    """It acts on whatever the header names, so changing rows changes its target.

    The thing it deletes is the thing being read, and the dialog says which --
    "are you sure?" with no name is a question nobody can answer.
    """
    from sherlock_project.database import default_database_path
    from sherlock_project.tui.confirm_screen import ConfirmScreen

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED), ("Reddit", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
    ) as (app, table, pilot):
        pane = app.query_one(ResultsPane)
        opened = pane._selected
        assert opened is not None

        await _open_actions(app, pilot)
        await _choose(app, pilot, "delete")
        assert isinstance(app.screen, ConfirmScreen)
        assert opened in str(app.screen.query_one(".dialog-title").render())

        await pilot.press("escape")
        await _settle(app, pilot)

        table.focus()
        await pilot.press("down")
        await _settle(app, pilot, lambda: pane._selected != opened)
        other = pane._selected
        assert other is not None and other != opened

        await _open_actions(app, pilot)
        await _choose(app, pilot, "delete")
        assert isinstance(app.screen, ConfirmScreen)
        title = str(app.screen.query_one(".dialog-title").render())
        assert other in title
        assert opened not in title, "the action is still aimed at the old row"
        assert "cannot be undone" in app.screen.query_one(
            "#confirm-detail"
        ).render().plain

        await pilot.press("escape")
        await _settle(app, pilot)

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert len(await db.list_usernames()) == 2, "cancelling deleted something"
    finally:
        await db.close()


async def test_confirming_from_the_menu_erases_the_open_username():
    """The menu item and the key are one action, so confirming does one thing."""
    from textual.widgets import Button

    from sherlock_project.database import default_database_path

    async with _list_with(
        marcus=[("GitHub", QueryStatus.CLAIMED)],
        keeper=[("Reddit", QueryStatus.CLAIMED)],
    ) as (app, table, pilot):
        pane = app.query_one(ResultsPane)
        doomed = pane._selected

        await _open_actions(app, pilot)
        await _choose(app, pilot, "delete")
        await pilot.click("#confirm-yes")
        await _settle(app, pilot, lambda: table.row_count == 1)

        assert table.row_count == 1
        # A record still stands, so the menu is still on offer for it.
        assert app.query_one("#record-actions", Button).display is True

    db = await SherlockDB.create(str(default_database_path()))
    try:
        remaining = [item.username for item in await db.list_usernames()]
    finally:
        await db.close()
    assert doomed not in remaining


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


async def _results_with(
    username: str, *, extractions: int = 0, unanalysed: int = 0
):
    """Seed a username, optionally with stored Pass 1 evidence.

    Three states, because the profile pane has to tell them apart:
      - neither         -> one hit whose page was never kept; nothing to work
                           from, and scanning again really is the fix
      - `extractions`   -> pages stored AND already read
      - `unanalysed`    -> pages stored and NEVER read, which is what a scan
                           without analysis leaves behind and what an analysis
                           pass consumes without re-fetching anything

    The real contract hash is used rather than a stand-in, because eligibility
    is now read back through it: a row stored under a fake hash looks stale,
    and would count as pending work that is not pending.
    """
    from sherlock_project.ai_engine import pass_one_contract_hash
    from sherlock_project.database import SherlockDB, default_database_path

    contract_hash = pass_one_contract_hash()
    seeded = extractions + unanalysed
    db = await SherlockDB.create(str(default_database_path()))
    try:
        for index in range(max(1, seeded)):
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
                response_text="<html>Avery Stone</html>" if seeded else None,
                transport="browser",
            )
        if extractions:
            pending = await db.get_pending_ai_extraction_ids(
                username, contract_hash=contract_hash
            )
            for site_id in pending[:extractions]:
                await db.update_result_ai_extraction(
                    site_id,
                    '{"full_name": ["Avery Stone"]}',
                    contract_hash=contract_hash,
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
    from textual.geometry import Region
    from textual.widgets import Button, Static

    await _results_with("noevidence")

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        # This fixture kept no page, so there is genuinely nothing to read and
        # a scan is the honest answer. The wording says which of the two
        # no-evidence states this is.
        assert "None of the stored pages gave any facts" in hint
        # Named for the trip it makes, not for a build it cannot do.
        assert "Scan this username with analysis" in str(
            app.query_one("#profile-build", Button).label
        )
        # No point offering anchors for a build that cannot happen.
        assert app.query_one("#profile-anchors", Button).display is False
        # And the pane says so once. The profile block's "No profile stored,
        # scanning builds one" is wrong here -- this username HAS been
        # scanned -- so the actions block is left to answer alone.
        block = app.query_one("#detail-profile", Static)
        drawn = " ".join(
            strip.text
            for strip in block.render_lines(
                Region(0, 0, block.region.width, block.region.height)
            )
        )
        assert "No profile stored" not in drawn


async def test_stored_pages_are_read_by_the_build_itself():
    """Unread stored pages are not a reason to leave the pane any more.

    It used to point at the SCAN tab to analyse them -- a trip for work that
    needs no network. Build now reads them first, then merges, which is the
    order a scan with analysis keeps; the card says so before you press it.
    """
    from textual.widgets import Button, Static

    await _results_with("unread", unanalysed=3)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "None of its stored pages has been analysed yet" in hint
        assert "+ 3 stored pages to read first" in hint
        assert "reads them, then merges" in hint
        build = app.query_one("#profile-build", Button)
        assert str(build.label) == "Build profile"
        # A real build now, so anchors are worth offering.
        assert app.query_one("#profile-anchors", Button).display is True

async def test_one_stored_page_is_not_described_in_the_plural():
    """"1 pages" is how a careful tool looks careless."""
    from textual.widgets import Static

    await _results_with("single", unanalysed=1)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "+ 1 stored page to read first" in hint

async def test_the_pointer_state_does_not_stick_to_the_next_username():
    """The flat treatment is a state, not a setting.

    It is carried by a class on `#profile-actions`, and a class that is added
    on one row and never removed is the classic way a master/detail pane starts
    lying: click a username with no evidence, click one with evidence, and the
    commit button would still be wearing the pointer's flat chrome -- and the
    Anchors button would still be laid out beside a spacer the stylesheet had
    hidden. Both directions are checked, because only removing it is the bug.
    """
    from textual.widgets import Button

    from sherlock_project.tui.results_pane import NO_EVIDENCE

    await _results_with("bare")
    await _results_with("stocked", extractions=2)

    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        await _open_profile_section(app, pilot)

        pane = app.query_one(ResultsPane)
        actions = app.query_one("#profile-actions")
        build = app.query_one("#profile-build", Button)

        for username, pointer in (("bare", True), ("stocked", False),
                                  ("bare", True)):
            pane.select_username(username)
            for _ in range(14):
                await pilot.pause()
            assert actions.has_class(NO_EVIDENCE) is pointer, username
            # And the layout that the class drives actually followed it.
            assert build.region.height == (1 if pointer else 3), username


@pytest.mark.parametrize("size", [(110, 34), (80, 30), (140, 40)])
async def test_the_pointer_button_draws_its_whole_label(size):
    """The pixels, unusually -- because the attribute was never the bug.

    This file prefers assertions about decisions, and one holds here: a control
    that cannot say what it does is not a control. But `Button.label` read back
    as "Scan with analysis" for the entire time the screen said "Scan with".
    `#profile-buttons` is a `1fr 12 20` grid and Textual SKIPS hidden children
    when it assigns cells, so hiding `#profile-anchors` in this state slid the
    build button out of the 20-cell column into the 12-cell one: ten usable
    cells for an eighteen-character label, clipped with no ellipsis to admit it.
    Only a render can catch that, and only across widths -- a fixed column hides
    the fault at whatever size it was last eyeballed at.
    """
    from textual.geometry import Region
    from textual.widgets import Button

    await _results_with("noevidence")

    app = SherlockUI()
    async with app.run_test(size=size) as pilot:
        await _open_profile_section(app, pilot)
        # The pointer state is drawn when the RECORD arrives, after the list.
        # Read before that, the button has no region and "draws" nothing --
        # which looked like a clipped label and was only a slow read.
        pane = app.query_one(ResultsPane)
        button = app.query_one("#profile-build", Button)
        await _settle(
            app,
            pilot,
            lambda: pane._record is not None and button.region.area > 0,
        )
        await _settle(app, pilot)

        drawn = " ".join(
            strip.text
            for strip in button.render_lines(
                Region(0, 0, button.region.width, button.region.height)
            )
        )
        for word in str(button.label).split():
            assert word in drawn, (
                f"{word!r} clipped out of the button at {size}: {drawn!r}"
            )


async def test_the_pointer_button_stays_operable_by_mouse_and_keyboard():
    """Flattening the chrome must not flatten the affordance.

    It loses its border here, which is a look, not a demotion: it is still a
    Button, so it stays in the Tab order and stays pressable both ways. The
    focus rule matters as much as the hover one -- the ID selector that styles
    it outranks Textual's `Button:focus`, so without an explicit focus style
    the only remaining cue is an 8/255 background shift and a keyboard user is
    left with no idea where they are.
    """
    from textual.widgets import Button

    await _results_with("noevidence")

    app = SherlockUI()
    async with app.run_test(size=(110, 34)) as pilot:
        await _open_profile_section(app, pilot)
        pane = app.query_one(ResultsPane)
        # The pointer state is drawn when the RECORD arrives, after the list.
        await _settle(app, pilot, lambda: pane._record is not None)
        await _settle(app, pilot)

        button = app.query_one("#profile-build", Button)
        assert button in app.screen.focus_chain

        app.set_focus(button)
        for _ in range(3):
            await pilot.pause()
        focused = button.styles.background
        app.set_focus(None)
        for _ in range(3):
            await pilot.pause()
        blurred = button.styles.background
        # Not merely different -- different enough to see across the row.
        assert focused != blurred
        delta = abs(
            focused.rgb[0] * 0.2126
            + focused.rgb[1] * 0.7152
            + focused.rgb[2] * 0.0722
            - (
                blurred.rgb[0] * 0.2126
                + blurred.rgb[1] * 0.7152
                + blurred.rgb[2] * 0.0722
            )
        )
        assert delta > 15, f"focus is invisible: {blurred} -> {focused}"


async def test_stored_evidence_offers_a_build_and_says_it_is_instant():
    """The unanchored path calls no model at all -- `run_synthesis_only` loads
    one only when anchors make it necessary. Worth saying, because "build a
    profile" otherwise reads as a slow operation."""
    from textual.widgets import Button, Static

    await _results_with("hasevidence", extractions=3)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "3 sites gave facts that are ready to merge" in hint
        assert "instant" in hint and "no model needed" in hint
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
        await _open_results(app, pilot)
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
        await _open_results(app, pilot)

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
        await _open_results(app, pilot)

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

    async def slow_synthesis(*, usernames, force, inline_anchors, reporter=None, **_):
        started.set()
        if reporter is not None:
            reporter.ai_model_starting()
        await release.wait()
        return {}

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", slow_synthesis)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

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
        # Build is gone, so a second press cannot start a competing synthesis
        # over the same rows -- and Stop has taken its place in the row.
        assert app.query_one("#profile-build").display is False
        assert app.query_one("#profile-stop").display is True

        release.set()
        for _ in range(20):
            await pilot.pause()


async def test_a_failed_build_leaves_the_reason_on_screen(monkeypatch):
    """A failure that disappears after five seconds is one nobody can act on."""
    from textual.widgets import Static

    from sherlock_project import sherlock as sherlock_module

    await _results_with("badbuild", extractions=1)

    async def failing(*, usernames, force, inline_anchors, reporter=None, **_):
        raise RuntimeError("LM Studio is not running")

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", failing)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        await pilot.click("#profile-build")
        for _ in range(25):
            await pilot.pause()

        shown = app.query_one("#profile-status", Static).render().plain
        assert "LM Studio is not running" in shown
        # And the controls come back, so it can be tried again.
        assert app.query_one("#profile-buttons").display is True


async def test_a_synthesis_that_failed_is_not_announced_as_a_built_profile(
    monkeypatch,
):
    """`synthesize_profiles` isolates each username and does not re-raise.

    So the absence of an exception says nothing about whether anything was
    built, and reading it as success is how this button came to report
    "Profile built for x" over a failure that had left the old profile in
    place. The reporter knew; nothing that could act on it did.
    """
    from textual.widgets import Static

    from sherlock_project import sherlock as sherlock_module
    from sherlock_project.tui.results_pane import ResultsPane

    await _results_with("quietfail", extractions=1)

    async def failed_but_returned(*, usernames, force, inline_anchors, reporter=None, **_):
        return {usernames[0]: RuntimeError("the model returned nothing usable")}

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", failed_but_returned)
    notified: list[str] = []
    monkeypatch.setattr(
        ResultsPane,
        "notify",
        lambda self, message, *args, **kwargs: notified.append(str(message)),
    )

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _open_profile_section(app, pilot)

        await pilot.click("#profile-build")
        status = app.query_one("#profile-status", Static)
        await _settle(
            app, pilot, lambda: "nothing usable" in status.render().plain
        )

        assert "the model returned nothing usable" in status.render().plain
        assert app.query_one("#profile-buttons").display is True
        assert not any("Profile built" in message for message in notified)


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
    assert "sherlock-rm <username>" in capsys.readouterr().out


async def test_unknown_arguments_are_reported_rather_than_ignored(capsys):
    """Someone typing `sherlock-rm ui --fresh` has an expectation about that run
    which this cannot meet, so silently dropping the flag would be worse."""
    console = Console(force_terminal=False, no_color=True)
    assert await run_ui(["--fresh"], console=console) == 2
    assert "takes no arguments" in capsys.readouterr().out


# -- the startup update check -------------------------------------------------


def _enable_update_check(enabled: bool = True) -> None:
    """Write a config with the update toggle in a known state.

    Written rather than monkeypatched because the whole point of the default is
    that it is read off the config file the same way every other setting is.
    """
    from pathlib import Path

    from sherlock_project.ai_config import (
        SherlockSettings,
        UpdateSettings,
        ai_config_path,
        save_settings,
    )

    save_settings(
        SherlockSettings(update=UpdateSettings(check_on_startup=enabled)),
        path=Path(str(ai_config_path())),
        environ={},
    )


def _a_newer_version() -> str:
    """A version this build will accept as newer, whatever this build is.

    Derived rather than written down. These tests originally hardcoded
    "0.2.0" as the newer side, which held only while the package was 0.1.0 --
    the first real version bump turned three of them red, because 0.2.0 had
    become what was installed and `is_newer` correctly refused to call it an
    upgrade. Bumping the major keeps the comparison true for every version
    this package will ever carry.
    """
    from sherlock_project import __version__
    from sherlock_project.updater import parse_version

    numbers = parse_version(__version__) or (0,)
    return ".".join(str(part) for part in (numbers[0] + 1, 0, 0))


def _stub_release(monkeypatch, version: str | None = None):
    """Answer the forge without going near it, and record that it was asked."""
    import sherlock_project.updater as updater_module
    from sherlock_project.updater import Release

    version = _a_newer_version() if version is None else version
    asked: list[str] = []

    async def fake_fetch(*, timeout: float = 10.0):
        asked.append("asked")
        return Release(
            tag=f"v{version}",
            version=version,
            url=f"https://example.invalid/releases/tag/v{version}",
        )

    monkeypatch.setattr(updater_module, "fetch_latest", fake_fetch)
    return asked


async def test_the_check_does_not_run_when_the_setting_is_off(monkeypatch):
    """The default, and the reason every other test in this file is offline.

    Asserting the forge was never ASKED, not merely that no dialog appeared: a
    check that runs and finds nothing looks identical on screen to one that
    never ran, and only one of those keeps a promise about what leaves the
    machine.
    """
    asked = _stub_release(monkeypatch)
    _enable_update_check(False)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot)

    assert asked == []


async def test_an_available_release_is_offered_and_nothing_is_installed(
    monkeypatch,
):
    """Found is not installed. The dialog is the whole of what a check may do
    on its own; replacing the program takes a press."""
    from textual.widgets import Button

    from sherlock_project.tui.update_screen import UpdateScreen

    asked = _stub_release(monkeypatch)
    _enable_update_check(True)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot, lambda: isinstance(app.screen, UpdateScreen))

        assert asked == ["asked"]
        assert isinstance(app.screen, UpdateScreen)
        # The committing control is not the one under the finger on open.
        assert app.focused is app.screen.query_one("#update-no", Button)
        # And the appbar has not claimed anything happened.
        assert app._update_installed is False


async def test_a_release_that_is_not_newer_is_not_offered(monkeypatch):
    """Equal versions are the common case on every launch after the first, and
    the check that this replaced would have offered a dialog for them."""
    from sherlock_project import __version__
    from sherlock_project.tui.update_screen import UpdateScreen

    asked = _stub_release(monkeypatch, version=__version__)
    _enable_update_check(True)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot)

        assert asked == ["asked"]
        assert not isinstance(app.screen, UpdateScreen)


async def test_an_unreachable_forge_is_silent(monkeypatch):
    """No release published yet is a 404, which is the single most likely
    answer on a repository that has never been tagged. It is not news, and an
    update check that raises a dialog saying it failed has cost more than it
    is worth."""
    import sherlock_project.updater as updater_module
    from sherlock_project.tui.update_screen import UpdateScreen

    async def no_answer(*, timeout: float = 10.0):
        return None

    monkeypatch.setattr(updater_module, "fetch_latest", no_answer)
    _enable_update_check(True)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot)

        assert not isinstance(app.screen, UpdateScreen)
        assert app._update_installed is False


async def test_a_finished_install_leaves_the_restart_note_in_the_appbar(
    monkeypatch,
):
    """The running process is still the old build, and stays that way until
    someone restarts it -- so the note has to persist rather than toast, and it
    has to name something doable from where the reader is sitting.
    """
    from textual.widgets import Static

    import sherlock_project.tui.update_screen as screen_module
    import sherlock_project.updater as updater_module
    from sherlock_project.tui.update_screen import UpdateScreen

    _stub_release(monkeypatch)
    _enable_update_check(True)
    monkeypatch.setattr(updater_module, "detect_install", lambda **kw: "pipx")
    monkeypatch.setattr(screen_module, "updater_available", lambda mode: True)

    async def clean_install(command, *, on_line=None):
        if on_line is not None:
            on_line("Installing collected packages: sherlock-rm")
        return 0, "Installing collected packages: sherlock-rm"

    monkeypatch.setattr(screen_module, "run_install", clean_install)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot, lambda: isinstance(app.screen, UpdateScreen))
        await pilot.click("#update-yes")
        await _settle(app, pilot, lambda: app._update_installed)

        assert app._update_installed is True
        drawn = app.query_one("#appbar", Static).render()
        text = drawn.plain if hasattr(drawn, "plain") else str(drawn)
        assert "updated" in text
        # Names a key the reader has, not a command line they are not at.
        assert "alt+q" in text


async def test_a_failed_install_says_so_and_does_not_claim_success(monkeypatch):
    """A failure that vanishes is a failure nobody can act on -- and this one
    has a real chance of being a locked file on Windows, where the files being
    replaced belong to the process replacing them."""
    from textual.widgets import Static

    import sherlock_project.tui.update_screen as screen_module
    import sherlock_project.updater as updater_module
    from sherlock_project.tui.update_screen import UpdateScreen

    _stub_release(monkeypatch)
    _enable_update_check(True)
    monkeypatch.setattr(updater_module, "detect_install", lambda **kw: "pipx")
    monkeypatch.setattr(screen_module, "updater_available", lambda mode: True)

    async def broken_install(command, *, on_line=None):
        return 1, "ERROR: could not install packages due to an OSError"

    monkeypatch.setattr(screen_module, "run_install", broken_install)

    app = SherlockUI()
    async with app.run_test() as pilot:
        await _settle(app, pilot, lambda: isinstance(app.screen, UpdateScreen))
        screen = app.screen
        await pilot.click("#update-yes")
        await _settle(
            app,
            pilot,
            lambda: "failed" in screen.query_one("#update-status", Static)
            .render()
            .plain,
        )

        shown = screen.query_one("#update-status", Static).render().plain
        assert "failed" in shown
        assert "OSError" in shown
        # The way out is named, and the app has not pretended it updated.
        assert "pipx install --force" in shown
        assert app._update_installed is False


async def test_rescan_drops_rows_the_site_list_no_longer_covers(monkeypatch):
    """"Re-scan all" is what finally removes a retired site's stored row.

    A username scanned under two different site lists holds the union of both,
    duplicates included, and nothing was ever removing the half that no longer
    exists. A re-scan rewrites every site the list still has, so the leftovers
    are exactly the rows it would otherwise leave behind.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for site_name in ("GitHub", "threads", "Ask.fm"):
            await db.save_result(
                username="someone",
                site_name=site_name,
                status=str(QueryStatus.CLAIMED),
                response_text="profile",
            )
    finally:
        await db.close()

    async def scan_spy(**kwargs):
        return {}

    await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": False},
        scan_spy=scan_spy,
        sites=("GitHub",),
        fresh=True,
    )

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert sorted(await db.get_saved_results("someone")) == ["GitHub"]
    finally:
        await db.close()


async def test_a_resumed_scan_removes_nothing(monkeypatch):
    """Only a re-scan prunes.

    A resume was not asked to be destructive, and its stored rows are the
    report -- dropping them mid-run would delete evidence the same run is about
    to present as restored results.
    """
    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        for site_name in ("GitHub", "threads"):
            await db.save_result(
                username="someone",
                site_name=site_name,
                status=str(QueryStatus.CLAIMED),
                response_text="profile",
                transport="browser",
            )
    finally:
        await db.close()

    async def scan_spy(**kwargs):
        return {}

    await _run_with_fakes(
        monkeypatch,
        {"scan.webbrowser": True},
        scan_spy=scan_spy,
        sites=("GitHub",),
        fresh=False,
    )

    db = await SherlockDB.create(str(default_database_path()))
    try:
        assert sorted(await db.get_saved_results("someone")) == ["GitHub", "threads"]
    finally:
        await db.close()


async def test_the_prune_set_is_the_manifest_before_the_nsfw_filter():
    """Pruning against what a run CHECKS would retire an earlier --nsfw run.

    `site_data_all` is the scan set and shrinks with the NSFW setting;
    `known_site_names` is what the site list still covers and does not. Reading
    the first would make an ordinary safe-for-work re-scan silently delete
    results the user deliberately went and collected.
    """
    from sherlock_project.tui.runner import build_scan_plan

    plan = await build_scan_plan(
        username="nobody", settings_values={"scan.nsfw": False}
    )

    assert plan.known_site_names > set(plan.site_data_all)
    assert len(plan.known_site_names) - len(plan.site_data_all) > 0


# -- navigation, focus and narrow layouts -------------------------------------
#
# Each of these was found by driving the app with the keyboard alone. None of
# them shows up in a test that only asserts what a widget holds: the content was
# right every time, and the screen still could not be used.


async def _seed_built_profile(username: str) -> None:
    from sherlock_project.database import SherlockDB, default_database_path

    await _results_with(username, extractions=2)
    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.update_username_profile_summary(
            username=username,
            profile_summary=(
                f'{{"username": "{username}", "input_hash": "h", "mode": "aggregate",'
                ' "resolution_status": "aggregated", "completeness": "partial",'
                ' "strong_profile": {"full_name": ["Avery"]},'
                ' "warnings": ["a diagnostic note"]}'
            ),
            input_hash="h",
        )
    finally:
        await db.close()


async def test_the_profile_section_holds_focus_so_its_keys_are_heard():
    """PROFILE was a keyboard trap.

    Switching to it hid the extraction list that had focus, and Textual then
    focused nothing -- so alt+left, Tab, `v` and `s` all went unheard, and the
    only ways out were the mouse or another tab. `v` and `s` exist only for
    this section and could never be pressed in it.
    """
    from textual.widgets import Static

    await _seed_built_profile("trapped")
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_profile_section(app, pilot)
        assert app.focused is app.query_one("#sec-profile")

        before = app.query_one("#detail-profile", Static).render()
        await pilot.press("v")
        await _settle(app, pilot)
        assert app.query_one(ResultsPane)._show_notes is True
        after = app.query_one("#detail-profile", Static).render()
        assert str(after) != str(before), "v changed nothing on screen"

        await pilot.press("alt+left")
        await _settle(app, pilot)
        assert app.query_one("#detail-switch").current == "sec-extractions"


async def test_the_footer_offers_a_sections_keys_only_on_that_section():
    """A footer offering `found only` on PROFILE and `notes` on SITES taught
    people that keys on this screen do nothing."""
    await _seed_built_profile("scoped")
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        assert pane.check_action("toggle_found_only", ()) is True
        assert pane.check_action("toggle_notes", ()) is False

        await _open_profile_section(app, pilot)
        assert pane.check_action("toggle_found_only", ()) is False
        assert pane.check_action("toggle_notes", ()) is True
        assert pane.check_action("build_profile", ()) is True


async def test_coming_back_to_results_keeps_your_place():
    """Every visit to RESULTS reloads it, and it used to land on row 0.

    Checking the scan tab and coming back lost the username you were reading.
    """
    from textual.widgets import DataTable

    await _seed(
        first=[("GitHub", QueryStatus.CLAIMED)],
        second=[("GitHub", QueryStatus.CLAIMED)],
        third=[("GitHub", QueryStatus.CLAIMED)],
    )
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_results(app, pilot)
        table = app.query_one("#username-list", DataTable)
        await _settle(app, pilot, lambda: table.row_count == 3)
        table.focus()
        from textual.widgets import TabbedContent

        tabs = app.query_one(TabbedContent)
        pane = app.query_one(ResultsPane)
        await pilot.press("down", "down")
        chosen = str(table.coordinate_to_cell_key((2, 0)).row_key.value)
        await _settle(app, pilot, lambda: pane._selected == chosen)
        assert pane._selected == chosen

        await pilot.press("alt+1")
        await _settle(app, pilot, lambda: tabs.active == "tab-scan")
        await pilot.press("alt+2")
        await _settle(app, pilot, lambda: tabs.active == "tab-results")
        await _settle(app, pilot, lambda: not pane._loading)
        await _settle(app, pilot)
        assert pane._selected == chosen
        assert table.cursor_row == 2


async def test_a_new_scan_at_the_top_is_opened_on_arrival():
    """The exception to keeping your place: a username that has just risen to
    the top of a most-recent-first list is a scan that just finished, and it is
    what someone came to RESULTS to read."""
    from textual.widgets import DataTable

    await _seed(older=[("GitHub", QueryStatus.CLAIMED)])
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._selected == "older")

        from textual.widgets import TabbedContent

        tabs = app.query_one(TabbedContent)
        await pilot.press("alt+1")
        await _settle(app, pilot, lambda: tabs.active == "tab-scan")
        await _seed(newest=[("GitHub", QueryStatus.CLAIMED)])
        await pilot.press("alt+2")
        await _settle(app, pilot, lambda: tabs.active == "tab-results")
        table = app.query_one("#username-list", DataTable)
        await _settle(app, pilot, lambda: table.row_count == 2)
        await _settle(app, pilot, lambda: pane._selected == "newest")
        assert pane._selected == "newest"


async def test_the_unresolved_count_survives_eighty_columns():
    """At 80 columns it was clipped clean off: "6 found ·" and then nothing.

    It is the number that keeps "no answer" from reading as "no account", and
    the one on that line the design says must never hide.
    """
    from textual.widgets import Static

    await _seed(
        narrow=[("GitHub", QueryStatus.CLAIMED)]
        + [(f"Slow{i}", QueryStatus.UNKNOWN) for i in range(12)]
    )
    app = SherlockUI()
    async with app.run_test(size=(80, 24)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        counts = app.query_one("#sites-counts", Static)
        drawn = " ".join(
            strip.text for strip in counts.render_lines(counts.region.reset_offset)
        )
        assert "12 unresolved" in drawn, drawn


async def test_eighty_columns_turns_the_list_into_a_picker():
    """The 37-cell list took almost half an 80-column screen and left the link
    column three characters wide. Narrow, the record gets the width and the
    list is one key away."""
    from textual.widgets import DataTable

    await _seed(
        one=[("GitHub", QueryStatus.CLAIMED)],
        two=[("Reddit", QueryStatus.CLAIMED)],
    )
    app = SherlockUI()
    async with app.run_test(size=(80, 24)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        listing = app.query_one("#username-list", DataTable)
        assert pane.has_class("-narrow")
        assert listing.region.width == 0, "the list is still taking a column"
        assert app.focused is not None, "nothing has focus, so no key is heard"

        await pilot.press("ctrl+l")
        await _settle(app, pilot)
        assert listing.region.width > 0
        assert app.focused is listing

        await pilot.press("down", "enter")
        await _settle(app, pilot)
        assert not pane.has_class("-picking")


async def test_the_section_underline_follows_a_relabelled_tab():
    """EXTRACTIONS gains a count after layout, and the underline stayed sized
    to the old label -- "3 PROFI" underlined while PROFILE was open."""
    from textual.widgets import Tab, Tabs
    from textual.widgets._tabs import Underline

    await _seed_built_profile("underlined")
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_profile_section(app, pilot)
        await _settle(app, pilot)
        tabs = app.query_one("#detail-tabs", Tabs)
        active = tabs.query_one("#tab-profile", Tab)
        start, end = active.virtual_region.shrink(active.styles.gutter).column_span
        underline = tabs.query_one(Underline)
        assert (underline.highlight_start, underline.highlight_end) == (start, end)


async def test_delete_in_the_anchor_editor_removes_an_anchor():
    """It deleted a character instead.

    The dialog opened with focus in a text field, where `del` is the field's
    own delete-forward -- so the one key it advertised for removing an anchor
    edited the box. With anchors listed, it now opens on the list.
    """
    from sherlock_project.profile_synthesis import IdentityAnchor
    from sherlock_project.tui.anchor_screen import AnchorScreen

    app = SherlockUI()
    async with app.run_test() as pilot:
        app.push_screen(
            AnchorScreen([IdentityAnchor(field="full_name", value="Avery Stone")])
        )
        await pilot.pause()
        screen = app.screen
        assert app.focused is screen.query_one("#anchor-list")

        await pilot.press("delete")
        await pilot.pause()
        assert screen._anchors == []


async def test_exporting_twice_keeps_both_files(tmp_path, monkeypatch):
    """The second export silently replaced the first."""
    monkeypatch.chdir(tmp_path)
    await _seed(exported=[("GitHub", QueryStatus.CLAIMED)])
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        pane.action_export()
        pane.action_export()
    written = sorted(path.name for path in tmp_path.glob("exported-*.json"))
    assert len(written) == 2, written


async def test_scan_again_from_the_menu_sets_up_the_scan_without_starting_it(
    monkeypatch,
):
    from textual.widgets import Input

    from sherlock_project.tui import runner as runner_module

    launched: list[str] = []

    async def fake_session(*, username, **_options):
        launched.append(username)

    monkeypatch.setattr(runner_module, "run_scan_session", fake_session)

    async with _list_with(
        again=[("GitHub", QueryStatus.CLAIMED)],
    ) as (app, _table, pilot):
        await _open_actions(app, pilot)
        await _choose(app, pilot, "rescan")
        await _settle(app, pilot)
        assert app.query_one("#target-input", Input).value == "again"
        assert app.focused is app.query_one("#target-input", Input)
        assert launched == [], "a menu item on another tab started a scan"


def test_a_long_database_path_is_cut_from_the_middle():
    """It wrapped onto a hidden second line, so the bar said "db" and nothing."""
    from sherlock_project.tui.app import _middle_truncate

    path = "/home/someone/.local/share/a/very/deep/tree/sherlock/sherlock.db"
    short = _middle_truncate(path, 30)
    assert len(short) == 30
    assert short.startswith("/home/")
    assert short.endswith("sherlock.db")
    assert _middle_truncate("/short.db", 30) == "/short.db"


# -- what terminals actually send ---------------------------------------------
#
# The pilot presses key NAMES, so a binding can pass every test above and still
# never fire in a real terminal: the bytes a terminal sends for alt+1 do not
# decode to "alt+1" at all. These feed the bytes each terminal family sends
# through Textual's own decoder and press whatever comes out, which is the path
# a real keypress takes.


def _decoded(sequence: str) -> str:
    from textual._xterm_parser import XTermParser

    parser = XTermParser()
    keys = [
        event.key
        for event in [*parser.feed(sequence), *parser.feed("")]
        if hasattr(event, "key")
    ]
    assert len(keys) == 1, keys
    return keys[0]


@pytest.mark.parametrize(
    ("terminal", "sequence", "tab"),
    [
        # ESC-prefixed alt: GNOME Terminal, Konsole, xterm (metaSendsEscape),
        # iTerm2 and Terminal.app with Option as Meta.
        ("esc-prefixed alt+2", "\x1b2", "tab-results"),
        ("esc-prefixed alt+3", "\x1b3", "tab-settings"),
        # macOS with Option left as a compose key: Option+2 types ™.
        ("macOS Option+2, no meta", "™", "tab-results"),
        # kitty / WezTerm / foot with the CSI-u keyboard protocol.
        ("CSI-u alt+2", "\x1b[50;3u", "tab-results"),
    ],
)
async def test_tab_keys_work_as_terminals_send_them(terminal, sequence, tab):
    from textual.widgets import Input, TabbedContent

    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        await _settle(app, pilot, lambda: app.tabs_ready)
        # The username field has focus on launch, which is the hard case: the
        # character must switch tabs rather than be typed into the field.
        assert isinstance(app.focused, Input)
        await pilot.press(_decoded(sequence))
        await _settle(app, pilot)
        assert app.query_one(TabbedContent).active == tab, terminal
        assert app.query_one("#target-input", Input).value == "", terminal


@pytest.mark.parametrize(
    ("terminal", "sequence"),
    [
        ("xterm-style alt+right (Linux, iTerm2 meta)", "\x1b[1;3C"),
        ("macOS Terminal.app Option+right (ESC f)", "\x1bf"),
    ],
)
async def test_section_keys_work_as_terminals_send_them(terminal, sequence):
    await _seed(sections=[("GitHub", QueryStatus.CLAIMED)])
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await _open_results(app, pilot)
        pane = app.query_one(ResultsPane)
        await _settle(app, pilot, lambda: pane._record is not None)
        await pilot.press(_decoded(sequence))
        await _settle(app, pilot)
        assert app.query_one("#detail-switch").current == "sec-extractions", terminal


async def test_show_all_works_as_terminals_send_it():
    """ESC f decodes as ctrl+right, which the username field owns -- so the
    shown key is alt+s, which arrives intact as ESC s."""
    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one(ScanPane)
        await pilot.press(_decoded("\x1bs"))
        await pilot.pause()
        assert pane._visible == set(STATUS_ORDER)


async def test_late_callbacks_after_the_app_closes_do_nothing():
    """A tab activation or a database read can finish after the app has gone.

    Both happened on CI: a queued TabActivated reached `_tab_activated` once the
    panes were unmounted (macOS), and the results loader's worker returned into
    `_fill_list` after its table was removed (Windows, in a test that never
    opened RESULTS). Each raised NoMatches inside the app on its way out. Called
    here deliberately after teardown, every one of those paths must be a no-op.
    """
    app = SherlockUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        pane = app.query_one(ResultsPane)
        results_tab = app.query_one("#tab-results")

    class Activated:
        pane = results_tab

    app._tab_activated(Activated())
    pane._fill_list()
    pane._fit_keys()
    pane._focus_section()
    pane.focus_default()
    pane._rehighlight_tabs()


async def test_no_button_loses_its_label_when_focused():
    """Clicking `verbose` or `found only` left an empty bar until focus moved.

    The app-wide focus rule draws a tall border above and below a raised
    button. It outranked the chip styling, so a ONE-ROW chip got both borders
    on focus -- and two borders on one row leave no row for the label. Every
    button on every screen state is focused here and must keep a row to draw
    its label in.
    """
    from textual.widgets import Button

    await _seed_built_profile("focused")
    await _results_with("pointer")
    app = SherlockUI()
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        await pilot.click("#toggle-ai")  # reveals the anchors "+" button
        await pilot.pause()

        async def check_visible_buttons(where: str) -> int:
            checked = 0
            for button in app.screen.query(Button):
                if not button.region.area or not button.focusable:
                    continue
                button.focus()
                await pilot.pause()
                assert button.content_region.height >= 1, (
                    f"{where}: #{button.id} has no row for its label when focused"
                )
                checked += 1
            return checked

        assert await check_visible_buttons("SCAN") >= 4

        await _open_profile_section(app, pilot)
        pane = app.query_one(ResultsPane)
        for username in ("focused", "pointer"):
            pane.select_username(username)
            await _settle(app, pilot, lambda u=username: pane._selected == u)
            await _settle(app, pilot)
            await check_visible_buttons(f"PROFILE {username}")

        await pilot.press("alt+left", "alt+left")
        await _settle(app, pilot)
        assert await check_visible_buttons("SITES") >= 1


async def test_a_long_username_wraps_the_title_not_the_buttons():
    """The name used to be in the Delete button, so the button grew with it.

    It is the title's job now: the title wraps inside the dialog and the two
    buttons stay one matched, fixed size whatever is being deleted.
    """
    from textual.widgets import Button

    from sherlock_project.tui.confirm_screen import ConfirmScreen

    name = "an_unreasonably_long_username_" * 4
    app = SherlockUI()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ConfirmScreen(
                f"Delete {name}?", "detail", confirm_label="Delete",
                cancel_label="Keep it", danger=True,
            )
        )
        await pilot.pause()
        screen = app.screen
        title = screen.query_one(".dialog-title")
        dialog = screen.query_one("#dialog")
        assert title.size.height > 1, "the title did not wrap"
        assert title.region.right <= dialog.region.right
        yes = screen.query_one("#confirm-yes", Button)
        no = screen.query_one("#confirm-no", Button)
        assert yes.size.width == no.size.width
        assert yes.outer_size.width <= 16


async def _store_profile(username: str, payload: dict) -> None:
    import json

    from sherlock_project.database import SherlockDB, default_database_path

    db = await SherlockDB.create(str(default_database_path()))
    try:
        await db.update_username_profile_summary(
            username=username,
            profile_summary=json.dumps({"username": username, "input_hash": "h", **payload}),
            input_hash="h",
        )
    finally:
        await db.close()


async def test_extractions_from_an_older_contract_are_not_offered_as_evidence():
    """The reported bug: "Rebuild" over evidence the build then ignored.

    The card counted every stored extraction; synthesis uses only those under
    the CURRENT pass-one contract. So a username analysed by an older version
    was offered "evidence from 28 sites is ready", the build used none of it,
    and the result was an empty profile under a list of every unread page id.
    The offer now counts what a build would use, so this is the analyse state.
    """
    from textual.widgets import Button, Static

    from sherlock_project.database import SherlockDB, default_database_path

    await _results_with("olderrun", unanalysed=4)
    db = await SherlockDB.create(str(default_database_path()))
    try:
        rows = await db.get_site_extractions("olderrun")
        for row in rows[:2]:
            await db.update_result_ai_extraction(
                row.site_id, '{"full_name": ["Ryan"]}',
                contract_hash="an-older-contract", model_key="vendor/m",
            )
    finally:
        await db.close()
    await _store_profile(
        "olderrun",
        {
            "mode": "aggregate",
            "resolution_status": "no_evidence",
            "completeness": "partial",
            "warnings": [
                "Pass-one extraction is still pending for site ids: "
                + ", ".join(str(n) for n in range(5000, 5300))
            ],
        },
    )

    app = SherlockUI()
    async with app.run_test(size=(120, 36)) as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        assert "No profile yet" in hint
        assert "older version of the analysis" in hint
        # The stale sites are not offered as evidence...
        assert "Site00" not in hint and "EVIDENCE  none yet" in hint
        # ...and the build reads every page again before merging.
        assert "+ 4 stored pages to read first" in hint
        assert str(app.query_one("#profile-build", Button).label) == "Build profile"
        # The empty profile draws nothing -- above all, not its id list.
        body = str(app.query_one("#detail-profile", Static).render())
        assert "5000" not in body and "No profile facts" not in body


async def test_a_built_profile_is_described_in_words_not_field_values():
    """`no_evidence · aggregate · partial` named fields, not facts about it."""
    from textual.widgets import Button, Static

    await _results_with("worded", extractions=2, unanalysed=3)
    await _store_profile(
        "worded",
        {
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
        },
    )

    app = SherlockUI()
    async with app.run_test(size=(120, 36)) as pilot:
        await _open_profile_section(app, pilot)

        hint = app.query_one("#profile-anchor-line", Static).render().plain
        for jargon in ("aggregate", "aggregated", "partial", "no_evidence"):
            assert jargon not in hint, jargon
        assert "Merged from every analysed site · no anchors" in hint
        assert "3 stored pages not analysed yet — Rebuild reads them first." in hint
        assert str(app.query_one("#profile-build", Button).label) == "Rebuild profile"


async def test_anchors_edited_after_a_build_say_they_are_not_applied_yet():
    """The anchors are listed in the profile itself; the line above says only
    what the profile cannot: that the edits are not in it yet."""
    from textual.widgets import Static

    from sherlock_project.profile_synthesis import IdentityAnchor

    await _results_with("edited", extractions=2)
    await _store_profile(
        "edited",
        {
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
        },
    )

    app = SherlockUI()
    async with app.run_test(size=(120, 36)) as pilot:
        await _open_profile_section(app, pilot)
        pane = app.query_one(ResultsPane)
        hint = app.query_one("#profile-anchor-line", Static)
        assert "Anchors changed" not in hint.render().plain

        pane._build_anchors = [IdentityAnchor(field="role", value="hacker")]
        pane._redraw_profile_actions(pane._record)
        assert "Anchors changed — rebuild to apply them." in hint.render().plain


async def test_the_anchor_field_is_free_text_and_obvious_values_fill_it():
    """No picker: every consumer canonicalises the field, so any name works.

    An email or a URL says what it is; the field box is not demanded for one.
    And a duplicate is caught the way synthesis would see it, not by spelling.
    """
    from textual.widgets import Input, Static

    from sherlock_project.tui.anchor_screen import AnchorScreen

    app = SherlockUI()
    async with app.run_test() as pilot:
        app.push_screen(AnchorScreen([]))
        await pilot.pause()
        screen = app.screen

        screen.query_one("#anchor-value", Input).value = "ryan@example.com"
        screen.action_add()
        screen.query_one("#anchor-field", Input).value = "Full  Name"
        screen.query_one("#anchor-value", Input).value = "Ryan Hale"
        screen.action_add()
        screen.query_one("#anchor-field", Input).value = "full_name"
        screen.query_one("#anchor-value", Input).value = "ryan hale"
        screen.action_add()
        await pilot.pause()

        assert [(a.field, a.value) for a in screen._anchors] == [
            ("email", "ryan@example.com"),
            ("full name", "Ryan Hale"),
        ]
        assert "already listed" in str(
            screen.query_one("#anchor-status", Static).render()
        )
        # A bare word could be anything, so it is not guessed.
        screen.query_one("#anchor-field", Input).value = ""
        screen.query_one("#anchor-value", Input).value = "hacker"
        screen.action_add()
        await pilot.pause()
        assert len(screen._anchors) == 2
        assert "field box" in str(screen.query_one("#anchor-status", Static).render())


async def test_adding_an_anchor_selects_it_instead_of_announcing_it():
    """"Added role=hacker" was a sentence about something the list shows."""
    from textual.widgets import DataTable, Input, Static

    from sherlock_project.tui.anchor_screen import AnchorScreen

    app = SherlockUI()
    async with app.run_test() as pilot:
        app.push_screen(AnchorScreen([]))
        await pilot.pause()
        screen = app.screen
        for field, value in (("full name", "Ryan"), ("role", "hacker")):
            screen.query_one("#anchor-field", Input).value = field
            screen.query_one("#anchor-value", Input).value = value
            screen.action_add()
        await pilot.pause()

        status = screen.query_one("#anchor-status", Static)
        assert status.display is False
        assert screen.query_one("#anchor-list", DataTable).cursor_row == 1


async def test_a_running_build_can_be_stopped(monkeypatch):
    """There was no way out of a build once started -- a cold model load has
    been measured at 187s. Stop sits where Build was, and Esc does the same."""
    import asyncio

    from textual.widgets import Button, Static

    from sherlock_project import sherlock as sherlock_module

    await _results_with("stoppable", extractions=2)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def endless(**kwargs):
        started.set()
        kwargs["reporter"].ai_model_starting()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(sherlock_module, "run_synthesis_only", endless)

    app = SherlockUI()
    async with app.run_test(size=(120, 36)) as pilot:
        await _open_profile_section(app, pilot)
        await pilot.click("#profile-build")
        # Not `_settle`: it waits for workers to finish, and this one never
        # does until it is stopped.
        for _ in range(50):
            await pilot.pause(0.02)
            if started.is_set():
                break
        assert started.is_set()

        stop = app.query_one("#profile-stop", Button)
        assert stop.display is True
        assert app.query_one("#profile-build", Button).display is False

        await pilot.press("escape")
        for _ in range(50):
            await pilot.pause(0.02)
            if cancelled.is_set() and not app.query_one(ResultsPane)._building:
                break
        assert cancelled.is_set()

        assert stop.display is False
        assert app.query_one("#profile-build", Button).display is True
        assert "Stopped" in str(app.query_one("#profile-status", Static).render())


async def test_the_build_status_leaves_no_blank_band():
    """Each phase line used to end in a newline, and the status carried a
    padding row below it, so a blank band sat between the progress and the
    rule. The lines are joined; the Stop row brings its own gap."""
    from textual.widgets import Static

    from sherlock_project.tui.reporter import TuiReporter

    await _results_with("tidy", extractions=2)
    app = SherlockUI()
    async with app.run_test(size=(120, 36)) as pilot:
        await _open_profile_section(app, pilot)
        pane = app.query_one(ResultsPane)
        reporter = TuiReporter()
        reporter.ai_model_starting()
        pane._build_reporter = reporter
        pane._set_building(True)
        pane._tick_build_status()
        await pilot.pause()

        status = app.query_one("#profile-status", Static)
        text = str(status.render())
        assert not text.endswith("\n")
        # One blank row above (separating it from the card), none below.
        assert status.styles.padding.bottom == 0
        pane._build_reporter = None
        pane._set_building(False)
