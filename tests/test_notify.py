import runpy
import sys
from dataclasses import replace
from io import StringIO

import pytest
from rich.console import Console

from sherlock_project import sherlock as sherlock_module
from sherlock_project.ai_engine import AIRequestTrace, StructuredResponseError
from sherlock_project.ai_provider import AIGenerationStats
from sherlock_project.notify import (
    INTERRUPTION_MESSAGE,
    QueryNotifyPrint,
    TerminalReporter,
)
from sherlock_project.profile_synthesis import ProfileSynthesis
from sherlock_project.result import QueryResult, QueryStatus
from sherlock_project.settings import TRANSPORT_DOC_URL


def _reporter(
    *,
    verbose: bool = False,
    no_color: bool = True,
    force_terminal: bool = False,
    print_all: bool = False,
) -> tuple[TerminalReporter, StringIO, StringIO]:
    output = StringIO()
    errors = StringIO()
    color_system = "standard" if force_terminal and not no_color else None
    console_options = {
        "force_terminal": force_terminal,
        "color_system": color_system,
        "no_color": no_color,
        "width": 180,
    }
    reporter = TerminalReporter(
        verbose=verbose,
        no_color=no_color,
        print_all=print_all,
        console=Console(file=output, **console_options),
        error_console=Console(file=errors, **console_options),
    )
    return reporter, output, errors


def _result(
    site_name: str,
    status: QueryStatus,
    *,
    query_time: float | None = None,
) -> QueryResult:
    return QueryResult(
        username="sample-user",
        site_name=site_name,
        site_url_user=f"https://example.test/{site_name}",
        status=status,
        query_time=query_time,
        context="request failed" if status is QueryStatus.UNKNOWN else None,
    )


def _structured_error() -> StructuredResponseError:
    return StructuredResponseError(
        "OSINT extraction",
        stop_reason="maxPredictedTokensReached",
        predicted_tokens=1024,
        max_tokens=1024,
        parsed_type="str",
        final_content_chars=4701,
        validation_error="json_invalid",
    )


def test_restored_hits_say_when_no_browser_found_them() -> None:
    """`show` already qualifies these rows; the scan has to as well.

    A resumed run reprints stored hits before scanning. Left unqualified, the
    reader acting on the output right now is the one not told that a browser
    never saw the page.
    """
    reporter, output, _ = _reporter()

    reporter.restored_results(
        username="blue",
        results={
            "FastHit": {
                "status": _result("FastHit", QueryStatus.CLAIMED),
                "transport": "http",
            },
            "BrowserHit": {
                "status": _result("BrowserHit", QueryStatus.CLAIMED),
                "transport": "browser",
            },
        },
        to_scan=3,
    )

    lines = output.getvalue().splitlines()
    fast = next(line for line in lines if "FastHit" in line)
    browser = next(line for line in lines if "BrowserHit" in line)
    assert "stored, no browser" in fast
    assert "stored" in browser
    assert "no browser" not in browser


def test_browserless_transport_states_the_limit_before_the_scan() -> None:
    """The warning has to carry the WHY, not just the mode name.

    "Fast transport" alone tells a reader nothing about how to read the results
    it is about to produce, which is the entire reason this prints.
    """
    reporter, output, _ = _reporter()

    reporter.browserless_transport()

    rendered = output.getvalue()
    assert "without a browser" in rendered
    assert "assembled in the browser" in rendered
    assert 'read as "not found"' in rendered
    # Printed, not only linked: OSC 8 does not survive a redirect to a file.
    assert TRANSPORT_DOC_URL in rendered


def test_scan_output_uses_instance_counters_and_keeps_claimed_sites() -> None:
    reporter, output, _ = _reporter(print_all=True)

    reporter.start("first", total=2)
    reporter.update(_result("ClaimedSite", QueryStatus.CLAIMED))
    reporter.update(_result("AvailableSite", QueryStatus.AVAILABLE))
    reporter.finish_scan(elapsed_time=1.25)

    reporter.start("second", total=1)
    reporter.update(_result("AnotherSite", QueryStatus.CLAIMED))
    reporter.finish_scan(elapsed_time=0.5)

    rendered = output.getvalue()
    assert "[+] ClaimedSite: https://example.test/ClaimedSite" in rendered
    assert "[x] AvailableSite: not found" in rendered
    assert "Scan complete for 'first': 1 found, 1 not found" in rendered
    assert "Scan complete for 'second': 1 found, 0 not found" in rendered
    # Nothing was unresolved in either scan, so the caveat line stays away.
    assert "gave no answer" not in rendered


def test_scan_summary_reports_unresolved_sites_separately_from_absent() -> None:
    """The scan must never let "could not determine" reach the user as "absent".

    Reporting only found-vs-total is what allowed a 30% inconclusive rate to be
    read as confirmed absence, so the counts and the caveat line are the
    contract here, not cosmetics.
    """
    reporter, output, _ = _reporter()

    reporter.start("target", total=5)
    reporter.update(_result("Hit", QueryStatus.CLAIMED))
    reporter.update(_result("Gone", QueryStatus.AVAILABLE))
    reporter.update(_result("Flaky", QueryStatus.UNKNOWN))
    reporter.update(_result("Guarded", QueryStatus.WAF))
    reporter.update(_result("Rejected", QueryStatus.ILLEGAL))
    reporter.finish_scan(elapsed_time=3.0)

    rendered = output.getvalue()
    # Absent is counted on its own and never inflated by the unresolved three.
    assert "Scan complete for 'target': 1 found, 1 not found" in rendered
    assert "3 sites of 5 gave no answer" in rendered
    # Each reason is named, because they need different responses from the user.
    assert "1 inconclusive" in rendered
    assert "1 blocked by bot protection" in rendered
    assert "1 rejected the username format" in rendered
    assert 'Not the same as "not found"' in rendered
    assert "sherlock show target --unresolved" in rendered


def test_scan_summary_counters_reset_between_usernames() -> None:
    """A second username must not inherit the first one's unresolved count."""
    reporter, output, _ = _reporter()

    reporter.start("first", total=1)
    reporter.update(_result("Flaky", QueryStatus.UNKNOWN))
    reporter.finish_scan(elapsed_time=1.0)

    reporter.start("second", total=1)
    reporter.update(_result("Hit", QueryStatus.CLAIMED))
    reporter.finish_scan(elapsed_time=1.0)

    rendered = output.getvalue()
    assert "1 site of 1 gave no answer" in rendered
    assert "sherlock show first --unresolved" in rendered
    # The clean second scan says nothing about the first scan's failure.
    assert "sherlock show second --unresolved" not in rendered
    assert rendered.count("gave no answer") == 1


def test_unresolved_summary_stays_ascii_for_redirected_windows_stdout() -> None:
    """Redirected stdout on Windows encodes as cp1252; the summary must survive.

    Same trap `show --json` already documents. A bar chart or box-drawing glyph
    here would turn `sherlock user > out.txt` into a UnicodeEncodeError.
    """
    reporter, output, _ = _reporter()

    reporter.start("target", total=2)
    reporter.update(_result("Flaky", QueryStatus.UNKNOWN))
    reporter.update(_result("Guarded", QueryStatus.WAF))
    reporter.finish_scan(elapsed_time=1.0)

    summary = "".join(
        line for line in output.getvalue().splitlines(keepends=True)
        if "gave no answer" in line or "Not the same as" in line
    )
    assert summary
    summary.encode("cp1252")


def test_no_color_output_has_no_terminal_escape_sequences() -> None:
    reporter, output, _ = _reporter(no_color=True, force_terminal=True)

    reporter.start("plain", total=1)
    reporter.update(_result("Example", QueryStatus.CLAIMED))
    reporter.finish_scan()
    reporter.finish(elapsed_time=2.0)

    rendered = output.getvalue()
    assert "\x1b[" not in rendered
    assert "Processing complete (2.00s total)" in rendered


def test_interactive_output_uses_live_progress_rendering() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)

    reporter.start("live", total=1)
    reporter.update(_result("Example", QueryStatus.CLAIMED))
    reporter.finish_scan()
    reporter.ai_pass_started()
    reporter.ai_scheduled()
    reporter.ai_job_started("Example")
    reporter.ai_job_finished("with_facts")
    reporter.ai_draining()
    reporter.ai_pass_finished()
    reporter.close()

    rendered = output.getvalue()
    assert "\x1b[" in rendered
    assert "Example: https://example.test/Example" in rendered
    assert "Profile extraction 1/1 · 1 with facts · 0 no facts" in rendered


def test_interactive_stage_rows_update_in_place_then_leave_one_final_line() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)

    reporter.ai_model_starting()
    model_task_id = reporter._ai_model_task_id
    assert model_task_id is not None
    assert reporter._progress._tasks[model_task_id].fields["status"] == "preparing"

    reporter.browser_status("installing")
    scanner_task_id = reporter._web_scanner_task_id
    assert scanner_task_id is not None
    reporter.browser_status("starting")
    assert reporter._web_scanner_task_id == scanner_task_id
    assert reporter._progress._tasks[scanner_task_id].fields["status"] == "preparing"

    reporter.ai_model_ready()
    reporter.browser_status("ready")
    reporter.close()

    rendered = output.getvalue()
    assert reporter._ai_model_task_id is None
    assert reporter._web_scanner_task_id is None
    assert rendered.count("Local AI model ready") == 1
    assert rendered.count("Web scanner ready") == 1
    assert "Starting stealth browser" not in rendered


def test_interactive_scan_uses_live_row_without_duplicate_start_line() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)

    reporter.start("sample-user", total=2)
    task_id = reporter._scan_task_id
    assert task_id is not None
    assert "Checking username" not in output.getvalue()

    reporter.update(_result("One", QueryStatus.AVAILABLE))
    task = reporter._progress._tasks[task_id]
    assert task.completed == 1
    assert task.fields["status"] == "1/2 · 0 found"

    reporter.finish_scan(elapsed_time=1.25)
    reporter.close()

    rendered = output.getvalue()
    assert reporter._scan_task_id is None
    assert rendered.count("Scan complete for 'sample-user'") == 1


def test_profile_extraction_is_lazy_and_indeterminate_until_draining() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)

    reporter.ai_scheduled()
    reporter.ai_scheduled()
    assert reporter._ai_task_id is None

    reporter.ai_pass_started()
    task_id = reporter._ai_task_id
    assert task_id is not None
    task = reporter._progress._tasks[task_id]
    assert task.total is None
    assert task.fields["status"] == "0/2"

    reporter.ai_job_started("Instagram")
    assert (
        reporter._progress._tasks[task_id].fields["status"]
        == "0/2 · Instagram"
    )
    reporter.ai_job_finished("with_facts")
    assert reporter._progress._tasks[task_id].total is None

    reporter.ai_draining()
    task = reporter._progress._tasks[task_id]
    assert task.total == 2
    assert task.completed == 1
    assert task.fields["status"] == "1/2 · finishing queue"

    reporter.ai_job_started("Threads")
    reporter.ai_job_finished("no_facts")
    reporter.ai_pass_finished()
    reporter.close()

    assert (
        "Profile extraction 2/2 · 1 with facts · 1 no facts"
        in output.getvalue()
    )


def test_profile_extraction_fraction_grows_after_temporarily_reaching_one() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)

    reporter.ai_scheduled()
    reporter.ai_job_started("Instagram")
    task_id = reporter._ai_task_id
    assert task_id is not None
    reporter.ai_job_finished("with_facts")
    task = reporter._progress._tasks[task_id]
    assert task.total is None
    assert task.fields["status"] == "1/1"

    reporter.ai_scheduled()
    reporter.ai_scheduled()
    assert reporter._progress._tasks[task_id].fields["status"] == "1/3"

    reporter.ai_job_started("Threads")
    reporter.ai_job_finished("no_facts")
    assert reporter._progress._tasks[task_id].fields["status"] == "2/3"

    reporter.ai_draining()
    task = reporter._progress._tasks[task_id]
    assert task.total == 3
    assert task.completed == 2

    reporter.ai_job_started("Steam")
    reporter.ai_job_finished("with_facts")
    reporter.ai_pass_finished()
    reporter.close()

    assert (
        "Profile extraction 3/3 · 2 with facts · 1 no facts"
        in output.getvalue()
    )


def test_profile_extraction_starts_determinate_after_scanning_already_closed() -> None:
    reporter, _, _ = _reporter(no_color=False, force_terminal=True)

    reporter.ai_scheduled()
    reporter.ai_scheduled()
    reporter.ai_draining()
    assert reporter._ai_task_id is None

    reporter.ai_job_started("Instagram")
    task_id = reporter._ai_task_id
    assert task_id is not None
    task = reporter._progress._tasks[task_id]
    assert task.total == 2
    assert task.completed == 0
    assert task.fields["status"] == "0/2 · Instagram"

    reporter.processing_interrupted()


def test_scanning_closes_while_model_loads_without_claiming_extraction_started() -> None:
    reporter, output, _ = _reporter()

    reporter.ai_model_starting()
    reporter.ai_scheduled()
    reporter.ai_draining()

    rendered = output.getvalue()
    assert "finishing profile extraction" not in rendered
    assert "Profile extraction started" not in rendered

    reporter.ai_model_ready()
    reporter.ai_job_started("Instagram")

    rendered = output.getvalue()
    assert "Profile extraction started (1 queued)" in rendered
    assert "[>] Profile extraction 0/1 · Instagram" in rendered


def test_model_failure_after_scanning_closes_never_claims_extraction_started() -> None:
    reporter, output, _ = _reporter()

    reporter.ai_model_starting()
    reporter.ai_scheduled()
    reporter.ai_draining()
    reporter.ai_model_failed(RuntimeError("private model error"))
    reporter.ai_job_deferred()
    reporter.ai_pass_finished()

    rendered = output.getvalue()
    assert "finishing profile extraction" not in rendered
    assert "Profile extraction started" not in rendered
    assert "Profile extraction unavailable · 1 pending" in rendered


def test_plain_profile_extraction_has_exact_summary_without_checkpoints() -> None:
    reporter, output, _ = _reporter()
    reporter.ai_pass_started()

    for site_name, outcome in (
        ("Instagram", "with_facts"),
        ("Empty", "no_facts"),
    ):
        reporter.ai_scheduled()
        reporter.ai_job_started(site_name)
        reporter.ai_job_finished(outcome)  # type: ignore[arg-type]
    reporter.ai_draining()
    reporter.ai_pass_finished()

    rendered = output.getvalue()
    assert "Profile extraction 1/2 (" not in rendered
    assert "[>] Profile extraction 0/1 · Instagram" in rendered
    assert "[>] Profile extraction 1/2 · Empty" in rendered
    assert (
        "[+] Profile extraction 2/2 · 1 with facts · 1 no facts"
        in rendered
    )
    assert "requests" not in rendered
    assert "tokens" not in rendered


def test_profile_extraction_zero_jobs_uses_exact_info_summary() -> None:
    reporter, output, _ = _reporter()
    reporter.ai_pass_started()
    reporter.ai_draining()
    reporter.ai_pass_finished()

    assert (
        "[*] Profile extraction · no sites required processing"
        in output.getvalue()
    )


def test_processing_interrupted_clears_progress_without_success_messages() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)
    reporter.ai_model_starting()
    reporter.browser_status("starting")
    reporter.start("sample-user", total=2)
    reporter.ai_pass_started()
    reporter.synthesis_started("sample-user")

    assert reporter._scan_task_id is not None
    assert reporter._ai_model_task_id is not None
    assert reporter._web_scanner_task_id is not None
    assert reporter._ai_task_id is not None
    assert reporter._synthesis_task_id is not None

    reporter.processing_interrupted()
    reporter.processing_interrupted()
    reporter.ai_model_ready()
    reporter.browser_status("ready")
    reporter.ai_pass_finished()
    reporter.close()

    rendered = output.getvalue()
    assert rendered.count("Processing interrupted") == 1
    assert "committed results were saved" in rendered
    assert "pending AI work can resume on the next --ai run" in rendered
    assert "Scan complete" not in rendered
    assert "AI pass one interrupted" not in rendered
    assert "Processing complete" not in rendered
    assert "Local AI model ready" not in rendered
    assert "Web scanner ready" not in rendered
    assert "[+] Profile extraction" not in rendered
    assert "[!] Profile extraction" not in rendered
    assert reporter._scan_task_id is None
    assert reporter._ai_model_task_id is None
    assert reporter._web_scanner_task_id is None
    assert reporter._ai_task_id is None
    assert reporter._synthesis_task_id is None


def test_cli_maps_main_interruption_result_to_exit_130(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(coroutine) -> int:
        coroutine.close()
        return 130

    monkeypatch.setattr(sherlock_module.asyncio, "run", fake_run)

    with pytest.raises(SystemExit) as error:
        sherlock_module.cli()

    assert error.value.code == 130
    assert capsys.readouterr().out == ""


def test_cli_maps_keyboard_interrupt_to_exit_130_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupted_run(coroutine) -> int:
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(sherlock_module.asyncio, "run", interrupted_run)

    with pytest.raises(SystemExit) as error:
        sherlock_module.cli()

    assert error.value.code == 130
    assert error.value.__suppress_context__ is True
    assert capsys.readouterr().out.strip() == INTERRUPTION_MESSAGE


def test_module_entrypoint_routes_through_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_calls = 0

    def interrupted_cli() -> None:
        nonlocal cli_calls
        cli_calls += 1
        raise SystemExit(130)

    monkeypatch.setattr(sys, "version_info", (3, 13))
    monkeypatch.setattr(sherlock_module, "cli", interrupted_cli)

    with pytest.raises(SystemExit) as error:
        runpy.run_module("sherlock_project", run_name="__main__")

    assert error.value.code == 130
    assert cli_calls == 1


def test_default_ai_errors_are_concise_and_never_expose_content() -> None:
    reporter, output, _ = _reporter()
    reporter.ai_pass_started()
    reporter.ai_scheduled()
    reporter.ai_job_started("Threads")
    reporter.ai_failed(
        "Threads",
        _structured_error(),
    )
    reporter.ai_job_finished("pending")
    reporter.ai_pass_finished()

    rendered = output.getvalue()
    assert "left pending for a future run" in rendered
    assert "stop=" not in rendered
    assert "predicted=" not in rendered
    assert "private reasoning" not in rendered
    assert "LM_STUDIO_INTERNAL" not in rendered


def test_verbose_ai_errors_include_only_safe_diagnostics() -> None:
    reporter, output, _ = _reporter(verbose=True)

    reporter.ai_failed("Threads", _structured_error())

    rendered = output.getvalue()
    assert "stop=maxPredictedTokensReached" in rendered
    assert "output_tokens=1024/1024" in rendered
    assert "final_chars=4701" in rendered
    assert "validation=json_invalid" in rendered
    assert "OSINT extraction" not in rendered


def _trace(*, valid: bool = True) -> AIRequestTrace:
    return AIRequestTrace(
        phase="pass_one",
        username="sample-user",
        site_name="Threads",
        site_id=None,
        attempt=1,
        provider="lmstudio",
        model_key="example/model",
        temperature=0.1,
        context_length=8192,
        max_tokens=1024,
        elapsed_seconds=3.5,
        stats=AIGenerationStats(
            input_tokens=120,
            output_tokens=30,
            reasoning_tokens=0,
            tokens_per_second=10,
            time_to_first_token_seconds=0.25,
        ),
        native_reasoning="",
        final_text=(
            '{"reasoning":"The profile names Jane.","extraction":{"full_name":"Jane"}}'
            if valid
            else "malformed private output"
        ),
        structured_reasoning="The profile names Jane." if valid else "",
        validated_output=(
            {
                "reasoning": "The profile names Jane.",
                "extraction": {"full_name": "Jane"},
            }
            if valid
            else None
        ),
        validation_error=None if valid else "json_invalid",
    )


def test_ai_trace_content_is_verbose_only_and_reasoning_is_labeled_transient() -> None:
    normal, normal_output, _ = _reporter()
    verbose, verbose_output, _ = _reporter(verbose=True)

    normal.ai_trace(_trace())
    verbose.ai_trace(_trace())

    assert "Jane" not in normal_output.getvalue()
    rendered = verbose_output.getvalue()
    assert "Manual reasoning (transient)" in rendered
    assert "The profile names Jane." in rendered
    assert "Validated extraction" in rendered
    assert '"full_name": "Jane"' in rendered
    assert "120" in rendered and "10.00 tok/s" in rendered
    assert rendered.count("may contain OSINT evidence") == 1


def test_verbose_profile_metrics_are_separate_from_exact_outcome_line() -> None:
    reporter, output, _ = _reporter(verbose=True)
    reporter.ai_pass_started()
    reporter.ai_scheduled()
    reporter.ai_job_started("Threads")
    reporter.ai_trace(_trace())
    reporter.ai_job_finished("with_facts")
    reporter.ai_draining()
    reporter.ai_pass_finished()

    rendered = output.getvalue()
    assert "Profile extraction diagnostics: requests 1" in rendered
    assert "tokens 120 in / 30 out" in rendered
    assert "request time 3.50s" in rendered
    assert "wall time " in rendered
    assert (
        "[+] Profile extraction 1/1 · 1 with facts · 0 no facts\n"
        in rendered
    )


def test_verbose_renders_cached_extractions_and_explains_missing_reasoning() -> None:
    normal, normal_output, _ = _reporter()
    verbose, verbose_output, _ = _reporter(verbose=True)
    records = [
        ("Instagram", '{"full_name":"Jane Doe"}'),
        ("Pending", None),
    ]

    normal.ai_cached_evidence("sample-user", records)
    verbose.ai_cached_evidence("sample-user", records)

    assert normal_output.getvalue() == ""
    rendered = verbose_output.getvalue()
    assert "cached 1, pending 1" in rendered
    assert "Cached pass-one extraction: Instagram" in rendered
    assert '"full_name": "Jane Doe"' in rendered
    assert "do not retain transient reasoning" in rendered


def test_verbose_ai_configuration_exposes_effective_nonsecret_settings() -> None:
    reporter, output, _ = _reporter(verbose=True)

    reporter.ai_configuration(
        model="example/model",
        base_url="http://localhost:1234",
        temperature=0.1,
        context_length=8192,
    )

    rendered = output.getvalue()
    assert "example/model" in rendered
    assert "temperature=0.1" in rendered
    assert "native reasoning=pass1 off/pass2 on" in rendered


def test_verbose_trace_shows_malformed_final_and_unexpected_native_reasoning() -> None:
    reporter, output, _ = _reporter(verbose=True)
    trace = _trace(valid=False)
    trace = replace(
        trace,
        native_reasoning="unexpected thought",
        stats=AIGenerationStats(reasoning_tokens=7),
    )

    reporter.ai_trace(trace)

    rendered = output.getvalue()
    assert "Malformed final response" in rendered
    assert "malformed private output" in rendered
    assert "unexpected thought" in rendered
    assert "despite reasoning-off mode" in rendered


def test_verbose_trace_stays_quiet_when_native_reasoning_is_the_design() -> None:
    """A model that cannot stop thinking is not ignoring us.

    Warning per site on a scan that deliberately relies on native reasoning
    buries the warnings that mean something.
    """
    reporter, output, _ = _reporter(verbose=True)
    trace = replace(
        _trace(),
        native_reasoning="planned thought",
        native_reasoning_expected=True,
        structured_reasoning="",
        stats=AIGenerationStats(reasoning_tokens=7),
    )

    reporter.ai_trace(trace)

    rendered = output.getvalue()
    assert "planned thought" in rendered
    assert "Native reasoning (transient)" in rendered
    assert "despite reasoning-off mode" not in rendered
    assert "Unexpected native reasoning" not in rendered


def test_verbose_trace_flags_a_reasoning_field_the_variant_forbade() -> None:
    reporter, output, _ = _reporter(verbose=True)
    trace = replace(_trace(), native_reasoning_expected=True)

    reporter.ai_trace(trace)

    assert "despite the no-reasoning prompt" in output.getvalue()


def test_verbose_trace_shows_expected_pass_two_native_reasoning() -> None:
    reporter, output, _ = _reporter(verbose=True)
    trace = replace(
        _trace(),
        phase="pass_two",
        native_reasoning="Compared only the supplied evidence.",
        structured_reasoning="",
        final_text='{"identity_status":"strong_match"}',
        validated_output={"identity_status": "strong_match"},
        stats=AIGenerationStats(reasoning_tokens=9),
    )

    reporter.ai_trace(trace)

    rendered = output.getvalue()
    assert "Native reasoning (transient)" in rendered
    assert "Compared only the supplied evidence." in rendered
    assert "Validated target decision" in rendered
    assert "despite reasoning-off mode" not in rendered


def test_ai_progress_tracks_all_outcomes() -> None:
    reporter, output, _ = _reporter()

    for site_name, outcome in (
        ("One", "with_facts"),
        ("Two", "no_facts"),
        ("Three", "pending"),
    ):
        reporter.ai_scheduled()
        reporter.ai_job_started(site_name)
        reporter.ai_job_finished(outcome)  # type: ignore[arg-type]
    reporter.ai_pass_finished()

    assert reporter.ai_stats.scheduled == 3
    assert reporter.ai_stats.completed == 3
    assert reporter.ai_stats.with_facts == 1
    assert reporter.ai_stats.no_facts == 1
    assert reporter.ai_stats.pending == 1
    rendered = output.getvalue()
    assert rendered.count("Profile extraction started") == 1
    assert (
        "Profile extraction 3/3 · 1 with facts · 1 no facts · 1 pending"
        in rendered
    )


def test_model_failure_defers_jobs_quietly_and_hides_error_details() -> None:
    reporter, output, errors = _reporter(verbose=True)
    reporter.ai_model_starting()
    reporter.ai_model_failed(RuntimeError("private model path and error"))

    for _ in range(2):
        reporter.ai_scheduled()
        reporter.ai_job_deferred()
    reporter.ai_pass_finished()

    rendered = output.getvalue() + errors.getvalue()
    assert rendered.count("Local AI model unavailable") == 1
    assert "RuntimeError" in rendered
    assert "private model path and error" not in rendered
    assert "Profile extraction 1/2" not in rendered
    assert reporter.ai_stats.completed == 2
    assert reporter.ai_stats.pending == 2
    assert reporter._ai_started is False
    assert "Profile extraction unavailable · 2 pending" in rendered


def test_synthesis_output_formats_exact_profile_keys_sources_and_warnings() -> None:
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "sample-user",
            "input_hash": "hash",
            "mode": "anchored",
            "resolution_status": "resolved",
            "completeness": "complete",
            "strong_profile": {
                "full_name": ["Sample Person"],
                "conference_talks": ["BlueHat 2026"],
            },
            "unsure_profile": {"employer": ["Example Labs"]},
            "provenance": [
                {
                    "field": "full_name",
                    "value": "Sample Person",
                    "source_site_ids": [7],
                    "origins": ["extraction"],
                },
                {
                    "field": "conference_talks",
                    "value": "BlueHat 2026",
                    "source_site_ids": [7],
                    "origins": ["extraction"],
                },
                {
                    "field": "employer",
                    "value": "Example Labs",
                    "source_site_ids": [8],
                    "origins": ["extraction"],
                },
            ],
            "source_decisions": [
                {
                    "site_id": 7,
                    "site_name": "Instagram",
                    "site_url": "https://instagram.com/sample-user",
                    "disposition": "included",
                    "identity_status": "strong_match",
                },
                {
                    "site_id": 8,
                    "site_name": "Mastodon",
                    "site_url": "https://mastodon.social/@sample-user",
                    "disposition": "included",
                    "identity_status": "unsure",
                },
                {
                    "site_id": 9,
                    "site_name": "Unrelated",
                    "disposition": "excluded",
                    "identity_status": "reject",
                },
                {
                    "site_id": 10,
                    "site_name": "Failed",
                    "disposition": "failed",
                },
            ],
            "warnings": ["Profile may be incomplete"],
        }
    )

    reporter.synthesis_started("sample-user")
    reporter.synthesis_finished("sample-user", profile, cache_hit=False)

    rendered = output.getvalue()
    assert "AI profile updated for 'sample-user'" in rendered
    # Sources are summarised by count and name; the URLs live behind --sources
    # and in the JSON export, so a value no longer drags its address inline.
    assert "Sample Person" in rendered
    assert "1 site: Instagram" in rendered
    assert "https://instagram.com/sample-user" not in rendered
    assert "conference_talks" in rendered
    assert "BlueHat 2026" in rendered
    assert "employer" in rendered
    assert "Example Labs" in rendered
    assert "1 site: Mastodon" in rendered
    assert "strong matches" in rendered and "Instagram" in rendered
    assert "unsure matches" in rendered and "Mastodon" in rendered
    assert "included" not in rendered
    assert "excluded" in rendered and "Unrelated" in rendered
    assert "failed" in rendered and "Failed" in rendered
    # Synthesis warnings are diagnostics; summarised unless --verbose.
    assert "1 note about how this was built" in rendered
    assert "Profile may be incomplete" not in rendered
    assert "Match key" not in rendered
    assert "strong identity match" not in rendered
    assert "\x1b[" not in rendered


def test_anchored_profile_colors_values_and_places_match_key_above_table() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "anchored",
            "resolution_status": "resolved",
            "completeness": "complete",
            "strong_profile": {
                "full_name": ["Avery Stone"],
                "roles": ["Hacker"],
            },
            "unsure_profile": {"roles": ["Security researcher"]},
            "provenance": [
                {
                    "field": "full_name",
                    "value": "Avery Stone",
                    "source_site_ids": [1],
                    "origins": ["extraction"],
                },
                {
                    "field": "roles",
                    "value": "Hacker",
                    "source_site_ids": [1],
                    "origins": ["extraction"],
                },
                {
                    "field": "roles",
                    "value": "Security researcher",
                    "source_site_ids": [2],
                    "origins": ["extraction"],
                },
            ],
            "source_decisions": [
                {
                    "site_id": 1,
                    "site_name": "DirectMatch",
                    "disposition": "included",
                    "identity_status": "strong_match",
                },
                {
                    "site_id": 2,
                    "site_name": "RoleMatch",
                    "disposition": "included",
                    "identity_status": "unsure",
                },
            ],
            "anchors": [
                {
                    "field": "roles",
                    "value": "Hacker",
                    "trust": "context",
                }
            ],
        }
    )

    reporter.render_profile(profile)

    rendered = output.getvalue()
    assert rendered.index("Match key") < rendered.index("full_name")
    assert "strong identity match" in rendered
    assert "unsure identity match" in rendered
    assert "\x1b[32mAvery Stone" in rendered
    assert "\x1b[33mSecurity researcher" in rendered
    # No closing sequence asserted: values now sit in their own table column,
    # so a short value carries its cell padding inside the styled span.
    assert "\x1b[32mHacker" in rendered
    # Confident and unsure are now separated by heading, not colour alone.
    assert rendered.index("CONFIDENT") < rendered.index("Avery Stone")
    assert rendered.index("UNSURE") < rendered.index("Security researcher")
    assert "1 site: DirectMatch" in rendered
    assert "1 site: RoleMatch" in rendered


def test_anchorless_profile_values_remain_uncolored() -> None:
    reporter, output, _ = _reporter(no_color=False, force_terminal=True)
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "unsure_profile": {},
            "provenance": [
                {
                    "field": "full_name",
                    "value": "Avery Stone",
                    "source_site_ids": [1],
                    "origins": ["extraction"],
                }
            ],
            "source_decisions": [
                {
                    "site_id": 1,
                    "site_name": "AggregateSource",
                    "disposition": "aggregated",
                }
            ],
        }
    )

    reporter.render_profile(profile)

    rendered = output.getvalue()
    assert "Match key" not in rendered
    assert "Avery Stone" in rendered
    assert "1 site: AggregateSource" in rendered
    assert "\x1b[32mAvery Stone" not in rendered
    assert "\x1b[33mAvery Stone" not in rendered


def test_query_notify_print_remains_the_terminal_reporter_compatibility_name() -> None:
    assert issubclass(QueryNotifyPrint, TerminalReporter)


def test_anchored_profile_shows_the_anchors_it_was_built_from() -> None:
    """An anchored profile is unreadable without the anchors.

    The renderer already reported how each site scored ("strong match"), but
    never what it matched against, so a stored profile could not be
    interpreted after the fact.
    """
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "anchored",
            "resolution_status": "resolved",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "anchors": [
                {"field": "name", "value": "Avery Stone"},
                {"field": "roles", "value": "Hacker", "trust": "context"},
                {"field": "city", "value": "Oslo", "source": "case notes"},
            ],
        }
    )

    reporter.render_profile(profile)

    rendered = output.getvalue()
    # A SECTION, headed and shaped exactly like CONFIDENT and MATCHING -- an
    # anchor is a field and a value about this person, same as they hold.
    assert "ANCHORS" in rendered
    assert "anchored to" not in rendered
    assert "name=Avery Stone" not in rendered
    for field, value in (("name", "Avery Stone"), ("roles", "Hacker"), ("city", "Oslo")):
        assert field in rendered
        assert value in rendered

    # Trust is named only when it is NOT the default. It no longer influences
    # anything observable and the UI stopped offering it, so printing "strong"
    # beside every anchor would restate a value nobody chose. A level somebody
    # did choose -- on the command line -- still shows.
    assert "strong" not in rendered
    assert "context" in rendered
    # Real provenance is kept...
    assert "case notes" in rendered


def test_internal_anchor_sources_are_not_printed() -> None:
    """"you typed it here" is not provenance.

    `command_line` and `user_interface` are how the code labels its own entry
    points. Printed beside every anchor as "(from user_interface)" they read as
    debug output and crowd out the part that means something.
    """
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "anchored",
            "resolution_status": "resolved",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "anchors": [
                {"field": "name", "value": "ryan", "trust": "verified",
                 "source": "user_interface"},
                {"field": "city", "value": "Oslo", "source": "case notes"},
            ],
        }
    )
    reporter.render_profile(profile)
    rendered = output.getvalue()

    assert "ANCHORS" in rendered
    assert "user_interface" not in rendered
    assert "command_line" not in rendered
    # The anchor itself, and a source worth having, both survive.
    assert "ryan" in rendered
    assert "case notes" in rendered


def test_anchorless_profile_shows_no_anchor_row() -> None:
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "partial",
            "strong_profile": {"full_name": ["Avery Stone"]},
        }
    )

    reporter.render_profile(profile)

    assert "anchored to" not in output.getvalue()


def test_format_sources_summarises_by_count_and_name() -> None:
    from sherlock_project.notify import _format_sources

    assert _format_sources([]) == ""
    assert _format_sources(["GitLab"]) == "1 site: GitLab"
    assert _format_sources(["A", "B"]) == "2 sites: A, B"
    # Beyond the limit the rest become a count, not a longer line.
    assert _format_sources(["A", "B", "C"]) == "3 sites: A, B +1"
    assert _format_sources(["A", "B", "C", "D"]) == "4 sites: A, B +2"


def test_profile_keeps_value_order_and_hides_nothing() -> None:
    """Presentation only: no reordering, no capping."""
    reporter, output, _ = _reporter()
    names = [f"name-{index:02d}" for index in range(30)]
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"display_name": names},
        }
    )

    reporter.render_profile(profile)

    rendered = output.getvalue()
    for name in names:
        assert name in rendered
    positions = [rendered.index(name) for name in names]
    assert positions == sorted(positions)
    assert "more" not in rendered


def test_show_sources_restores_the_full_urls() -> None:
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "provenance": [
                {
                    "field": "full_name",
                    "value": "Avery Stone",
                    "source_site_ids": [1],
                    "origins": ["extraction"],
                }
            ],
            "source_decisions": [
                {
                    "site_id": 1,
                    "site_name": "Mastodon",
                    "site_url": "https://mastodon.social/@avery",
                    "disposition": "aggregated",
                }
            ],
        }
    )

    reporter.render_profile(profile, show_sources=True)

    rendered = output.getvalue().replace("\n", "")
    assert "https://mastodon.social/@avery" in rendered


def test_profile_notes_are_summarised_unless_verbose() -> None:
    """Synthesis warnings are diagnostics, not user-facing text.

    They name internal site ids, model exception types and token counts, so
    printing them above the values buried the one caveat that matters in noise
    a reader cannot act on.
    """
    payload = {
        "username": "fixture_handle",
        "input_hash": "hash",
        "mode": "anchored",
        "resolution_status": "resolved",
        "completeness": "partial",
        "strong_profile": {"full_name": ["Avery Stone"]},
        "warnings": [
            "Pass-one extraction is still pending for site ids: 264, 793",
            "Pass-two decision failed for site id 466 (StructuredResponseError)",
        ],
    }

    quiet, quiet_output, _ = _reporter()
    quiet.render_profile(ProfileSynthesis.model_validate(payload))
    rendered = quiet_output.getvalue().replace("\n", " ")
    assert "2 notes about how this was built" in rendered
    assert "site ids: 264" not in rendered
    assert "StructuredResponseError" not in rendered

    loud, loud_output, _ = _reporter(verbose=True)
    loud.render_profile(ProfileSynthesis.model_validate(payload))
    detailed = loud_output.getvalue().replace("\n", " ")
    assert "StructuredResponseError" in detailed
    assert "2 notes about how this was built" not in detailed


def test_aggregate_profile_states_the_different_people_caveat_up_front() -> None:
    """The one caveat that changes how the values should be read.

    It must not depend on the synthesis warning list, and it must appear above
    the values rather than a hundred lines below them.
    """
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "aggregate",
            "resolution_status": "aggregated",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
        }
    )

    reporter.render_profile(profile)

    rendered = output.getvalue()
    flat = rendered.replace("\n", " ")
    assert "may describe different people who share this username" in flat
    assert rendered.index("No anchors used") < rendered.index("Avery Stone")


def test_anchored_profile_omits_the_aggregate_caveat() -> None:
    reporter, output, _ = _reporter()
    profile = ProfileSynthesis.model_validate(
        {
            "username": "fixture_handle",
            "input_hash": "hash",
            "mode": "anchored",
            "resolution_status": "resolved",
            "completeness": "complete",
            "strong_profile": {"full_name": ["Avery Stone"]},
            "anchors": [{"field": "name", "value": "Avery Stone"}],
        }
    )

    reporter.render_profile(profile)

    assert "No anchors used" not in output.getvalue()


def test_other_model_extractions_are_reported_once_and_only_on_mismatch():
    """Silent when the models agree; loud, with counts, when they do not.

    A line that prints on every run stops being read, and this one only has
    meaning right after someone changes model.
    """
    reporter, output, _ = _reporter()

    reporter.ai_extractions_from_other_models(
        username="blue",
        configured_model="vendor/large",
        counts={"vendor/large": 12},
    )
    assert output.getvalue() == ""

    reporter.ai_extractions_from_other_models(
        username="blue",
        configured_model="vendor/large",
        counts={},
    )
    assert output.getvalue() == ""

    reporter.ai_extractions_from_other_models(
        username="blue",
        configured_model="vendor/large",
        counts={"vendor/large": 2, "vendor/small": 9, None: 3},
    )
    text = output.getvalue()

    assert "12 stored extractions" in text
    assert "'blue'" in text
    assert "vendor/large" in text
    assert "9 from vendor/small" in text
    assert "3 from an unrecorded model" in text
    # The remedy has to be nameable, or the warning is just bad news.
    assert "sherlock blue --ai --fresh" in text


def test_other_model_extractions_uses_singular_for_one():
    reporter, output, _ = _reporter()

    reporter.ai_extractions_from_other_models(
        username="blue",
        configured_model="vendor/large",
        counts={"vendor/small": 1},
    )

    assert "1 stored extraction for" in output.getvalue()
