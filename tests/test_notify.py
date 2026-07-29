from dataclasses import replace
from io import StringIO
import runpy
import sys

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
    assert "Scan complete for 'first': 1 found across 2 sites" in rendered
    assert "Scan complete for 'second': 1 found across 1 site" in rendered


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
    assert (
        "Sample Person [Instagram — https://instagram.com/sample-user]"
        in rendered
    )
    assert "conference_talks" in rendered
    assert (
        "BlueHat 2026 [Instagram — https://instagram.com/sample-user]"
        in rendered
    )
    assert "employer" in rendered
    assert (
        "Example Labs [Mastodon — https://mastodon.social/@sample-user]"
        in rendered
    )
    assert "strong matches" in rendered and "Instagram" in rendered
    assert "unsure matches" in rendered and "Mastodon" in rendered
    assert "included" not in rendered
    assert "excluded" in rendered and "Unrelated" in rendered
    assert "failed" in rendered and "Failed" in rendered
    assert "[!] Profile may be incomplete" in rendered
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
    assert "\x1b[32mAvery Stone\x1b[0m" in rendered
    assert "\x1b[33mSecurity researcher\x1b[0m" in rendered
    assert "\x1b[32mHacker\x1b[0m" in rendered
    assert "[DirectMatch]" in rendered
    assert "[RoleMatch]" in rendered


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
    assert "[AggregateSource]" in rendered
    assert "\x1b[32mAvery Stone" not in rendered
    assert "\x1b[33mAvery Stone" not in rendered


def test_query_notify_print_remains_the_terminal_reporter_compatibility_name() -> None:
    assert issubclass(QueryNotifyPrint, TerminalReporter)
