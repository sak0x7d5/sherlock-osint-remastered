"""
Sherlock: Find Usernames Across Social Networks Module

This module contains the main logic to search for usernames at social
networks.
"""

import sys

try:
    from sherlock_project.__init__ import import_error_test_var  # noqa: F401
except ImportError:
    print("Did you run Sherlock with `python3 sherlock/sherlock.py ...`?")
    print("This is an outdated method. Please see https://github.com/sak0x7d5/sherlock-osint-remastered for up to date instructions.")
    sys.exit(1)

import asyncio
import csv
import os
import re
from argparse import ArgumentParser, ArgumentTypeError, RawDescriptionHelpFormatter
from collections.abc import Awaitable, Callable, Sequence
from json import dumps as json_dumps
from json import loads as json_loads
from time import perf_counter

import pandas as pd
import requests
from playwright.async_api import APIResponse, Response
from playwright.async_api import Error as PlaywrightError

from sherlock_project.__init__ import (
    __longname__,
    __shortname__,
    __version__,
    forge_api_latest_release,
)
from sherlock_project.ai_config import AIConfigError, AISettings, load_ai_settings
from sherlock_project.ai_engine import (
    AIService,
    PassOneKeyRegistry,
    pass_one_contract_hash,
)
from sherlock_project.ai_setup import run_ai_setup
from sherlock_project.content_extraction import extract_profile_content
from sherlock_project.database import SherlockDB, default_database_path
from sherlock_project.detection import QueryConfidence, evaluate
from sherlock_project.investigation_context import (
    build_investigation_context,
    parse_inline_anchor,
)
from sherlock_project.notify import (
    INTERRUPTION_MESSAGE,
    QueryNotify,
    QueryNotifyPrint,
    TerminalReporter,
)
from sherlock_project.pass_one_runtime import hydrate_pass_one_key_registry
from sherlock_project.playwright_engine import PlaywrightEngine
from sherlock_project.profile_synthesis import IdentityAnchor
from sherlock_project.result import QueryResult, QueryStatus
from sherlock_project.show import run_show
from sherlock_project.sites import SitesInformation
from sherlock_project.synthesis_pipeline import synthesize_username_profile
from sherlock_project.wmn_adapter import normalize_username, preferred_transport

# Markers live in the initial HTML, and waiting for full 'load' costs sites that
# never quiesce -- Telegram times out at 45s on 'load' but resolves immediately
# on 'domcontentloaded'.
PAGE_WAIT_UNTIL = "domcontentloaded"


async def await_response(completed_task: asyncio.Task) -> tuple[APIResponse | Response | None, str, str | None]:
    response = None
    error_context = "General Unknown Error"
    exception_text = None

    try:
        response = await completed_task

        if hasattr(response, "status") and response.status is not None:
            error_context = None  # Success

    except PlaywrightError as err:
        error_msg = str(err).lower()
        exception_text = str(err)

        if "timeout" in error_msg:
            error_context = "Timeout Error"
        elif any(x in error_msg for x in ["econnreset", "connection reset"]):
            error_context = "Connection Reset Error"
        elif any(x in error_msg for x in ["enotfound", "getaddrinfo", "name or service not known"]):
            error_context = "DNS Error"
        elif any(x in error_msg for x in ["socket hang up", "disconnected", "tls", "secure connection"]):
            error_context = "Network / TLS Error"
        elif "parse error" in error_msg:
            error_context = "Parse Error (WAF?)"
        else:
            error_context = "Playwright Error"

    except Exception as err:          # Final safety net
        error_context = "Unknown Error"
        exception_text = f"{type(err).__name__}: {err}"

    return response, error_context, exception_text

def interpolate_string(input_object, username):
    if isinstance(input_object, str):
        return input_object.replace("{}", username)
    elif isinstance(input_object, dict):
        return {k: interpolate_string(v, username) for k, v in input_object.items()}
    elif isinstance(input_object, list):
        return [interpolate_string(i, username) for i in input_object]
    return input_object


async def _probe_site(
    *,
    engine,
    net_info,
    site_username,
    transport,
    timeout,
    wants_profile_body,
    fetch,
    fetch_kwargs,
):
    """Run one site's entire fetch sequence as a single task.

    A site may need a second request: an API result that decided nothing is
    worth re-asking against the profile page, and a confirmed hit found over an
    API needs that page's markup for the AI pass. Both used to be awaited inside
    the consumer loop, which meant one slow page load stalled the reporting of
    every other site that had already finished -- 680 sites would sit at zero
    results while the loop blocked on a single 60-second timeout.

    Owning the sequence here keeps the second request concurrent and under the
    engine's semaphore like every other fetch, so the consumer only ever does
    cheap work and results appear as they land.

    The richer body is attached to the response as ``profile_text`` rather than
    replacing ``text``: detection must keep reading the endpoint its markers
    were written against, while storage and extraction want the profile page.
    """
    response = await fetch(**fetch_kwargs)

    detection_rule = net_info.get("detection")
    if response is None or not detection_rule:
        return response

    # A second look is worth it whenever the profile lives somewhere other than
    # the URL just probed -- that page is a GET regardless of how the check was
    # made, so even a POST-checked site can be re-read in the browser. Keyed on
    # the URLs rather than the transport: a site whose check endpoint is an API
    # still wants the real profile page for extraction, whichever way it was
    # fetched.
    can_reach_profile = bool(
        net_info.get("urlProfile")
        and net_info.get("urlProfile") != net_info.get("url")
    )
    if not can_reach_profile:
        return response

    verdict = evaluate(detection_rule, response.status, response.text)

    if not verdict.is_decided:
        retry = await _retry_on_profile_page(
            engine=engine,
            net_info=net_info,
            site_username=site_username,
            timeout=timeout,
        )
        if retry is not None:
            retried_response, retried_verdict = retry
            if retried_verdict.is_decided:
                return retried_response
        return response

    if wants_profile_body and verdict.exists is True:
        profile_text = await _fetch_profile_content(
            engine=engine,
            net_info=net_info,
            site_username=site_username,
            timeout=timeout,
        )
        if profile_text:
            response.profile_text = profile_text

    return response


async def _retry_on_profile_page(engine, net_info, site_username, timeout):
    """Re-probe an undecided API result against the human-facing profile page.

    An API endpoint that matched neither side of its rule is often answerable
    from the page a person would open: the profile carries markup the API never
    returns. It is also the content the AI pass wants, so a hit found this way
    arrives with a body attached rather than an empty API envelope.

    Returns ``(response, verdict)``, or None if the retry could not be made.
    Failure is swallowed on purpose -- this is a bonus attempt on a result that
    is already undecided, and it must never cost the caller the scan.
    """
    try:
        profile_url = interpolate_string(
            net_info["urlProfile"], site_username.replace(' ', '%20')
        )
        response = await engine.fetch_with_page(
            url=profile_url,
            headers=net_info.get("headers") or {},
            timeout=timeout * 1000,
            wait_until=PAGE_WAIT_UNTIL,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return None

    if response is None:
        return None

    return response, evaluate(net_info["detection"], response.status, response.text)


async def _fetch_profile_content(engine, net_info, site_username, timeout) -> str | None:
    """Fetch the human-facing profile page for AI extraction.

    Separate from detection on purpose: the check endpoint may be an API whose
    response contains nothing worth extracting. Returns None on any failure --
    the account has already been confirmed, and losing the richer content must
    never downgrade a correct result.
    """
    try:
        profile_url = interpolate_string(
            net_info["urlProfile"], site_username.replace(' ', '%20')
        )
        response = await engine.fetch_with_page(
            url=profile_url,
            headers=net_info.get("headers") or {},
            timeout=timeout * 1000,
            wait_until=PAGE_WAIT_UNTIL,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return None

    return getattr(response, "text", None) if response is not None else None


def check_for_parameter(username):
    """checks if {?} exists in the username
    if exist it means that sherlock is looking for more multiple username"""
    return "{?}" in username


checksymbols = ["_", "-", "."]

AIEnqueue = Callable[[int], Awaitable[None]]
AICancel = Callable[[], None]


def multiple_usernames(username):
    """replace the parameter with with symbols and return a list of usernames"""
    allUsernames = []
    for i in checksymbols:
        allUsernames.append(username.replace("{?}", i))
    return allUsernames


def expand_usernames(usernames: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for username in usernames:
        candidates = (
            multiple_usernames(username)
            if check_for_parameter(username)
            else [username]
        )
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)
    return expanded


def restore_saved_results(
    username: str,
    saved_rows: dict[str, dict],
    site_data_all: dict,
) -> dict:
    """Rebuild scan-shaped results from rows already in the database.

    A scan only returns the sites it actually checked, so without this a repeat
    run reports nothing and writes empty exports while every hit sits in
    SQLite. Restored entries carry the same shape as live ones so the report
    and the exporters cannot tell the difference.

    Rows that cannot be reconstructed are skipped rather than raised: a single
    unparseable status must not cost the user the rest of their stored results.
    """
    restored: dict = {}

    for site_name, row in saved_rows.items():
        try:
            status = QueryStatus(row["status"])
        except ValueError:
            continue

        confidence = None
        stored_confidence = row.get("confidence")
        if stored_confidence:
            try:
                confidence = QueryConfidence(stored_confidence)
            except ValueError:
                confidence = None

        site_url = row.get("site_url") or ""
        net_info = site_data_all.get(site_name, {})

        restored[site_name] = {
            "url_main": net_info.get("urlMain"),
            "url_user": site_url,
            "status": QueryResult(
                username,
                site_name,
                site_url,
                status,
                query_time=row.get("query_time_ms"),
                context=row.get("error_context"),
                confidence=confidence,
            ),
            "http_status": row.get("status_code") or "",
            "response_text": "",
        }

    return restored


async def synthesize_profiles(
    *,
    db: SherlockDB,
    ai_service: AIService,
    usernames: list[str],
    force: bool,
    inline_anchors: Sequence[IdentityAnchor] = (),
    reporter: TerminalReporter | None = None,
) -> None:
    context = build_investigation_context(inline_anchors)
    for username in usernames:
        if reporter is not None:
            reporter.synthesis_started(username)
        try:
            result = await synthesize_username_profile(
                db=db,
                ai_service=ai_service,
                username=username,
                context=context,
                force=force,
            )
        except Exception as error:
            if reporter is not None:
                reporter.synthesis_failed(username, error)
            continue

        if reporter is not None:
            reporter.synthesis_finished(
                username,
                result.profile,
                cache_hit=result.cache_hit,
            )


async def report_cached_ai_evidence(
    *,
    db: SherlockDB,
    usernames: Sequence[str],
    contract_hash: str,
    reporter: TerminalReporter | None,
) -> None:
    if reporter is None or not reporter.verbose:
        return
    for username in usernames:
        records = await db.get_ai_profile_evidence(username)
        reporter.ai_cached_evidence(
            username,
            [
                (record.site_name, record.ai_extraction)
                for record in records
                if record.ai_extraction_contract_hash == contract_hash
            ],
        )


async def run_synthesis_only(
    *,
    usernames: list[str],
    force: bool,
    inline_anchors: Sequence[IdentityAnchor] = (),
    reporter: TerminalReporter | None = None,
    ai_settings: AISettings | None = None,
) -> None:
    database_path = default_database_path()
    if reporter is not None:
        reporter.debug(f"Using database at {database_path}")
    db = await SherlockDB.create(str(database_path))
    ai_service: AIService | None = None
    contract_hash = pass_one_contract_hash()
    interrupted = False
    try:
        await report_cached_ai_evidence(
            db=db,
            usernames=usernames,
            contract_hash=contract_hash,
            reporter=reporter,
        )
        needs_model = build_investigation_context(inline_anchors).has_anchors
        if needs_model and reporter is not None:
            reporter.ai_model_starting()
        trace_callback = reporter.ai_trace if reporter is not None else None
        ai_service = (
            await AIService.create(
                settings=ai_settings,
                trace_callback=trace_callback,
            )
            if needs_model
            else AIService(trace_callback=trace_callback)
        )
        if needs_model and reporter is not None:
            reporter.ai_model_ready()
        await synthesize_profiles(
            db=db,
            ai_service=ai_service,
            usernames=usernames,
            force=force,
            inline_anchors=inline_anchors,
            reporter=reporter,
        )
    except asyncio.CancelledError:
        interrupted = True
        raise
    finally:
        cleanup_error, cleanup_cancellation = await _close_ai_service_and_db(
            ai_service=ai_service,
            db=db,
        )
        if not interrupted:
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            if cleanup_error is not None:
                raise cleanup_error


async def _close_ai_service_and_db(
    *,
    ai_service: AIService | None,
    db: SherlockDB,
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    """Attempt both closers and report the last failure plus any cancellation."""
    cleanup_error: BaseException | None = None
    cleanup_cancellation: asyncio.CancelledError | None = None
    if ai_service is not None:
        try:
            await ai_service.close()
        except BaseException as error:
            cleanup_error = error
            if isinstance(error, asyncio.CancelledError):
                cleanup_cancellation = error

    try:
        await db.close()
    except BaseException as error:
        cleanup_error = error
        if isinstance(error, asyncio.CancelledError):
            cleanup_cancellation = error

    return cleanup_error, cleanup_cancellation


async def ai_worker(
    ai_queue: asyncio.Queue[int],
    sherlock_db: SherlockDB,
    ai_service: AIService,
    reporter: TerminalReporter | None = None,
) -> None:
    contract_hash = ai_service.pass_one_contract_hash
    key_registry = PassOneKeyRegistry()
    hydrated_usernames: set[str] = set()

    while True:
        try:
            site_id = await ai_queue.get()
        except asyncio.QueueShutDown:
            return

        site_name = f"site id {site_id}"
        try:
            job = await sherlock_db.get_ai_extraction_job(
                site_id=site_id,
                contract_hash=contract_hash,
            )
            if job is None:
                if reporter is not None:
                    reporter.ai_job_started(site_name)
                    reporter.ai_job_finished("skipped")
                continue
            site_name = job.site_name
            if reporter is not None:
                reporter.ai_job_started(site_name)
            if job.username not in hydrated_usernames:
                await hydrate_pass_one_key_registry(
                    sherlock_db=sherlock_db,
                    registry=key_registry,
                    username=job.username,
                    contract_hash=contract_hash,
                )
                hydrated_usernames.add(job.username)
            site_content = await asyncio.to_thread(
                extract_profile_content,
                job.response_text,
            )
            if not site_content:
                await sherlock_db.update_result_ai_extraction(
                    site_id=site_id,
                    ai_extraction="{}",
                    contract_hash=contract_hash,
                )
                if reporter is not None:
                    reporter.ai_job_finished("no_facts")
                continue

            response = await ai_service.extract_profile(
                username=job.username,
                site_name=job.site_name,
                site_content=site_content,
                known_profile_keys=key_registry.names(job.username),
            )

            await sherlock_db.update_result_ai_extraction(
                site_id=site_id,
                ai_extraction=json_dumps(
                    response.extraction,
                    ensure_ascii=False,
                ),
                contract_hash=contract_hash,
            )
            key_registry.add(job.username, response.extraction)
            if reporter is not None:
                reporter.ai_job_finished(
                    "with_facts" if response.extraction else "no_facts"
                )
        except Exception as error:
            if reporter is not None:
                reporter.ai_failed(site_name, error)
                reporter.ai_job_finished("pending")
        finally:
            ai_queue.task_done()


async def _drain_deferred_ai_jobs(
    ai_queue: asyncio.Queue[int],
    reporter: TerminalReporter | None = None,
) -> None:
    """Drain queued IDs after model startup fails, leaving rows retryable."""
    while True:
        try:
            await ai_queue.get()
        except asyncio.QueueShutDown:
            return

        try:
            if reporter is not None:
                reporter.ai_job_deferred()
        finally:
            ai_queue.task_done()


async def run_ai_pipeline(
    ai_queue: asyncio.Queue[int],
    sherlock_db: SherlockDB,
    reporter: TerminalReporter | None = None,
    ai_settings: AISettings | None = None,
) -> AIService | None:
    """Load the local model, then own pass-one processing until shutdown."""
    try:
        ai_service = await AIService.create(
            settings=ai_settings,
            trace_callback=(reporter.ai_trace if reporter is not None else None),
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        if reporter is not None:
            reporter.ai_model_failed(error)
        await _drain_deferred_ai_jobs(ai_queue, reporter)
        return None

    if reporter is not None:
        reporter.ai_model_ready()

    try:
        await ai_worker(
            ai_queue=ai_queue,
            sherlock_db=sherlock_db,
            ai_service=ai_service,
            reporter=reporter,
        )
    except BaseException:
        await ai_service.close()
        raise

    return ai_service


def _cancel_ai_pipeline_now(
    *,
    ai_queue: asyncio.Queue[int] | None,
    ai_pipeline_task: asyncio.Task[AIService | None] | None,
) -> None:
    """Signal active pass-one work to stop without waiting for its cleanup."""
    if ai_queue is not None:
        ai_queue.shutdown(immediate=True)
    if ai_pipeline_task is not None and not ai_pipeline_task.done():
        ai_pipeline_task.cancel()


async def sherlock(
    username: str,
    engine: PlaywrightEngine,
    db: SherlockDB,
    site_data: dict[str, dict[str, str]],
    query_notify: QueryNotify,
    enqueue_ai: AIEnqueue | None = None,
    force_ai_extraction: bool = False,
    dump_response: bool = False,
    proxy: str | None = None,
    timeout: int = 60,
    on_cancel: AICancel | None = None,
) -> dict[str, dict[str, str | QueryResult]]:
    """Run Sherlock Analysis.

    Checks for existence of username on various social media sites.

    Keyword Arguments:
    username               -- String indicating username that report
                              should be created against.
    site_data              -- Dictionary containing all of the site data.
    query_notify           -- Object with base type of QueryNotify().
                              This will be used to notify the caller about
                              query results.
    enqueue_ai             -- Optional callback that schedules a saved result
                              for AI extraction.
    force_ai_extraction    -- Re-run AI extraction for freshly saved claimed
                              results even when their source is unchanged.
    proxy                  -- String indicating the proxy URL
    timeout                -- Time in seconds to wait before timing out request.
                              Default is 60 seconds.
    on_cancel              -- Optional synchronous callback invoked before site
                              task cleanup when the scan is interrupted.

    Return Value:
    Dictionary containing results from report. Key of dictionary is the name
    of the social network site, and the value is another dictionary with
    the following keys:
        url_main:      URL of main site.
        url_user:      URL of user on site (if account exists).
        status:        QueryResult() object indicating results of test for
                       account existence.
        http_status:   HTTP status code of query which checked for existence on
                       site.
        response_text: Text that came back from request.  May be None if
                       there was an HTTP error when checking for existence.
    """

    # Notify caller that we are starting the query.
    query_notify.start(username, total=len(site_data))

    # Results from analysis of all sites
    results_total = {}
    tasks = {}
    
    for social_network, net_info in site_data.items():
        # Results from analysis of this specific site
        results_site = {"url_main": net_info.get("urlMain")}

        # Record URL of main site

        headers = {}

        if "headers" in net_info:
            # Override/append any extra headers required by a given site.
            headers.update(net_info["headers"])

        # Some sites silently discard characters ('john.doe' and 'johndoe' are
        # the same account there), so probing the literal string false-negatives.
        site_username = normalize_username(username, net_info.get("strip_bad_char"))

        # The URL a person would open. For the 216 sites whose check endpoint is
        # an API, that is not the URL being probed -- reporting the probe URL
        # would hand the user 'keybase.io/_/api/1.0/user/lookup.json?...'
        # instead of their profile.
        url = interpolate_string(
            net_info.get("urlProfile") or net_info["url"],
            site_username.replace(' ', '%20'),
        )

        # Don't make request if username is invalid for the site
        regex_check = net_info.get("regexCheck")
        if regex_check and re.search(regex_check, username) is None:
            # No need to do the check at the site: this username is not allowed.
            results_site["status"] = QueryResult(
                username, social_network, url, QueryStatus.ILLEGAL
            )
            results_site["url_user"] = ""
            results_site["http_status"] = ""
            results_site["response_text"] = ""
            query_notify.update(results_site["status"])
            results_total[social_network] = results_site
        else:
            # URL of user on site (if it exists)
            results_site["url_user"] = url
            url_probe = net_info.get("urlProbe")
            request_method = net_info.get("request_method")
            request_payload = net_info.get("request_payload")
            request = None


            if request_method is not None:
                request = engine.get_request_fn(request_method)

            if request_payload is not None:
                request_payload = interpolate_string(request_payload, site_username)

            if url_probe is None:
                # What actually gets fetched to decide existence.
                url_probe = interpolate_string(
                    net_info["url"], site_username.replace(' ', '%20')
                )
            else:
                # There is a special URL for probing existence separate
                # from where the user profile normally can be found.
                url_probe = interpolate_string(url_probe, site_username)

            # shared params
            base_kwargs = {
                'url': url_probe,
                'headers': headers,
                'timeout': timeout * 1000,
            }

            if proxy is not None:
                base_kwargs['proxy'] = {"http": proxy, "https": proxy}

            # Every rule needs a response body -- a two-sided rule is decided by
            # its markers, and a marker cannot be matched against a HEAD. The
            # legacy manifest's status-only rules were fetched with HEAD, which
            # is why 54% of confirmed hits reached the AI pipeline with nothing
            # in them.
            transport = preferred_transport(net_info)

            # page.goto can only issue a GET, so a rule needing another method
            # stays on the API transport. Test the method, not whether the key
            # is present: the WMN adapter always records one, so a presence
            # check silently routed every single site to the API and left the
            # browser path dead -- which is exactly how client-rendered sites
            # like Threads and Instagram came back with their markers missing.
            if transport == "browser" and (request_method or "GET") == "GET":
                task = asyncio.create_task(_probe_site(
                    engine=engine,
                    net_info=net_info,
                    site_username=site_username,
                    transport="browser",
                    timeout=timeout,
                    wants_profile_body=enqueue_ai is not None,
                    fetch=engine.fetch_with_page,
                    fetch_kwargs={"wait_until": PAGE_WAIT_UNTIL, **base_kwargs},
                ))
            else:
                # page.goto cannot issue anything but a GET, so a rule with an
                # explicit method stays on the API transport regardless.
                if request is None:
                    request = engine.get_request_fn('GET')
                transport = "api"
                task = asyncio.create_task(_probe_site(
                    engine=engine,
                    net_info=net_info,
                    site_username=site_username,
                    transport=transport,
                    timeout=timeout,
                    wants_profile_body=enqueue_ai is not None,
                    fetch=engine.fetch_with_api,
                    fetch_kwargs={
                        "request_fn": request,
                        "request_payload": request_payload,
                        **base_kwargs,
                    },
                ))

            # store in tasks as key to retrieve later using as_completed
            tasks[task] = {
                'social_network': social_network,
                'net_info': net_info,
                'results_site': results_site,
                'transport': transport,
                'site_username': site_username,
            }
           
    # receive tasks as soon as they are completed
    # use tasks dict retrieve social_network, net_info, results_site
    try:
        async for completed_task in asyncio.as_completed(tasks):
            social_network = tasks[completed_task]['social_network']
            results_site = tasks[completed_task]['results_site']
            net_info = tasks[completed_task]['net_info']
            
            # Retrieve other site information again
            url = results_site.get("url_user")
            status = results_site.get("status")
            if status is not None:
                # We have already determined the user doesn't exist here
                continue

            transport = tasks[completed_task]['transport']
            site_username = tasks[completed_task]['site_username']
            detection_rule = net_info.get("detection")

            r, error_text, _exception_text = await await_response(completed_task=completed_task)

            # Get response time for response of our request.
            try:
                response_time = r.elapsed
            except AttributeError:
                response_time = None

            # Attempt to get request information
            try:
                http_status = r.status
            except Exception:
                http_status = "?"
            try:
                response_text = r.text
            except Exception:
                response_text = ""

            query_status = QueryStatus.UNKNOWN
            query_confidence = None
            error_context = None

            # As WAFs advance and evolve, they will occasionally block Sherlock and
            # lead to false positives and negatives. Fingerprints should be added
            # here to filter results that fail to bypass WAFs. Fingerprints should
            # be highly targetted. Comment at the end of each fingerprint to
            # indicate target and date fingerprinted.
            WAFHitMsgs = [
                r'.loading-spinner{visibility:hidden}body.no-js .challenge-running{display:none}body.dark{background-color:#222;color:#d9d9d9}body.dark a{color:#fff}body.dark a:hover{color:#ee730a;text-decoration:underline}body.dark .lds-ring div{border-color:#999 transparent transparent}body.dark .font-red{color:#b20f03}body.dark', # 2024-05-13 Cloudflare
                r'<span id="challenge-error-text">', # 2024-11-11 Cloudflare error page
                r'AwsWafIntegration.forceRefreshToken', # 2024-11-11 Cloudfront (AWS)
                r'{return l.onPageView}}),Object.defineProperty(r,"perimeterxIdentifiers",{enumerable:' # 2024-04-09 PerimeterX / Human Security
            ]

            if error_text is not None:
                error_context = error_text

            elif any(hitMsg in r.text for hitMsg in WAFHitMsgs):
                query_status = QueryStatus.WAF

            elif not detection_rule:
                error_context = f"No detection rule for {social_network}"
                query_status = QueryStatus.UNKNOWN

            else:
                # Any second request this site needed already happened inside
                # its own task, so the consumer stays cheap and results are
                # reported the moment they land.
                verdict = evaluate(detection_rule, r.status, r.text)

                query_confidence = verdict.confidence
                if verdict.exists is True:
                    query_status = QueryStatus.CLAIMED
                elif verdict.exists is False:
                    query_status = QueryStatus.AVAILABLE
                else:
                    # Neither side of the rule matched. Reporting this rather
                    # than defaulting to CLAIMED is the point of the two-sided
                    # data: a stale rule and a real account no longer look alike.
                    query_status = QueryStatus.UNKNOWN
                    error_context = verdict.reason

            if dump_response:
                dump_lines = [
                    "+++++++++++++++++++++",
                    f"TARGET NAME   : {social_network}",
                    f"USERNAME      : {site_username}",
                    f"TARGET URL    : {url}",
                    f"TRANSPORT     : {transport}",
                ]
                if detection_rule:
                    dump_lines.extend(
                        [
                            f"EXPECT HIT    : {detection_rule['exists']}",
                            f"EXPECT MISS   : {detection_rule['missing']}",
                        ]
                    )
                dump_lines.extend(["Results...", f"RESPONSE CODE : {http_status}"])
                dump_lines.extend(
                    [
                        ">>>>> BEGIN RESPONSE TEXT",
                        str(response_text),
                        "<<<<< END RESPONSE TEXT",
                        f"VERDICT       : {query_status} ({query_confidence})",
                        f"REASON        : {error_context}",
                        "+++++++++++++++++++++",
                    ]
                )
                query_notify.raw("\n".join(dump_lines))

            # Notify caller about results of query.
            result: QueryResult = QueryResult(
                username=username,
                site_name=social_network,
                site_url_user=url,
                status=query_status,
                query_time=response_time,
                context=error_context,
                confidence=query_confidence,
            )
            query_notify.update(result)

            # Detection and extraction want different things. An API endpoint
            # decides existence cheaply but returns a JSON envelope with nothing
            # in it worth reading, so a confirmed hit found that way carries the
            # profile page's markup alongside it. Detection has already read the
            # endpoint its markers were written against; what gets stored and
            # extracted should be the richer body.
            profile_text = getattr(r, "profile_text", None)
            if profile_text:
                response_text = profile_text

            should_run_ai = (
                enqueue_ai is not None
                and query_status is QueryStatus.CLAIMED
                and bool(response_text)
            )

            site_id = await db.save_result(
                username=username,
                site_url=url,
                status_code=http_status,
                status=str(query_status),
                response_text=response_text,
                site_name=social_network,
                query_time_ms=response_time,
                error_context=error_context,
                confidence=str(query_confidence) if query_confidence else None,
                force_ai_extraction=(
                    force_ai_extraction and should_run_ai
                ),
            )
            if should_run_ai:
                await enqueue_ai(site_id)

            # Save status of request
            results_site["status"] = result

            # Save results from request
            results_site["http_status"] = http_status
            results_site["response_text"] = response_text

            # Add this site's results into final dictionary with all of the other results.
            results_total[social_network] = results_site
    except (asyncio.CancelledError, KeyboardInterrupt) as interruption:
        if on_cancel is not None:
            try:
                on_cancel()
            except BaseException as error:
                interruption.add_note(
                    "Cancellation callback failed: "
                    f"{type(error).__name__}: {error}"
                )

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results_total


def timeout_check(value):
    """Check Timeout Argument.

    Checks timeout for validity.

    Keyword Arguments:
    value                  -- Time in seconds to wait before timing out request.

    Return Value:
    Floating point number representing the time (in seconds) that should be
    used for the timeout.

    NOTE:  Will raise an exception if the timeout in invalid.
    """

    float_value = float(value)

    if float_value <= 0:
        raise ArgumentTypeError(
            f"Invalid timeout value: {value}. Timeout must be a positive number."
        )

    return float_value

async def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1:3] == ["setup", "ai"]:
        return await run_ai_setup(sys.argv[3:])
    # Same argv-sniffing shape as `setup ai` above: intercepted before the scan
    # parser is built, so `show` never has to satisfy the scan's arguments.
    # The cost is that a username literally called "show" cannot be scanned as
    # a bare argument, the same trade already made for "setup".
    if len(sys.argv) >= 2 and sys.argv[1] == "show":
        return await run_show(sys.argv[2:])
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=f"{__longname__} (Version {__version__})",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__shortname__} v{__version__}",
        help="Display version information and dependencies.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        "-d",
        "--debug",
        action="store_true",
        dest="verbose",
        default=False,
        help="Display extra debugging information and metrics.",
    )
    parser.add_argument(
        "--folderoutput",
        "-fo",
        dest="folderoutput",
        help="If using multiple usernames, the output of the results will be saved to this folder.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        help="If using single username, the output of the result will be saved to this file.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        dest="csv",
        default=False,
        help="Create Comma-Separated Values (CSV) File.",
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        dest="xlsx",
        default=False,
        help="Create the standard file for the modern Microsoft Excel spreadsheet (xlsx).",
    )
    parser.add_argument(
        "--site",
        action="append",
        metavar="SITE_NAME",
        dest="site_list",
        default=[],
        help="Limit analysis to just the listed sites. Add multiple options to specify more than one site.",
    )
    parser.add_argument(
        "--proxy",
        "-p",
        metavar="PROXY_URL",
        action="store",
        dest="proxy",
        default=None,
        help="Make requests over a proxy. e.g. socks5://127.0.0.1:1080",
    )
    parser.add_argument(
        "--dump-response",
        action="store_true",
        dest="dump_response",
        default=False,
        help="Dump the HTTP response to stdout for targeted debugging.",
    )
    parser.add_argument(
        "--json",
        "-j",
        metavar="JSON_FILE",
        dest="json_file",
        default=None,
        help="Load data from a JSON file or an online, valid, JSON file. Upstream PR numbers also accepted.",
    )
    parser.add_argument(
        "--timeout",
        action="store",
        metavar="TIMEOUT",
        dest="timeout",
        type=timeout_check,
        default=60,
        help="Time (in seconds) to wait for response to requests (Default: 60)",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        dest="print_all",
        default=False,
        help="Output sites where the username was not found.",
    )
    parser.add_argument(
        "--print-found",
        action="store_true",
        dest="print_found",
        default=True,
        help="Output sites where the username was found (also if exported as file).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        dest="no_color",
        default=False,
        help="Don't color terminal output",
    )
    parser.add_argument(
        "username",
        nargs="+",
        metavar="USERNAMES",
        action="store",
        help="One or more usernames to check with social networks. Check similar usernames using {?} (replace to '_', '-', '.').",
    )
    parser.add_argument(
        "--browse",
        "-b",
        action="store_true",
        dest="browse",
        default=False,
        help="Browse to all results on default browser.",
    )

    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        default=False,
        help="Force the use of the local data.json file.",
    )

    parser.add_argument(
        "--nsfw",
        action="store_true",
        default=False,
        help="Include checking of NSFW sites from default list.",
    )

    parser.add_argument(
        "--txt",
        action="store_true",
        dest="output_txt",
        default=False,
        help="Enable creation of a txt file",
    )

    parser.add_argument(
        "--ignore-exclusions",
        action="store_true",
        dest="ignore_exclusions",
        default=False,
        help="Ignore upstream exclusions (may return more false positives)",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help=(
            "Re-scan every site for the username instead of skipping sites "
            "already saved in the database. Cached AI extractions are kept for "
            "sites whose content did not change."
        ),
    )

    parser.add_argument(
        "--ai",
        action="store_true",
        default=False,
        help=(
            "Extract structured profile data with the configured local AI "
            "model. With --site, refresh only those sites and run pass one only."
        ),
    )

    parser.add_argument(
        "--ai-synthesize-only",
        action="store_true",
        default=False,
        help="Build AI profiles from existing database extractions without scanning.",
    )

    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        type=parse_inline_anchor,
        metavar="[TRUST:]FIELD=VALUE",
        help=(
            "Add a run-only identity anchor to every username. Repeat for "
            "multiple values; unprefixed values use context trust."
        ),
    )

    parser.add_argument(
        "--force-ai-synthesis",
        action="store_true",
        default=False,
        help=(
            "Rebuild AI profiles during a normal --ai run even when inputs are "
            "unchanged. --ai-synthesize-only always rebuilds."
        ),
    )

    args = parser.parse_args()
    targeted_ai_scan = args.ai and bool(args.site_list)

    if args.ai and args.ai_synthesize_only:
        parser.error("--ai and --ai-synthesize-only cannot be used together")
    synthesis_only_scan_options: list[str] = []
    if args.ai_synthesize_only and args.site_list:
        synthesis_only_scan_options.append("--site")
    if args.ai_synthesize_only and args.local:
        synthesis_only_scan_options.append("--local")
    if args.ai_synthesize_only and args.fresh:
        synthesis_only_scan_options.append("--fresh")
    if synthesis_only_scan_options:
        parser.error(
            f"{' and '.join(synthesis_only_scan_options)} cannot be used with "
            "--ai-synthesize-only. Synthesis-only performs no scan and uses all "
            "saved extractions. Refresh selected sites with --ai --site first, "
            "then run --ai-synthesize-only separately."
        )
    if args.anchor and not (args.ai or args.ai_synthesize_only):
        parser.error("--anchor requires --ai or --ai-synthesize-only")
    if args.force_ai_synthesis and not (args.ai or args.ai_synthesize_only):
        parser.error("--force-ai-synthesis requires an AI mode")
    if targeted_ai_scan and (args.anchor or args.force_ai_synthesis):
        parser.error(
            "--anchor and --force-ai-synthesis affect pass two, "
            "but --site --ai runs pass one only. Refresh the site without those "
            "options, then run --ai-synthesize-only with your anchors."
        )

    all_usernames = expand_usernames(args.username)
    needs_ai_model = args.ai or (
        args.ai_synthesize_only
        and build_investigation_context(args.anchor).has_anchors
    )
    ai_settings: AISettings | None = None
    if needs_ai_model:
        try:
            ai_settings = load_ai_settings()
        except AIConfigError as error:
            parser.error(str(error))

    query_notify = QueryNotifyPrint(
        result=None,
        verbose=args.verbose,
        print_all=args.print_all,
        browse=args.browse,
        no_color=args.no_color,
    )
    query_notify.debug("Verbose diagnostics enabled")
    if ai_settings is not None:
        query_notify.ai_configuration(
            model=ai_settings.model,
            base_url=ai_settings.base_url,
            temperature=ai_settings.temperature,
            context_length=ai_settings.context_length,
        )
    start_time = perf_counter()
    if args.ai_synthesize_only:
        query_notify.info(
            "Synthesis-only mode: rebuilding profiles from saved extractions"
        )
        interrupted = False
        try:
            await run_synthesis_only(
                usernames=all_usernames,
                force=True,
                inline_anchors=args.anchor,
                reporter=query_notify,
                ai_settings=ai_settings,
            )
        except asyncio.CancelledError:
            interrupted = True
        finally:
            if interrupted:
                query_notify.processing_interrupted()
            else:
                query_notify.finish(elapsed_time=perf_counter() - start_time)
        return 130 if interrupted else 0

    # Check for newer version of Sherlock. If it exists, let the user know about it
    try:
        # requests is synchronous; off-thread so a slow forge cannot stall the
        # event loop for the full timeout before the scan has even started.
        latest_release_response = await asyncio.to_thread(
            requests.get, forge_api_latest_release, timeout=10
        )
        latest_release_raw = latest_release_response.text
        latest_release_json = json_loads(latest_release_raw)
        latest_remote_tag = latest_release_json["tag_name"]

        if latest_remote_tag[1:] != __version__:
            query_notify.update_available(
                __version__,
                latest_remote_tag[1:],
                latest_release_json["html_url"],
            )

    except Exception as error:
        query_notify.debug(
            f"Sherlock update check failed ({type(error).__name__})"
        )

    if args.proxy is not None:
        query_notify.info(f"Using proxy {args.proxy}")

    # Check if both output methods are entered as input.
    if args.output is not None and args.folderoutput is not None:
        query_notify.fatal("You can only use one output method")
        sys.exit(1)

    # Check validity for single username output.
    if args.output is not None and len(args.username) != 1:
        query_notify.fatal("--output can only be used with one username")
        sys.exit(1)

    # Create object with all information about sites we are aware of.
    try:
        if args.local:
            # The bundled manifest is already the default; --local now only
            # guarantees it, rather than switching away from a remote fetch.
            sites = SitesInformation(honor_exclusions=False)
        else:
            json_file_location = args.json_file
            # If --json parameter is a number, interpret it as a pull request number
            if args.json_file and args.json_file.isnumeric():
                pull_number = args.json_file
                pull_url = f"https://api.github.com/repos/sherlock-project/sherlock/pulls/{pull_number}"
                pull_request_response = await asyncio.to_thread(
                    requests.get, pull_url, timeout=10
                )
                pull_request_json = json_loads(pull_request_response.text)

                # Check if it's a valid pull request
                if "message" in pull_request_json:
                    query_notify.fatal(
                        f"Pull request #{pull_number} was not found"
                    )
                    sys.exit(1)

                head_commit_sha = pull_request_json["head"]["sha"]
                json_file_location = f"https://raw.githubusercontent.com/sherlock-project/sherlock/{head_commit_sha}/sherlock_project/resources/data.json"

            sites = SitesInformation(
                data_file_path=json_file_location,
                honor_exclusions=not args.ignore_exclusions,
                do_not_exclude=args.site_list,
            )
    except Exception as error:
        query_notify.fatal(
            "Unable to load site definitions",
            detail=type(error).__name__,
        )
        sys.exit(1)

    if not args.nsfw:
        sites.remove_nsfw_sites(do_not_remove=args.site_list)

    # Create original dictionary from SitesInformation() object.
    # Eventually, the rest of the code will be updated to use the new object
    # directly, but this will glue the two pieces together.
    site_data_all = {site.name: site.information for site in sites}
    if args.site_list == []:
        # Not desired to look at a sub-set of sites
        site_data = site_data_all
    else:
        # User desires to selectively run queries on a sub-set of the site list.
        # Make sure that the sites are supported & build up pruned site database.
        site_data = {}
        site_missing = []
        for site in args.site_list:
            counter = 0
            for existing_site, existing_data in site_data_all.items():
                if site.lower() == existing_site.lower():
                    site_data[existing_site] = existing_data
                    counter += 1
            if counter == 0:
                # Build up list of sites not supported for future error message.
                site_missing.append(f"'{site}'")

        if site_missing:
            query_notify.warning(
                f"Requested sites were not found: {', '.join(site_missing)}"
            )

        if not site_data:
            query_notify.fatal("None of the requested sites are available")
            sys.exit(1)

    # Run report on all specified users.
    database_path = default_database_path()
    query_notify.debug(f"Using database at {database_path}")
    db = await SherlockDB.create(str(database_path))
    results = {}
    ai_service: AIService | None = None
    ai_queue: asyncio.Queue[int] | None = None
    ai_pipeline_task: asyncio.Task[AIService | None] | None = None
    ai_pipeline_awaited = False
    enqueue_ai_callback: AIEnqueue | None = None
    scheduled_ai_ids: set[int] = set()
    current_pass_one_contract_hash: str | None = None
    interrupted = False
    ai_cancellation_signalled = False

    def cancel_ai_pipeline() -> None:
        nonlocal ai_cancellation_signalled
        if ai_cancellation_signalled:
            return
        ai_cancellation_signalled = True
        _cancel_ai_pipeline_now(
            ai_queue=ai_queue,
            ai_pipeline_task=ai_pipeline_task,
        )

    try:
        if args.ai:
            current_pass_one_contract_hash = pass_one_contract_hash()
            if not targeted_ai_scan:
                await report_cached_ai_evidence(
                    db=db,
                    usernames=all_usernames,
                    contract_hash=current_pass_one_contract_hash,
                    reporter=query_notify,
                )
            query_notify.ai_model_starting()
            if targeted_ai_scan:
                query_notify.targeted_ai_mode()
            ai_queue = asyncio.Queue()
            ai_pipeline_task = asyncio.create_task(
                run_ai_pipeline(
                    ai_queue=ai_queue,
                    sherlock_db=db,
                    reporter=query_notify,
                    ai_settings=ai_settings,
                ),
                name="ai-pipeline",
            )
            # Let model loading begin before browser startup and site scanning.
            await asyncio.sleep(0)

            async def enqueue_ai_result(site_id: int) -> None:
                if site_id in scheduled_ai_ids:
                    return

                if ai_queue is None:
                    raise RuntimeError("AI queue is not available")

                scheduled_ai_ids.add(site_id)
                try:
                    await ai_queue.put(site_id)
                    query_notify.ai_scheduled()
                except BaseException:
                    scheduled_ai_ids.remove(site_id)
                    raise

            enqueue_ai_callback = enqueue_ai_result

        async with PlaywrightEngine(
            headless=True,
            status_callback=query_notify.browser_status,
            cancellation_callback=cancel_ai_pipeline if args.ai else None,
        ) as engine:
            for username in all_usernames:
                # If no site list was provided, skip sites already in the
                # database. --fresh opts out of that resume behaviour and
                # re-checks everything.
                restored_results: dict = {}
                if not args.site_list and not args.fresh:
                    saved_rows = await db.get_saved_results(username=username)
                    site_data = {
                        site_name: site_data_all[site_name]
                        for site_name in site_data_all
                        if site_name not in saved_rows
                    }
                    # Everything skipped above still belongs in the report.
                    # Without this the run describes itself rather than the
                    # username, and a second scan looks like it found nothing.
                    restored_results = restore_saved_results(
                        username=username,
                        saved_rows=saved_rows,
                        site_data_all=site_data_all,
                    )
                    query_notify.restored_results(
                        username=username,
                        results=restored_results,
                        to_scan=len(site_data),
                    )

                # An empty site_data here means every site was already stored;
                # an unmatched --site exits fatally further up, so this cannot
                # be a bad selection. Running the scan anyway would announce
                # "checking 0 sites" and "0 found" directly beneath the stored
                # results we just reported.
                results: dict = {}
                if site_data:
                    scan_started_at = perf_counter()
                    results = await sherlock(
                        username=username,
                        engine=engine,
                        db=db,
                        site_data=site_data,
                        query_notify=query_notify,
                        dump_response=args.dump_response,
                        proxy=args.proxy,
                        timeout=args.timeout,
                        enqueue_ai=enqueue_ai_callback,
                        force_ai_extraction=targeted_ai_scan,
                        on_cancel=cancel_ai_pipeline if args.ai else None,
                    )
                    query_notify.finish_scan(
                        elapsed_time=perf_counter() - scan_started_at
                    )
                # Restored first so freshly-scanned sites win on any overlap.
                # There should be none by construction, but the scan is the
                # newer evidence either way.
                results = {**restored_results, **results}

                if enqueue_ai_callback is not None and not targeted_ai_scan:
                    if current_pass_one_contract_hash is None:
                        raise RuntimeError("Pass 1 contract hash is unavailable")
                    pending_ids = await db.get_pending_ai_extraction_ids(
                        username,
                        contract_hash=current_pass_one_contract_hash,
                    )
                    for site_id in pending_ids:
                        await enqueue_ai_callback(site_id)

        if ai_queue is not None and ai_pipeline_task is not None:
            query_notify.ai_draining()
            ai_queue.shutdown()
            ai_service = await ai_pipeline_task
            ai_pipeline_awaited = True
            await ai_queue.join()
            query_notify.ai_pass_finished()
            if targeted_ai_scan and ai_service is not None:
                query_notify.targeted_ai_complete()

        if ai_service is not None and not targeted_ai_scan:
            await synthesize_profiles(
                db=db,
                ai_service=ai_service,
                usernames=all_usernames,
                force=args.force_ai_synthesis,
                inline_anchors=args.anchor,
                reporter=query_notify,
            )

    except asyncio.CancelledError:
        interrupted = True

    finally:
        if ai_pipeline_task is not None and not ai_pipeline_awaited:
            cancel_ai_pipeline()

            pipeline_result = (
                await asyncio.gather(
                    ai_pipeline_task,
                    return_exceptions=True,
                )
            )[0]
            if (
                ai_service is None
                and pipeline_result is not None
                and not isinstance(pipeline_result, BaseException)
            ):
                ai_service = pipeline_result

        try:
            cleanup_error, cleanup_cancellation = (
                await _close_ai_service_and_db(
                    ai_service=ai_service,
                    db=db,
                )
            )
        finally:
            query_notify.close()

        if cleanup_cancellation is not None:
            interrupted = True
        if cleanup_error is not None and not interrupted:
            raise cleanup_error

    if interrupted:
        query_notify.processing_interrupted()
        return 130

    elapsed_time = (perf_counter() - start_time)
    if args.output:
        result_file = args.output
    elif args.folderoutput:
        # The usernames results should be stored in a targeted folder.
        # If the folder doesn't exist, create it first
        os.makedirs(args.folderoutput, exist_ok=True)
        result_file = os.path.join(args.folderoutput, f"{username}.txt")
    else:
        result_file = f"{username}.txt"

    if args.output_txt:
        with open(result_file, "w", encoding="utf-8") as file:
            exists_counter = 0
            for website_name in results:
                dictionary = results[website_name]
                if dictionary.get("status").status == QueryStatus.CLAIMED:
                    exists_counter += 1
                    file.write(dictionary["url_user"] + "\n")
            file.write(f"Total Websites Username Detected On : {exists_counter}\n")

    if args.csv:
        result_file = f"{username}.csv"
        if args.folderoutput:
            # The usernames results should be stored in a targeted folder.
            # If the folder doesn't exist, create it first
            os.makedirs(args.folderoutput, exist_ok=True)
            result_file = os.path.join(args.folderoutput, result_file)

        with open(result_file, "w", newline="", encoding="utf-8") as csv_report:
            writer = csv.writer(csv_report)
            writer.writerow(
                [
                    "username",
                    "name",
                    "url_main",
                    "url_user",
                    "exists",
                    "http_status",
                    "response_time_s",
                ]
            )
            for site in results:
                if (
                    args.print_found
                    and not args.print_all
                    and results[site]["status"].status != QueryStatus.CLAIMED
                ):
                    continue

                response_time_s = results[site]["status"].query_time
                if response_time_s is None:
                    response_time_s = ""
                writer.writerow(
                    [
                        username,
                        site,
                        results[site]["url_main"],
                        results[site]["url_user"],
                        str(results[site]["status"].status),
                        results[site]["http_status"],
                        response_time_s,
                    ]
                )
    if args.xlsx:
        usernames = []
        names = []
        url_main = []
        url_user = []
        exists = []
        http_status = []
        response_time_s = []

        for site in results:
            if (
                args.print_found
                and not args.print_all
                and results[site]["status"].status != QueryStatus.CLAIMED
            ):
                continue

            if response_time_s is None:
                response_time_s.append("")
            else:
                response_time_s.append(results[site]["status"].query_time)
            usernames.append(username)
            names.append(site)
            url_main.append(results[site]["url_main"])
            url_user.append(results[site]["url_user"])
            exists.append(str(results[site]["status"].status))
            http_status.append(results[site]["http_status"])

        DataFrame = pd.DataFrame(
            {
                "username": usernames,
                "name": names,
                "url_main": [f'=HYPERLINK(\"{u}\")' for u in url_main],
                "url_user": [f'=HYPERLINK(\"{u}\")' for u in url_user],
                "exists": exists,
                "http_status": http_status,
                "response_time_s": response_time_s,
            }
        )
        DataFrame.to_excel(f"{username}.xlsx", sheet_name="sheet1", index=False)

    query_notify.finish(elapsed_time=elapsed_time)
    return 0


def cli() -> None:
    """Synchronous console-script entrypoint."""
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print(INTERRUPTION_MESSAGE)
        raise SystemExit(130) from None

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    cli()
