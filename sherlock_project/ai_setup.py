"""Interactive and automatable setup for Sherlock's AI provider."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from sherlock_project.ai_config import (
    DEFAULT_AI_CONTEXT_LENGTH,
    DEFAULT_AI_TEMPERATURE,
    DEFAULT_LM_STUDIO_BASE_URL,
    AIConfigError,
    AISettings,
    ai_config_path,
    save_ai_settings,
    try_load_ai_settings,
)
from sherlock_project.ai_provider import AIModelInfo, AIProviderError, LMStudioProvider


def _lms_server_url() -> str | None:
    try:
        completed = subprocess.run(
            ["lms", "server", "status", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("running") is not True:
        return None
    port = payload.get("port")
    if (
        not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        return None
    return f"http://127.0.0.1:{port}"


def discover_setup_base_url(
    explicit: str | None,
    *,
    existing: AISettings | None,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    return (
        explicit
        or environment.get("LM_STUDIO_BASE_URL")
        or (existing.base_url if existing is not None else None)
        or _lms_server_url()
        or DEFAULT_LM_STUDIO_BASE_URL
    )


def _model_table(models: Sequence[AIModelInfo]) -> Table:
    table = Table(title="Downloaded LM Studio models")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Quant")
    table.add_column("Thinking")
    table.add_column("State")
    for index, model in enumerate(models, start=1):
        if not model.reasoning_options:
            thinking = "not used"
        elif model.supports_reasoning_off:
            thinking = "off supported"
        else:
            thinking = "required"
        table.add_row(
            str(index),
            f"{model.display_name}\n[dim]{model.key}[/dim]",
            model.params or "?",
            model.quantization or "?",
            thinking,
            "loaded" if model.loaded else "downloaded",
        )
    return table


def _select_model(
    *,
    parser: ArgumentParser,
    models: list[AIModelInfo],
    requested: str | None,
    existing: AISettings | None,
    console: Console,
    interactive: bool,
) -> AIModelInfo:
    if requested:
        selected = next((model for model in models if model.key == requested), None)
        if selected is None:
            parser.error(f"LM Studio model {requested!r} is not downloaded")
        if not selected.supports_reasoning_off:
            parser.error(
                f"LM Studio model {requested!r} requires native thinking and "
                "is incompatible with Sherlock's reasoning-off pipeline"
            )
        return selected

    compatible = [model for model in models if model.supports_reasoning_off]
    if not compatible:
        parser.error(
            "No downloaded LM Studio LLM can run with native thinking disabled"
        )
    if not interactive:
        parser.error("--model is required when setup input is not interactive")

    console.print(_model_table(models))
    default_index = models.index(compatible[0]) + 1
    if existing is not None:
        for index, model in enumerate(models, start=1):
            if model.key == existing.model and model.supports_reasoning_off:
                default_index = index
                break
    while True:
        selected_index = IntPrompt.ask(
            "Select a model",
            default=default_index,
            console=console,
        )
        if not 1 <= selected_index <= len(models):
            console.print("[yellow]Choose a number from the table.[/yellow]")
            continue
        selected = models[selected_index - 1]
        if not selected.supports_reasoning_off:
            console.print(
                "[yellow]That model requires native thinking. Choose a model "
                "marked 'off supported' or 'not used'.[/yellow]"
            )
            continue
        return selected


def build_setup_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="sherlock setup ai")
    parser.add_argument(
        "--base-url",
        help="LM Studio server URL. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--model",
        help="Exact downloaded LM Studio model key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=f"Extraction temperature (default: {DEFAULT_AI_TEMPERATURE}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Print the stored configuration and its file path, then exit. "
            "Reads the config file only -- no server required."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable terminal colors.",
    )
    return parser


def _report_settings(
    settings: AISettings | None,
    *,
    path: Path,
    console: Console,
    environ: Mapping[str, str],
) -> int:
    """Print the stored configuration without contacting the provider.

    The wizard is otherwise the only way to see which model is configured, and
    it needs a live server to get that far -- so reading your own settings used
    to require running the software those settings point at. The config path is
    printed because it resolves to a per-user directory that differs by OS and
    is not otherwise shown anywhere, which is what makes "just open the file"
    unhelpful advice.
    """
    if settings is None:
        console.print("[yellow][!] AI is not configured.[/yellow]")
        console.print(
            Text("Nothing stored at ", style="dim").append(Text(str(path))),
            soft_wrap=True,
        )
        console.print("Configure it with `sherlock setup ai`.")
        # 1, not 2: nothing is broken, the answer is "nothing is stored". Same
        # convention as `sherlock show`, so a script can branch on it.
        return 1

    for label, value in (
        ("provider", settings.provider),
        ("model", settings.model),
        ("endpoint", settings.base_url),
        ("temperature", str(settings.temperature)),
        ("context length", str(settings.context_length)),
    ):
        # Text(), not an f-string into markup: a model key or URL carrying
        # square brackets would otherwise be eaten as a Rich style tag.
        # soft_wrap because Rich otherwise word-wraps to the terminal width and
        # will break a long model key or URL across lines -- the same defect
        # that once produced invalid JSON out of `show --json`.
        console.print(
            Text(f"{label:>15}: ", style="dim").append(Text(value)),
            soft_wrap=True,
        )
    console.print()
    console.print(Text("config file: ", style="dim").append(Text(str(path))),
                  soft_wrap=True)

    # load_ai_settings has already applied the override, so the endpoint above
    # is the effective one, not necessarily what the file says. Saying so is
    # the difference between a useful readout and a misleading one.
    if environ.get("LM_STUDIO_BASE_URL"):
        console.print(
            "[yellow][!] Endpoint comes from LM_STUDIO_BASE_URL, which "
            "overrides the config file.[/yellow]"
        )
    return 0


async def run_ai_setup(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    config_path: Path | None = None,
    console: Console | None = None,
    stdin_isatty: bool | None = None,
) -> int:
    parser = build_setup_parser()
    args = parser.parse_args(list(argv))
    environment = os.environ if environ is None else environ
    destination = config_path or ai_config_path(environment)
    existing = try_load_ai_settings(path=destination, environ=environment)

    if args.show:
        conflicting = [
            name
            for name, value in (
                ("--base-url", args.base_url),
                ("--model", args.model),
                ("--temperature", args.temperature),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(
                f"--show reads the stored configuration; "
                f"{' and '.join(conflicting)} would change it"
            )
        return _report_settings(
            existing,
            path=destination,
            console=console
            or Console(
                no_color=args.no_color,
                color_system=None if args.no_color else "auto",
                highlight=False,
            ),
            environ=environment,
        )

    base_url = discover_setup_base_url(
        args.base_url,
        existing=existing,
        environ=environment,
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else (
            existing.temperature
            if existing is not None
            else DEFAULT_AI_TEMPERATURE
        )
    )
    provisional_model = args.model or (existing.model if existing else "setup")
    try:
        provisional = AISettings(
            base_url=base_url,
            model=provisional_model,
            temperature=temperature,
            context_length=(
                existing.context_length
                if existing is not None
                else DEFAULT_AI_CONTEXT_LENGTH
            ),
        )
    except ValueError as error:
        parser.error(str(error))

    output = console or Console(
        no_color=args.no_color,
        color_system=None if args.no_color else "auto",
        highlight=False,
    )
    provider = LMStudioProvider(
        provisional,
        api_token=environment.get("LM_API_TOKEN"),
    )
    try:
        models = await provider.list_models()
    except AIProviderError as error:
        output.print(f"[red]\\[x] {error}[/red]")
        output.print(
            "Start the LM Studio server, then run `sherlock setup ai` again."
        )
        return 2
    finally:
        await provider.close()

    if not models:
        output.print("[red]\\[x] LM Studio has no downloaded LLMs.[/red]")
        output.print("Download a model in LM Studio, then run setup again.")
        return 2

    models.sort(key=lambda model: (model.display_name.casefold(), model.key))
    selected = _select_model(
        parser=parser,
        models=models,
        requested=args.model,
        existing=existing,
        console=output,
        interactive=(
            sys.stdin.isatty() if stdin_isatty is None else stdin_isatty
        ),
    )
    settings = provisional.model_copy(update={"model": selected.key})
    try:
        saved_to = save_ai_settings(
            settings,
            path=destination,
            environ=environment,
        )
    except AIConfigError as error:
        output.print(f"[red]\\[x] {error}[/red]")
        return 2

    output.print(
        f"[green][+] AI configured with {selected.display_name}[/green]"
    )
    # soft_wrap: the config path is long and Rich would otherwise break it
    # across lines, leaving a path nobody can copy.
    output.print(Text(str(saved_to), style="dim"), soft_wrap=True)
    return 0
