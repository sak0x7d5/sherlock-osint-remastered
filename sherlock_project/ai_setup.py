"""Interactive and automatable setup for Sherlock's AI provider."""

from __future__ import annotations

import os
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
    DEFAULT_LLAMACPP_BASE_URL,
    AIConfigError,
    AISettings,
    ai_config_path,
    save_ai_settings,
    try_load_ai_settings,
)
from sherlock_project.ai_provider import AIModelInfo, AIProviderError, LlamaCppProvider


def discover_setup_base_url(
    explicit: str | None,
    *,
    existing: AISettings | None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Where to look for llama-server, most specific source first.

    There used to be a fourth source between the stored value and the default:
    shelling out to `lms server status --json` to ask LM Studio which port it
    had picked. llama.cpp has no such CLI -- the port is whatever `--port` was
    passed at launch and nothing publishes it -- so discovery is gone and the
    default is simply llama-server's own 8080.
    """
    environment = os.environ if environ is None else environ
    return (
        explicit
        or environment.get("LLAMA_SERVER_BASE_URL")
        or (existing.base_url if existing is not None else None)
        or DEFAULT_LLAMACPP_BASE_URL
    )


def _model_table(models: Sequence[AIModelInfo]) -> Table:
    table = Table(title="Model loaded by llama-server")
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


def _warn_native_reasoning(model: AIModelInfo, *, console: Console) -> None:
    """Say what picking an always-thinking model costs, without blocking it.

    These used to be refused outright, which turned "the only model I have"
    into a dead end for about a quarter of a typical library -- reasoning
    models are exactly what many people have downloaded lately. They are not
    actually incompatible: the request declares reasoning explicitly and LM
    Studio returns thinking as a separate output item, so the structured reply
    still parses.

    What they cost is no longer the double reasoning it used to be: Pass 1
    sends such a model a variant prompt and schema that ask for the reasoning
    natively and forbid it in the JSON, so the traversal happens once. What is
    left is that the tuned path, the examples, and the measurements are all on
    the other arrangement, and that thinking is not free in time. Say that,
    rather than implying the model is broken or that it is fine.
    """
    if model.supports_reasoning_off:
        return
    console.print(
        f"[yellow]\\[!] {model.display_name} always thinks natively and cannot "
        "be told not to.[/yellow]"
    )
    console.print(
        "    Per-site extraction switches to a prompt that asks for that "
        "thinking natively instead of inside the answer, so the work is not "
        "done twice. Each site is still slower, and extraction quality is "
        "likely to be lower than a model that can think with reasoning off: "
        "this path is the less tested of the two. Extra output budget is "
        "allowed to keep answers from being cut off."
    )
    if model.reasoning_options:
        console.print(
            Text(
                "    thinking modes offered by this model: "
                + ", ".join(model.reasoning_options),
                style="dim",
            )
        )


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
            parser.error(f"llama-server has not loaded model {requested!r}")
        _warn_native_reasoning(selected, console=console)
        return selected

    preferred = [model for model in models if model.supports_reasoning_off]
    if not interactive:
        parser.error("--model is required when setup input is not interactive")

    console.print(_model_table(models))
    if not preferred:
        console.print(
            "[yellow]\\[!] None of your models can turn native thinking off. "
            "Any of them can still be used -- see the note below.[/yellow]"
        )
    default_index = models.index(preferred[0]) + 1 if preferred else 1
    if existing is not None:
        for index, model in enumerate(models, start=1):
            if model.key == existing.model:
                default_index = index
                break
    while True:
        try:
            selected_index = IntPrompt.ask(
                "Select a model",
                default=default_index,
                console=console,
            )
        except EOFError:
            # isatty() is not a reliable interactivity test on Windows, so the
            # guard above can be skipped even when nothing can answer. NUL is a
            # character device, which means `sherlock setup ai < NUL` --
            # explicitly "I have no keyboard" -- reports isatty() as True.
            # Measured: piped stdin gives False, NUL gives True.
            #
            # Reaching a prompt with no input is the same situation the guard
            # exists for, so it gets the same message instead of an unhandled
            # EOFError traceback out of rich. This hits scheduled tasks, CI
            # runners, Docker without -i, and service contexts.
            #
            # KeyboardInterrupt is deliberately NOT caught: cli() already turns
            # it into the standard interruption message and exit 130, and
            # routing it here would downgrade a clean cancel into an error.
            parser.error(
                "--model is required when setup input is not interactive"
            )
        if not 1 <= selected_index <= len(models):
            console.print("[yellow]Choose a number from the table.[/yellow]")
            continue
        selected = models[selected_index - 1]
        _warn_native_reasoning(selected, console=console)
        return selected


def build_setup_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="sherlock setup ai")
    parser.add_argument(
        "--base-url",
        help="llama-server URL. Defaults to http://127.0.0.1:8080.",
    )
    parser.add_argument(
        "--model",
        help="Model key to record. Defaults to whatever llama-server loaded.",
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
    if environ.get("LLAMA_SERVER_BASE_URL"):
        console.print(
            "[yellow][!] Endpoint comes from LLAMA_SERVER_BASE_URL, which "
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
    provider = LlamaCppProvider(
        provisional,
        api_token=environment.get("LLAMA_API_TOKEN"),
    )
    try:
        models = await provider.list_models()
    except AIProviderError as error:
        output.print(f"[red]\\[x] {error}[/red]")
        output.print(
            "Start llama-server, then run `sherlock setup ai` again, e.g.\n"
            "  llama-server -m model.gguf -c 8192 --port 8080 --jinja "
            "--reasoning-format deepseek"
        )
        return 2
    finally:
        await provider.close()

    if not models:
        output.print("[red]\\[x] llama-server has no model loaded.[/red]")
        output.print(
            "It serves whatever GGUF it was launched with -- restart it "
            "with -m pointing at a model file."
        )
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
