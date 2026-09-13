"""Interactive and automatable setup for Sherlock's AI provider."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
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
from sherlock_project.llama_server import LlamaServerError, ManagedLlamaServer


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
    # No Thinking column. Whether a model can be told to stop thinking is not
    # in anything llama-server publishes -- `/v1/models` carries modalities and
    # no capability block -- so the only way to fill it would be to load every
    # model in the list and ask. For a directory of nineteen that is minutes
    # and gigabytes to populate one column. Printing "not used" on every row
    # instead was worse than leaving it out: it read as a determination when it
    # was really "did not ask". The answer is measured once, against the model
    # actually chosen, by the provider's reasoning probe.
    table = Table(title="Models llama-server can serve")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Quant")
    table.add_column("State")
    for index, model in enumerate(models, start=1):
        table.add_row(
            str(index),
            f"{model.display_name}\n[dim]{model.key}[/dim]",
            model.params or "?",
            model.quantization or "?",
            # "downloaded" was LM Studio's word for it. In router mode nothing
            # is downloaded here -- every entry is already on disk, and the
            # only question is whether it is in memory yet.
            "loaded" if model.loaded else "available",
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


def _ask_for_models_folder(
    *,
    console: Console,
    error: LlamaServerError,
) -> str | None:
    """Ask where the models are. One question, no searching.

    Reached only when discovery came up empty, and only when someone is there
    to answer. The alternative -- printing `sherlock-rm setup ai --models-dir
    <folder>` and exiting -- ends the session by handing back a command,
    which is the thing the whole flow is meant to avoid.

    Nothing is guessed. Sherlock does not search a machine for models it was
    never told about -- that was overhead and coupling to other tools' private
    files, for a question one line answers. A CLI cannot browse, so this asks;
    the TUI picker has a folder row that opens a real browser, which is the
    better door for anyone who has it.
    """
    console.print(f"[yellow]\\[!] {error}[/yellow]")
    try:
        answer = Prompt.ask(
            "Path to your models folder (blank to give up)",
            default="",
            console=console,
        ).strip()
    except EOFError:
        # Same situation the interactive guard covers: isatty() is not a
        # reliable answer on Windows, so a prompt can be reached with nothing
        # able to reply. Give up quietly rather than raise out of rich.
        return None
    if not answer:
        return None
    return answer


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
            # character device, which means `sherlock-rm setup ai < NUL` --
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
    parser = ArgumentParser(prog="sherlock-rm setup ai")
    parser.add_argument(
        "--base-url",
        help="llama-server URL. Defaults to http://127.0.0.1:8080.",
    )
    parser.add_argument(
        "--model",
        help="Model key to record. Defaults to whatever llama-server loaded.",
    )
    parser.add_argument(
        "--models-dir",
        help=(
            "Folder your GGUFs live in, holding one directory per model. "
            "Stored so Sherlock can start llama-server for you when none is "
            "running. Ignored while a server is already listening -- its own "
            "directory was fixed when it launched."
        ),
    )
    parser.add_argument(
        "--server-binary",
        help=(
            "Path to the llama-server executable. Found on PATH when omitted."
        ),
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
        console.print("Configure it with `sherlock-rm setup ai`.")
        # 1, not 2: nothing is broken, the answer is "nothing is stored". Same
        # convention as `sherlock-rm show`, so a script can branch on it.
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
            models_dir=(
                args.models_dir
                if args.models_dir is not None
                else (existing.models_dir if existing is not None else None)
            ),
            server_binary=(
                args.server_binary
                if args.server_binary is not None
                else (existing.server_binary if existing is not None else None)
            ),
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

    interactive = sys.stdin.isatty() if stdin_isatty is None else stdin_isatty

    # Setup starts its own server rather than asking the user to. Stopped again
    # below: this is a configuration step, and startup only indexes models
    # rather than loading any weights, so it costs about a second.
    server = ManagedLlamaServer(provisional)
    try:
        status = await server.ensure_running()
    except LlamaServerError as error:
        # Nothing found. Ask WHERE the models are rather than printing a
        # command to go and run -- that was the instruction-shaped dead end
        # this whole flow exists to remove.
        chosen = (
            _ask_for_models_folder(console=output, error=error)
            if interactive
            else None
        )
        if chosen is None:
            output.print(f"[red]\\[x] {error}[/red]")
            await server.stop()
            return 2
        provisional = provisional.model_copy(update={"models_dir": chosen})
        server = ManagedLlamaServer(provisional)
        try:
            status = await server.ensure_running()
        except LlamaServerError as retry_error:
            output.print(f"[red]\\[x] {retry_error}[/red]")
            await server.stop()
            return 2
    if status.started_by_us:
        output.print(f"[dim]{status.detail}[/dim]")

    provider = LlamaCppProvider(
        provisional,
        api_token=environment.get("LLAMA_API_TOKEN"),
    )
    try:
        models = await provider.list_models()
    except AIProviderError as error:
        output.print(f"[red]\\[x] {error}[/red]")
        return 2
    finally:
        await provider.close()
        await server.stop()

    if not models:
        output.print("[red]\\[x] No models available.[/red]")
        output.print(
            "Point Sherlock at the folder your .gguf files are in:\n"
            "  sherlock-rm setup ai --models-dir <folder>\n"
            "Any layout works -- it searches inside."
        )
        return 2

    models.sort(key=lambda model: (model.display_name.casefold(), model.key))
    selected = _select_model(
        parser=parser,
        models=models,
        requested=args.model,
        existing=existing,
        console=output,
        interactive=interactive,
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
