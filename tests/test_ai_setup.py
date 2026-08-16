from io import StringIO
from pathlib import Path
from typing import ClassVar

import pytest
from rich.console import Console

from sherlock_project import ai_setup
from sherlock_project.ai_config import (
    DEFAULT_LLAMACPP_BASE_URL,
    AISettings,
    load_ai_settings,
    save_ai_settings,
)
from sherlock_project.ai_provider import (
    AIModelInfo,
    AIProviderUnavailableError,
)


def _model(
    key: str,
    *,
    reasoning: tuple[str, ...] = (),
) -> AIModelInfo:
    return AIModelInfo(
        key=key,
        display_name=key.title(),
        quantization="Q4_K_M",
        params="8B",
        loaded=False,
        max_context_length=32768,
        reasoning_options=reasoning,
    )


class FakeSetupProvider:
    # Set on the class by tests and read through type(self), so class-level is
    # the intent here rather than an accidental shared instance default.
    models: ClassVar[list[AIModelInfo]] = []
    error: Exception | None = None
    settings_seen = None
    close_calls = 0

    def __init__(self, settings, **_kwargs) -> None:
        type(self).settings_seen = settings

    async def list_models(self) -> list[AIModelInfo]:
        if type(self).error is not None:
            raise type(self).error
        return type(self).models

    async def close(self) -> None:
        type(self).close_calls += 1


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch: pytest.MonkeyPatch):
    FakeSetupProvider.models = []
    FakeSetupProvider.error = None
    FakeSetupProvider.settings_seen = None
    FakeSetupProvider.close_calls = 0
    monkeypatch.setattr(ai_setup, "LlamaCppProvider", FakeSetupProvider)


def _console() -> tuple[Console, StringIO]:
    output = StringIO()
    return (
        Console(
            file=output,
            force_terminal=False,
            no_color=True,
            color_system=None,
            width=160,
        ),
        output,
    )


@pytest.mark.asyncio
async def test_noninteractive_setup_saves_selected_compatible_model(
    tmp_path: Path,
):
    FakeSetupProvider.models = [
        _model("reasoning/model", reasoning=("off", "on")),
        _model("plain/model"),
    ]
    path = tmp_path / "config.toml"
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        [
            "--base-url",
            "http://localhost:8000",
            "--model",
            "reasoning/model",
            "--temperature",
            "0.2",
            "--no-color",
        ],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=False,
    )

    assert result == 0
    settings = load_ai_settings(path=path, environ={})
    assert settings.model == "reasoning/model"
    assert settings.base_url == "http://localhost:8000"
    assert settings.temperature == 0.2
    assert "AI configured" in output.getvalue()
    assert FakeSetupProvider.close_calls == 1


@pytest.mark.asyncio
async def test_interactive_setup_accepts_a_thinking_only_choice_with_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Choosing an always-thinking model is allowed, and says what it costs.

    It used to be refused and the prompt re-asked, which made "the only model I
    have" a dead end for roughly a quarter of a typical library.
    """
    FakeSetupProvider.models = [
        _model("aaa-thinking/model", reasoning=("on",)),
        _model("zzz-plain/model"),
    ]
    monkeypatch.setattr(
        ai_setup.IntPrompt,
        "ask",
        lambda *_args, **_kwargs: 1,
    )
    path = tmp_path / "config.toml"
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        ["--base-url", "http://localhost:8000", "--no-color"],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=True,
    )
    rendered = output.getvalue()

    assert result == 0
    assert load_ai_settings(path=path, environ={}).model == "aaa-thinking/model"
    assert "Model loaded by llama-server" in rendered
    assert "always thinks natively" in rendered
    # The warning has to say what it costs, not just that something is unusual.
    assert "quality is likely to be lower" in rendered


@pytest.mark.asyncio
async def test_interactive_setup_defaults_to_a_reasoning_off_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Allowed is not recommended: the preselected entry still avoids them."""
    FakeSetupProvider.models = [
        _model("aaa-thinking/model", reasoning=("on",)),
        _model("zzz-plain/model"),
    ]
    defaults: list[object] = []

    def capture(*_args, **kwargs):
        defaults.append(kwargs["default"])
        return kwargs["default"]

    monkeypatch.setattr(ai_setup.IntPrompt, "ask", capture)
    path = tmp_path / "config.toml"
    console, _ = _console()

    result = await ai_setup.run_ai_setup(
        ["--base-url", "http://localhost:8000", "--no-color"],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=True,
    )

    assert result == 0
    assert defaults == [2]
    assert load_ai_settings(path=path, environ={}).model == "zzz-plain/model"


@pytest.mark.asyncio
async def test_noninteractive_setup_saves_a_thinking_only_model_with_a_warning(
    tmp_path: Path,
):
    """--model naming an always-thinking model is honoured, loudly."""
    FakeSetupProvider.models = [_model("thinking/model", reasoning=("on",))]
    path = tmp_path / "config.toml"
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        [
            "--base-url",
            "http://localhost:8000",
            "--model",
            "thinking/model",
            "--no-color",
        ],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=False,
    )

    assert result == 0
    assert load_ai_settings(path=path, environ={}).model == "thinking/model"
    assert "always thinks natively" in output.getvalue()


@pytest.mark.asyncio
async def test_setup_still_rejects_a_model_that_is_not_downloaded(tmp_path: Path):
    """Relaxing the thinking rule must not relax the existence check."""
    FakeSetupProvider.models = [_model("thinking/model", reasoning=("on",))]

    with pytest.raises(SystemExit):
        await ai_setup.run_ai_setup(
            [
                "--base-url",
                "http://localhost:8000",
                "--model",
                "absent/model",
            ],
            environ={},
            config_path=tmp_path / "config.toml",
            stdin_isatty=False,
        )

    assert not (tmp_path / "config.toml").exists()


@pytest.mark.asyncio
async def test_setup_provider_failure_preserves_existing_config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("existing", encoding="utf-8")
    FakeSetupProvider.error = AIProviderUnavailableError("server unavailable")
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        ["--base-url", "http://localhost:8000", "--model", "model"],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=False,
    )

    assert result == 2
    assert path.read_text(encoding="utf-8") == "existing"
    assert "server unavailable" in output.getvalue()


@pytest.mark.asyncio
async def test_setup_with_no_models_does_not_write_config(tmp_path: Path):
    console, output = _console()
    path = tmp_path / "config.toml"

    result = await ai_setup.run_ai_setup(
        ["--base-url", "http://localhost:8000", "--model", "model"],
        environ={},
        config_path=path,
        console=console,
        stdin_isatty=False,
    )

    assert result == 2
    assert not path.exists()
    assert "no model loaded" in output.getvalue()


def test_setup_base_url_precedence():
    """Explicit, then environment, then stored, then the default.

    There used to be a fifth source patched in here: `_lms_server_url`, which
    shelled out to `lms server status --json` to learn the port LM Studio had
    chosen. llama.cpp publishes nothing equivalent -- the port is whatever
    `--port` was passed -- so the chain ends at llama-server's own 8080.
    """
    existing = type("Existing", (), {"base_url": "http://localhost:7000"})()

    assert ai_setup.discover_setup_base_url(
        "http://localhost:6000",
        existing=existing,  # type: ignore[arg-type]
        environ={"LLAMA_SERVER_BASE_URL": "http://localhost:5000"},
    ) == "http://localhost:6000"
    assert ai_setup.discover_setup_base_url(
        None,
        existing=existing,  # type: ignore[arg-type]
        environ={"LLAMA_SERVER_BASE_URL": "http://localhost:5000"},
    ) == "http://localhost:5000"
    assert ai_setup.discover_setup_base_url(
        None,
        existing=existing,  # type: ignore[arg-type]
        environ={},
    ) == "http://localhost:7000"
    assert ai_setup.discover_setup_base_url(
        None,
        existing=None,
        environ={},
    ) == DEFAULT_LLAMACPP_BASE_URL


@pytest.mark.asyncio
async def test_show_prints_stored_settings_without_touching_the_provider(
    tmp_path: Path,
):
    """Reading your own settings must not require the server to be running.

    The wizard is otherwise the only way to see which model is configured, and
    it needs a live model list to get that far.
    """
    FakeSetupProvider.error = AIProviderUnavailableError("server is down")
    path = tmp_path / "config.toml"
    save_ai_settings(
        AISettings(
            base_url="http://localhost:9999",
            model="vendor/large",
            temperature=0.3,
            context_length=32768,
        ),
        path=path,
        environ={},
    )
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        ["--show", "--no-color"],
        environ={},
        config_path=path,
        console=console,
    )
    text = output.getvalue()

    assert result == 0
    assert FakeSetupProvider.settings_seen is None
    assert FakeSetupProvider.close_calls == 0
    assert "vendor/large" in text
    assert "http://localhost:9999" in text
    assert "0.3" in text
    assert "32768" in text
    # The path is the point: it resolves per-OS and is shown nowhere else.
    assert str(path) in text


@pytest.mark.asyncio
async def test_show_reports_an_unconfigured_install_without_failing_hard(
    tmp_path: Path,
):
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        ["--show", "--no-color"],
        environ={},
        config_path=tmp_path / "missing.toml",
        console=console,
    )
    text = output.getvalue()

    # 1 is "nothing stored", matching `sherlock show`; 2 is reserved for
    # something being wrong.
    assert result == 1
    assert "not configured" in text
    assert "sherlock setup ai" in text
    assert FakeSetupProvider.close_calls == 0


@pytest.mark.asyncio
async def test_show_flags_an_endpoint_coming_from_the_environment(
    tmp_path: Path,
):
    """The printed endpoint is the effective one, so say when it is overridden."""
    path = tmp_path / "config.toml"
    save_ai_settings(
        AISettings(base_url="http://localhost:8080", model="vendor/large"),
        path=path,
        environ={},
    )
    console, output = _console()

    result = await ai_setup.run_ai_setup(
        ["--show", "--no-color"],
        environ={"LLAMA_SERVER_BASE_URL": "http://elsewhere:4321"},
        config_path=path,
        console=console,
    )
    text = output.getvalue()

    assert result == 0
    assert "http://elsewhere:4321" in text
    assert "LLAMA_SERVER_BASE_URL" in text


@pytest.mark.asyncio
async def test_show_rejects_options_that_would_change_the_configuration(
    tmp_path: Path,
):
    with pytest.raises(SystemExit):
        await ai_setup.run_ai_setup(
            ["--show", "--model", "vendor/other"],
            environ={},
            config_path=tmp_path / "config.toml",
            console=_console()[0],
        )


@pytest.mark.asyncio
async def test_setup_survives_a_terminal_that_cannot_answer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """A prompt with nothing behind it must not raise a traceback.

    isatty() is not a reliable interactivity test on Windows: NUL is a
    character device, so `sherlock setup ai < NUL` -- explicitly "I have no
    keyboard" -- reports True and skips the non-interactive guard. Reaching the
    prompt anyway used to end in an unhandled EOFError out of rich.
    """
    FakeSetupProvider.models = [_model("plain/model")]
    path = tmp_path / "config.toml"
    console, _ = _console()

    def no_input(*_args, **_kwargs):
        raise EOFError("EOF when reading a line")

    original = ai_setup.IntPrompt.ask
    ai_setup.IntPrompt.ask = staticmethod(no_input)
    try:
        with pytest.raises(SystemExit) as exit_info:
            await ai_setup.run_ai_setup(
                ["--base-url", "http://localhost:8000", "--no-color"],
                environ={},
                config_path=path,
                console=console,
                stdin_isatty=True,  # the lie NUL tells on Windows
            )
    finally:
        ai_setup.IntPrompt.ask = original

    assert exit_info.value.code == 2
    assert not path.exists()
    # argparse writes its own errors to stderr, not through the console.
    assert "--model is required" in capsys.readouterr().err
