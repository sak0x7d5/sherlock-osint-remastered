from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sherlock_project.ai_config import load_ai_settings
from sherlock_project.ai_provider import (
    AIModelInfo,
    AIProviderUnavailableError,
)
from sherlock_project import ai_setup


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
    models: list[AIModelInfo] = []
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
    monkeypatch.setattr(ai_setup, "LMStudioProvider", FakeSetupProvider)


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
async def test_interactive_setup_rejects_thinking_only_choice_then_selects_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    FakeSetupProvider.models = [
        _model("aaa-thinking/model", reasoning=("on",)),
        _model("zzz-plain/model"),
    ]
    selections = iter([1, 2])
    monkeypatch.setattr(
        ai_setup.IntPrompt,
        "ask",
        lambda *_args, **_kwargs: next(selections),
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

    assert result == 0
    assert load_ai_settings(path=path, environ={}).model == "zzz-plain/model"
    rendered = output.getvalue()
    assert "Downloaded LM Studio models" in rendered
    assert "requires native thinking" in rendered


@pytest.mark.asyncio
async def test_setup_rejects_thinking_only_model(tmp_path: Path):
    FakeSetupProvider.models = [_model("thinking/model", reasoning=("on",))]

    with pytest.raises(SystemExit):
        await ai_setup.run_ai_setup(
            [
                "--base-url",
                "http://localhost:8000",
                "--model",
                "thinking/model",
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
    assert "no downloaded LLMs" in output.getvalue()


def test_setup_base_url_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ai_setup, "_lms_server_url", lambda: "http://localhost:9000")
    existing = type("Existing", (), {"base_url": "http://localhost:7000"})()

    assert ai_setup.discover_setup_base_url(
        "http://localhost:6000",
        existing=existing,  # type: ignore[arg-type]
        environ={"LM_STUDIO_BASE_URL": "http://localhost:5000"},
    ) == "http://localhost:6000"
    assert ai_setup.discover_setup_base_url(
        None,
        existing=existing,  # type: ignore[arg-type]
        environ={"LM_STUDIO_BASE_URL": "http://localhost:5000"},
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
    ) == "http://localhost:9000"
