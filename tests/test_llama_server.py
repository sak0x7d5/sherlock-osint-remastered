"""Starting llama-server, and the rule about never touching someone else's."""

from pathlib import Path

import pytest

from sherlock_project import llama_server
from sherlock_project.ai_config import AISettings
from sherlock_project.llama_server import (
    LlamaServerError,
    ManagedLlamaServer,
    resolve_binary,
)


def _settings(**overrides) -> AISettings:
    base = {
        "base_url": "http://127.0.0.1:8080",
        "model": "Qwen3-8B-GGUF",
    }
    return AISettings(**{**base, **overrides})


async def test_no_models_dir_returns_without_awaiting_anything(
    monkeypatch: pytest.MonkeyPatch,
):
    """Load-order guarantee, not an optimisation.

    `run_ai_pipeline` only gets model loading started before the browser opens
    because nothing between the task starting and the load yields control. A
    health probe here would hand the loop to the scan and stop the two
    overlapping for everyone running their own server.
    """
    def explode(*_args, **_kwargs):
        raise AssertionError("must not probe when it cannot start anything")

    monkeypatch.setattr(llama_server, "is_listening", explode)
    status = await ManagedLlamaServer(_settings()).ensure_running()

    assert status.running is False
    assert status.started_by_us is False
    assert "models directory" in status.detail


async def test_a_server_already_listening_is_adopted_not_restarted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Someone else's server was started with their flags, for their reasons."""
    async def listening(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(llama_server, "is_listening", listening)
    server = ManagedLlamaServer(_settings(models_dir=str(tmp_path)))
    status = await server.ensure_running()

    assert status.running is True
    assert status.started_by_us is False
    assert server.started_by_us is False


async def test_stop_is_a_no_op_for_a_server_we_did_not_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def listening(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(llama_server, "is_listening", listening)
    server = ManagedLlamaServer(_settings(models_dir=str(tmp_path)))
    await server.ensure_running()

    # Nothing to assert against beyond "does not raise and does not spawn":
    # there is no process handle precisely because we adopted one.
    await server.stop()
    assert server.started_by_us is False


async def test_a_missing_models_directory_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def not_listening(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(llama_server, "is_listening", not_listening)
    server = ManagedLlamaServer(
        _settings(models_dir=str(tmp_path / "nope"))
    )

    with pytest.raises(LlamaServerError, match="does not exist"):
        await server.ensure_running()


async def test_a_missing_binary_names_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def not_listening(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(llama_server, "is_listening", not_listening)
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: None)
    server = ManagedLlamaServer(_settings(models_dir=str(tmp_path)))

    with pytest.raises(LlamaServerError, match="--server-binary"):
        await server.ensure_running()


def test_configured_binary_wins_over_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: "/on/path")
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")

    assert resolve_binary(_settings(server_binary=str(binary))) == str(binary)
    # A configured path that is not a file resolves to nothing rather than
    # silently falling back, or the error would name the wrong binary.
    assert resolve_binary(_settings(server_binary=str(tmp_path / "absent"))) is None


def test_the_launch_command_carries_the_load_bearing_flags(tmp_path: Path):
    """`--reasoning-format deepseek` is not decoration.

    It is what routes native thinking into `message.reasoning_content` instead
    of leaving it inline in `content`, which the structured-output path depends
    on. `--jinja` is what makes chat_template_kwargs work at all.
    """
    server = ManagedLlamaServer(
        _settings(models_dir=str(tmp_path), base_url="http://127.0.0.1:9999")
    )
    command = server._command("llama-server")

    assert command[0] == "llama-server"
    assert "--models-dir" in command
    assert command[command.index("--port") + 1] == "9999"
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert "--jinja" in command
