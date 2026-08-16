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


async def test_no_models_dir_falls_back_to_the_usual_places(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """An unconfigured install should still find models somebody else put there.

    Asking the user where their models are is a question the tool can usually
    answer itself, and every question is a step between "installed" and
    "working".
    """
    monkeypatch.setattr(
        llama_server, "default_model_roots", lambda: [tmp_path / "auto"]
    )
    server = ManagedLlamaServer(_settings())

    assert server.search_roots() == [tmp_path / "auto"]


def test_discovery_is_recursive_and_ignores_non_chat_ggufs(tmp_path: Path):
    """Layout must never reach the user, and projectors are not chat models.

    `--models-dir` demands <dir>/<repo>/*.gguf exactly and fails silently
    otherwise, which is unknowable from outside. Searching recursively and
    writing absolute paths into a preset removes the rule entirely.
    """
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "Deep-Model-Q4_K_M.gguf").write_text("", encoding="utf-8")
    (tmp_path / "Flat-Model-Q8_0.gguf").write_text("", encoding="utf-8")
    # Ships beside a multimodal model and cannot answer a chat request.
    (tmp_path / "mmproj-Deep-Model-F16.gguf").write_text("", encoding="utf-8")

    found = llama_server.discover_models([tmp_path])

    assert sorted(found) == ["Deep-Model-Q4_K_M", "Flat-Model-Q8_0"]


def test_preset_carries_the_flags_each_instance_needs(tmp_path: Path):
    """Per-model, not process-wide: router instances do not inherit our flags.

    jinja is what makes chat_template_kwargs work at all, and deepseek is what
    routes thinking into reasoning_content instead of leaving it inline where
    it breaks the JSON parse.
    """
    destination = tmp_path / "cfg" / "llama-models.ini"
    llama_server.write_preset(
        {"Some-Model-Q4_K_M": tmp_path / "weird place" / "Some-Model-Q4_K_M.gguf"},
        destination,
    )
    written = destination.read_text(encoding="utf-8")

    assert "[Some-Model-Q4_K_M]" in written
    assert "jinja = 1" in written
    assert "reasoning-format = deepseek" in written
    assert "weird place/Some-Model-Q4_K_M.gguf" in written


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


async def test_a_missing_models_folder_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def not_listening(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(llama_server, "is_listening", not_listening)
    monkeypatch.setattr(llama_server.shutil, "which", lambda _n: "llama-server")
    server = ManagedLlamaServer(_settings(models_dir=str(tmp_path / "nope")))

    with pytest.raises(LlamaServerError, match="does not exist"):
        await server.ensure_running()


async def test_a_folder_with_no_models_says_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Never "go start a server" -- the tool does that. Only "where are they"."""
    async def not_listening(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr(llama_server, "is_listening", not_listening)
    monkeypatch.setattr(llama_server.shutil, "which", lambda _n: "llama-server")
    server = ManagedLlamaServer(_settings(models_dir=str(tmp_path)))

    with pytest.raises(LlamaServerError, match="No .gguf models found"):
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


def test_the_server_launches_from_a_generated_preset(tmp_path: Path):
    """Preset, not directory: the layout rule must never reach the user."""
    server = ManagedLlamaServer(
        _settings(models_dir=str(tmp_path), base_url="http://127.0.0.1:9999")
    )
    preset = tmp_path / "llama-models.ini"
    command = server._command("llama-server", preset)

    assert command[0] == "llama-server"
    assert command[command.index("--models-preset") + 1] == str(preset)
    assert command[command.index("--port") + 1] == "9999"
    # NOT --models-dir: that one imposes a directory layout on the user.
    assert "--models-dir" not in command
