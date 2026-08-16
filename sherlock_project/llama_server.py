"""Make a llama-server exist, without the user having to think about one.

Running a model server is llama.cpp's problem, not the user's. Nobody should
have to learn a command, a flag set, or a directory convention to get an
optional feature working, so this module owns all of it: find the binary, find
the models wherever they already are, start the server, stop it afterwards.

Two llama.cpp details are hidden here on purpose, because both leak
implementation at the user:

- `--models-dir` demands `<dir>/<repo>/*.gguf` EXACTLY. One level too high
  serves zero models, exits 0, and reports it only in the server's own stdout,
  which from the API is indistinguishable from an empty machine. Unknowable
  from outside, so a generated `--models-preset` is used instead -- it takes
  absolute paths, so any layout works and the rule never surfaces.
- `--models-dir` is launch-only. `POST /props` in router mode is per-model
  routing and rejects anything else, so a directory cannot be changed on a
  running server. Choosing one therefore means owning the process.

The rule it keeps, everywhere: **never touch a server it did not start.**
Somebody running their own llama-server has made a decision, possibly with
flags and models quite unlike ours, and adopting it is right while restarting
or stopping it is not. `stop()` therefore no-ops unless this process spawned
something.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from sherlock_project.ai_config import AISettings, ai_config_path

SERVER_EXECUTABLE = "llama-server"
MODEL_SUFFIX = ".gguf"
# How deep to walk a models root. Deep enough for publisher/repo/file and the
# HuggingFace cache's blobs layout, shallow enough that pointing this at a home
# directory by mistake does not turn into a full disk crawl.
MAX_SCAN_DEPTH = 6
# GGUFs that are not standalone chat models. Multimodal projectors and vocoders
# ship beside the model they belong to and cannot answer a chat request, so
# offering them in the picker only produces a confusing failure later.
_NON_CHAT_MARKERS = ("mmproj", "vocoder", "-embed", "embedding")
# How long to wait for a freshly spawned server to answer /health. A cold start
# reads the model index and can be slow on a spinning disk or a big directory;
# it does not load any weights yet, so this is not the model-load wait.
STARTUP_TIMEOUT_SECONDS = 90.0
HEALTH_POLL_SECONDS = 0.5
# Given to `--reasoning-format`. `deepseek` is what routes native thinking into
# `message.reasoning_content` instead of leaving it inline in `content`, which
# the whole structured-output path depends on. Do NOT pass `none` here.
REASONING_FORMAT = "deepseek"


class LlamaServerError(RuntimeError):
    """Sherlock could not start llama-server, with a reason worth printing."""


@dataclass(frozen=True, slots=True)
class ServerStatus:
    running: bool
    started_by_us: bool
    detail: str


def _is_chat_model(path: Path) -> bool:
    name = path.name.casefold()
    return not any(marker in name for marker in _NON_CHAT_MARKERS)


def discover_models(roots: Sequence[Path]) -> dict[str, Path]:
    """Every usable GGUF under these roots, keyed by a stable display name.

    Recursive and layout-agnostic ON PURPOSE. `--models-dir` demands
    `<dir>/<repo>/*.gguf` exactly, which is a rule about llama.cpp's internals
    that a user has no reason to know and no way to discover -- getting it
    wrong serves zero models and says so only in the server's own stdout. A
    generated preset accepts absolute paths, so nothing about where a file sits
    has to reach the user at all.

    Keyed on the filename stem rather than the parent directory: it is stable
    across rescans, unique in practice, and carries the quantization, which is
    the thing people actually want to tell two copies of one model apart by.
    """
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        root_depth = len(root.parts)
        # os.walk with PRUNING, not rglob. rglob walks the entire tree and
        # leaves depth to be filtered afterwards, which is no help at all: the
        # cost is the walk. Pointing this at a home directory -- which the
        # folder browser does by default -- then crawls every file the user
        # owns. Measured as 92s in the test suite before this loop replaced it.
        for current, directories, filenames in os.walk(root):
            here = Path(current)
            if len(here.parts) - root_depth >= MAX_SCAN_DEPTH:
                directories.clear()
            # Big, uninteresting, and common directly under a home directory.
            directories[:] = [
                name for name in directories if not name.startswith((".", "$"))
            ]
            for filename in sorted(filenames):
                if not filename.endswith(MODEL_SUFFIX):
                    continue
                path = here / filename
                if not _is_chat_model(path):
                    continue
                # First root wins, so an explicitly configured directory beats
                # an auto-detected one holding the same file.
                found.setdefault(path.stem, path)
    return found


def write_preset(models: Mapping[str, Path], destination: Path) -> Path:
    """Write the INI llama-server reads with `--models-preset`.

    `jinja` and `reasoning-format` are per-model here rather than global
    process flags, because in router mode each model is launched as its own
    instance from this preset -- flags passed to the parent are not inherited.
    Both are load-bearing: jinja is what makes `chat_template_kwargs` work at
    all, and deepseek is what routes thinking into `reasoning_content` instead
    of leaving it inline where it would break the JSON parse.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for name, path in sorted(models.items()):
        sections.append(
            f"[{name}]\n"
            "jinja = 1\n"
            f"reasoning-format = {REASONING_FORMAT}\n"
            f"model = {path.as_posix()}\n"
        )
    destination.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    return destination


def resolve_binary(settings: AISettings) -> str | None:
    """The configured executable, else whatever is on PATH."""
    configured = settings.server_binary
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which(SERVER_EXECUTABLE)


def _host_and_port(base_url: str) -> tuple[str, int]:
    parsed = urlsplit(base_url)
    return parsed.hostname or "127.0.0.1", parsed.port or 8080


async def is_listening(base_url: str, *, timeout: float = 2.0) -> bool:
    """Whether anything answers on the endpoint.

    A 503 counts as listening: that is llama-server's "still loading", which
    means the port is taken and starting a second one would only fail to bind.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            response = await client.get("/health")
    except httpx.RequestError:
        return False
    return response.status_code < 500 or response.status_code == 503


class ManagedLlamaServer:
    """Owns a llama-server process, but only one it started itself."""

    def __init__(
        self,
        settings: AISettings,
        *,
        preset_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._process: asyncio.subprocess.Process | None = None
        # Beside the config rather than in the models folder: that folder may
        # be read-only, on a network share, or shared with another tool, and
        # this file is Sherlock's bookkeeping rather than the user's data.
        self._preset_path = preset_path or (
            ai_config_path().parent / "llama-models.ini"
        )

    @property
    def started_by_us(self) -> bool:
        return self._process is not None

    def _command(self, binary: str, preset: Path) -> list[str]:
        host, port = _host_and_port(self._settings.base_url)
        return [
            binary,
            "--models-preset", str(preset),
            "--host", host,
            "--port", str(port),
        ]

    async def ensure_running(self) -> ServerStatus:
        """Make a server exist, without the user having to think about one.

        Order matters and each step is cheap before the expensive one:

        1. Something already listening is adopted untouched.
        2. Otherwise every GGUF under the configured folder -- or the usual
           places, when nothing is configured -- is found recursively and
           written into a preset file, so layout never matters.
        3. llama-server is started from that preset.

        The first check is the only await on the "nothing to do" path, and
        `run_ai_pipeline` depends on that: it gets model loading under way
        before the browser opens only because nothing yields first. Discovery
        touches the filesystem, so it runs off-thread.
        """
        if await is_listening(self._settings.base_url):
            return ServerStatus(
                running=True,
                started_by_us=False,
                detail="Using the llama-server already listening.",
            )

        binary = resolve_binary(self._settings)
        if binary is None:
            raise LlamaServerError(
                f"Could not find {SERVER_EXECUTABLE}. Install llama.cpp and "
                "make sure it is on your PATH, or point Sherlock at it with "
                "`sherlock setup ai --server-binary <path>`."
            )

        configured = self._settings.models_dir
        if not configured:
            raise LlamaServerError(
                "No models folder is set. Choose one in the model picker, or "
                "pass --models-dir <folder>."
            )
        root = Path(configured).expanduser()
        if not root.is_dir():
            raise LlamaServerError(f"Models folder does not exist: {configured}")

        models = await asyncio.to_thread(discover_models, [root])
        if not models:
            raise LlamaServerError(
                f"No .gguf models found in {root} -- any layout works, it "
                "searches inside, so this folder has none."
            )

        preset = write_preset(models, self._preset_path)
        await self._spawn(binary, preset)
        return ServerStatus(
            running=True,
            started_by_us=True,
            detail=(
                f"Started llama-server with {len(models)} model"
                f"{'' if len(models) == 1 else 's'}."
            ),
        )

    async def _spawn(self, binary: str, preset: Path) -> None:
        # Its own process group, so a Ctrl-C in the terminal reaches Sherlock
        # and lets the ordered shutdown stop the child, rather than killing the
        # server first and leaving the scan talking to a corpse.
        extra: dict[str, object] = {}
        if sys.platform == "win32":
            extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            extra["start_new_session"] = True

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command(binary, preset),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                env=os.environ.copy(),
                **extra,  # type: ignore[arg-type]
            )
        except OSError as error:
            raise LlamaServerError(
                f"Could not start {binary}: {error}"
            ) from error

        deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            if self._process.returncode is not None:
                code = self._process.returncode
                self._process = None
                raise LlamaServerError(
                    f"llama-server exited immediately (code {code}). Its "
                    f"preset is at {preset} if you need to look."
                )
            if await is_listening(self._settings.base_url):
                return
            await asyncio.sleep(HEALTH_POLL_SECONDS)

        await self.stop()
        raise LlamaServerError(
            f"llama-server did not answer within {STARTUP_TIMEOUT_SECONDS:.0f}s."
        )

    async def stop(self) -> None:
        """Stop the server AND its model instances, if we started them.

        The tree matters, not just the process. In router mode llama-server is
        a supervisor: each model it loads runs as its OWN llama-server child,
        which is what `status.args` in `/v1/models` is showing. Terminating
        only the parent leaves those children resident -- measured once as a
        multi-gigabyte model still in memory after Sherlock had exited and
        reported a clean shutdown, which is exactly the leak LM Studio's idle
        TTL used to cover.
        """
        process = self._process
        if process is None:
            return
        self._process = None
        if process.returncode is not None:
            return

        if sys.platform == "win32":
            # No process-group signal reaches a Windows child tree, so ask the
            # OS to walk it. /T is the whole point of this call.
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(process.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            # start_new_session put the parent in its own group, so the
            # children are in it too and one signal reaches all of them.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            # Awaited so the child is reaped rather than lingering as a zombie
            # holding the port against the next run.
            await process.wait()
