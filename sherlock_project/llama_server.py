"""Start `llama-server` when nothing is already listening.

`--models-dir` is a LAUNCH flag. There is no runtime equivalent -- `POST /props`
in router mode is per-model routing and answers "model name is missing from the
request" to anything else -- so a user cannot choose where their models live
unless something launches the server for them. That is the only reason this
module exists.

The rule it keeps, everywhere: **never touch a server it did not start.**
Somebody running their own llama-server has made a decision, possibly with
flags and a directory quite unlike ours, and adopting it is right while
restarting or stopping it is not. `stop()` therefore no-ops unless `start()`
actually spawned something in this process.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from sherlock_project.ai_config import AISettings

SERVER_EXECUTABLE = "llama-server"
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

    def __init__(self, settings: AISettings) -> None:
        self._settings = settings
        self._process: asyncio.subprocess.Process | None = None

    @property
    def started_by_us(self) -> bool:
        return self._process is not None

    def _command(self, binary: str) -> list[str]:
        host, port = _host_and_port(self._settings.base_url)
        models_dir = str(Path(self._settings.models_dir or "").expanduser())
        return [
            binary,
            "--models-dir", models_dir,
            "--host", host,
            "--port", str(port),
            "--jinja",
            "--reasoning-format", REASONING_FORMAT,
        ]

    async def ensure_running(self) -> ServerStatus:
        """Adopt a running server, or start one from the configured directory.

        Returns WITHOUT AWAITING when no models directory is stored. That is
        not just an optimisation: `run_ai_pipeline` is supposed to get model
        loading under way before the browser opens, and it only wins that race
        because nothing between the task starting and the load yields control.
        A health probe here would hand the loop to the scan, and the two would
        stop overlapping for every user who never configured a directory --
        which is everyone running their own server.
        """
        if not self._settings.models_dir:
            return ServerStatus(
                running=False,
                started_by_us=False,
                detail=(
                    "No models directory is set, so llama-server is left to "
                    "the user. Set one with "
                    "`sherlock setup ai --models-dir <folder>` to have "
                    "Sherlock start it."
                ),
            )

        if await is_listening(self._settings.base_url):
            return ServerStatus(
                running=True,
                started_by_us=False,
                detail="Using the llama-server already listening.",
            )

        models_dir = Path(self._settings.models_dir).expanduser()
        if not models_dir.is_dir():
            raise LlamaServerError(
                f"Models directory does not exist: {models_dir}"
            )

        binary = resolve_binary(self._settings)
        if binary is None:
            raise LlamaServerError(
                f"Could not find {SERVER_EXECUTABLE} on PATH. Install llama.cpp, "
                "or set the path with `sherlock setup ai --server-binary <path>`."
            )

        await self._spawn(binary)
        return ServerStatus(
            running=True,
            started_by_us=True,
            detail=f"Started llama-server on {self._settings.base_url}.",
        )

    async def _spawn(self, binary: str) -> None:
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
                *self._command(binary),
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
                    f"llama-server exited immediately (code {code}). The most "
                    "likely cause is the models directory layout: it needs one "
                    "directory per model, each holding its .gguf file."
                )
            if await is_listening(self._settings.base_url):
                return
            await asyncio.sleep(HEALTH_POLL_SECONDS)

        await self.stop()
        raise LlamaServerError(
            f"llama-server did not answer within {STARTUP_TIMEOUT_SECONDS:.0f}s."
        )

    async def stop(self) -> None:
        """Stop the server, if and only if we are the ones who started it."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            # Already gone between the check above and here.
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            # Awaited so the child is reaped rather than lingering as a zombie
            # holding the port against the next run.
            await process.wait()
