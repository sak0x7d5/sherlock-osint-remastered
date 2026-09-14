"""Finding out whether a newer release exists, and replacing this one with it.

Deliberately not a module that knows anything about screens. The TUI drives it
and draws the result; the functions here answer questions and run a command.

**Three things can go wrong that are not failures.** No release exists yet, the
forge is unreachable, and the installed version cannot be determined. All three
mean "no update to offer" and none of them is worth interrupting anyone over --
an update check that produces an error dialog has cost more than it is worth.
So every entry point here returns a value rather than raising, and the caller
has one thing to test: did I get a Release back.

**Installing is the part that cannot be made universal.** This package is
installed by pipx, by uv, by pip into some virtualenv, from a source checkout,
or baked into a Docker image, and two of those must never be written to:

  - A source checkout or an editable install IS the user's working tree. A
    forced reinstall over it replaces the thing they are editing.
  - A Docker image is immutable; the install would evaporate with the
    container and the next `docker run` would be the old build again.

Both are detected and refused with the command that would actually work, which
is more useful than attempting it and leaving a mess. That refusal is the whole
reason `detect_install` exists rather than the caller just shelling out to pip.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from platformdirs import user_cache_path

from sherlock_project import forge_api_latest_release

# Where the tarball comes from. Derived from the API URL rather than written
# twice, so a fork or a rename moves both halves together -- the constant in
# `__init__.py` already carries a comment about why pointing at upstream would
# be wrong, and a second literal here would be the way that comment stops
# being true.
REPOSITORY_URL = "https://github.com/sak0x7d5/sherlock-osint-remastered"

# How long a check is good for. An update that landed four hours ago can wait
# until the next launch; GitHub's unauthenticated limit is 60 requests an hour
# per address, and a tool that is opened and closed repeatedly should not spend
# that allowance on the same answer.
CHECK_INTERVAL_SECONDS = 6 * 60 * 60

# The version string `get_version()` falls back to when neither the installed
# metadata nor a pyproject is readable. Not a version, and must never be
# compared as one.
UNKNOWN_VERSION = "unknown"

InstallMode = Literal["pipx", "uv", "pip", "source", "docker"]


@dataclass(frozen=True, slots=True)
class Release:
    """A published release, as much of it as is worth showing."""

    tag: str
    version: str
    url: str


def parse_version(text: str) -> tuple[int, ...] | None:
    """A dotted version as numbers, or None if it is not one.

    Returning None rather than raising because every caller's answer to an
    unparseable version is the same: do not claim an update. A tag someone
    pushed as `release-3` or `v2.0-rc1` is not a thing to compare against
    `0.1.0`, and guessing is how a release candidate gets installed over a
    stable build.
    """
    candidate = text.strip()
    if candidate.startswith(("v", "V")):
        candidate = candidate[1:]
    if not candidate:
        return None
    parts = candidate.split(".")
    numbers: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        numbers.append(int(part))
    return tuple(numbers)


def is_newer(remote: str, installed: str) -> bool:
    """Whether `remote` is a later version than `installed`.

    Ordered comparison, not inequality. The check this replaces asked
    `latest_remote_tag[1:] != __version__`, which answers "different" -- so a
    tag that was retracted, or a local build ahead of the last release, or any
    tag not shaped `vX.Y.Z`, all reported an update available and offered to
    install something older than what was already there.

    `unknown` is not a version and loses to everything, including itself: a
    build whose own version cannot be read has no business deciding it is out
    of date.
    """
    if installed == UNKNOWN_VERSION:
        return False
    left = parse_version(remote)
    right = parse_version(installed)
    if left is None or right is None:
        return False
    return left > right


def cache_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the last-checked stamp lives.

    A cache, not config: it is written by the program rather than by a person,
    and losing it costs one extra HTTP request. `SHERLOCK_CACHE` overrides it,
    mirroring `SHERLOCK_CONFIG` and `SHERLOCK_DB` so that a test -- or someone
    keeping a machine spotless -- can point all three somewhere disposable.
    """
    environment = os.environ if environ is None else environ
    override = environment.get("SHERLOCK_CACHE")
    if override:
        return Path(override).expanduser()
    return user_cache_path("sherlock", appauthor=False) / "update-check.json"


def checked_recently(
    *,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Whether a check happened recently enough to skip this one.

    Unreadable, missing or corrupt stamp all mean "no", because the cost of
    being wrong is one HTTP request and the cost of raising is a UI that will
    not open.
    """
    moment = time.time() if now is None else now
    try:
        stamp = json.loads(cache_path(environ).read_text(encoding="utf-8"))
        last = float(stamp["checked_at"])
    except Exception:
        return False
    return 0 <= moment - last < CHECK_INTERVAL_SECONDS


def record_check(
    *,
    now: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Stamp the time of a completed check. Failure to write is not an error."""
    moment = time.time() if now is None else now
    try:
        destination = cache_path(environ)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"checked_at": moment}), encoding="utf-8"
        )
    except Exception:
        return


async def fetch_latest(*, timeout: float = 10.0) -> Release | None:
    """The newest published release, or None for every way of not having one.

    `requests` is synchronous, so it goes off-thread for the same reason the
    scan's own check does: a forge that takes the full timeout to answer must
    not hold the event loop -- here that would be a UI that does not draw until
    it gives up.

    The blanket except is deliberate and matches the scan path. No release
    published yet is a 404, which reaches us as a KeyError on `tag_name`, and
    it is the single most likely outcome on a repository that has never been
    tagged. It is not news.
    """
    import requests

    try:
        response = await asyncio.to_thread(
            requests.get, forge_api_latest_release, timeout=timeout
        )
        payload = json.loads(response.text)
        tag = str(payload["tag_name"])
        version = tag[1:] if tag.startswith(("v", "V")) else tag
        return Release(tag=tag, version=version, url=str(payload["html_url"]))
    except Exception:
        return None


def detect_install(
    *,
    environ: Mapping[str, str] | None = None,
    module_file: str | None = None,
) -> InstallMode:
    """How this copy got here, which decides whether it can replace itself.

    Ordered by how conclusive each signal is, not by how common the mode is.
    Docker first because the image sets a variable saying so and nothing else
    can; the source probe second because a checkout can also be inside a
    virtualenv and would otherwise be mistaken for a pip install and written
    over.
    """
    environment = os.environ if environ is None else environ

    # Set by the Dockerfile, and until now read by nothing at all.
    if environment.get("SHERLOCK_ENV") == "docker":
        return "docker"

    # The same probe `get_version()` uses for its pyproject fallback, and for
    # the same reason: an editable install leaves the real repository two
    # directories up, so finding a pyproject there means this IS the checkout.
    here = Path(module_file or __file__).resolve()
    if (here.parent.parent / "pyproject.toml").exists():
        return "source"

    root = str(here).replace("\\", "/").lower()
    if "/pipx/" in root:
        return "pipx"
    if "/uv/tools/" in root or "/uv/tool/" in root:
        return "uv"
    return "pip"


def install_command(mode: InstallMode, tag: str) -> list[str] | None:
    """The command that replaces this install, or None if nothing should run.

    Pinned to the tag rather than tracking a branch: the release is the thing
    that was offered, and resolving to whatever the default branch holds by the
    time the download starts would install something nobody was shown.

    `--force` is required rather than defensive. The version in the package
    metadata moves with the tag, and without it a resolver that sees a
    satisfied requirement does nothing and reports success.
    """
    target = f"git+{REPOSITORY_URL}@{tag}"
    if mode == "pipx":
        return ["pipx", "install", "--force", target]
    if mode == "uv":
        return ["uv", "tool", "install", "--force", target]
    if mode == "pip":
        return [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--force-reinstall", target,
        ]
    return None


def manual_instruction(mode: InstallMode, tag: str) -> str:
    """What to tell someone whose install cannot update itself.

    Never a flag they are not in a position to type: this is read inside a
    running app, so it names the command for the shell they will go back to.
    """
    if mode == "docker":
        return (
            "This is the Docker image, which cannot rewrite itself. "
            "Rebuild it:\n"
            "  git pull && docker build -t sherlock-osint-remastered ."
        )
    if mode == "source":
        return (
            "This is a source checkout, and updating over it would replace "
            "the tree you are working in. Update it with git:\n"
            "  git pull && pip install ."
        )
    return f"  pipx install --force git+{REPOSITORY_URL}@{tag}"


async def run_install(
    command: Sequence[str],
    *,
    on_line: Callable[[str], None] | None = None,
) -> tuple[int, str]:
    """Run an installer, streaming its output, and report how it ended.

    `create_subprocess_exec` rather than `subprocess.run` for the reason
    `llama_server` gives: this is called from the event loop, and the
    synchronous form would freeze the screen for the minutes a resolve and
    build takes -- which is the exact stretch the caller needs to animate.

    stderr is folded into stdout because pip says useful things on both and the
    caller has one line to show. The last line is returned as well as streamed,
    so a failure has something to quote without the caller keeping its own
    copy.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except OSError as error:
        # A missing pipx or uv lands here rather than as an exit code.
        return 127, f"could not run {command[0]}: {error}"

    last = ""
    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode(errors="replace").strip()
        if not line:
            continue
        last = line
        if on_line is not None:
            on_line(line)
    code = await process.wait()
    return code, last


def updater_available(mode: InstallMode) -> bool:
    """Whether the tool this mode needs is actually on PATH.

    Checked before offering, not after failing: "Update" that dies on
    `FileNotFoundError` is worse than a dialog that explains it cannot.
    """
    if mode == "pipx":
        return shutil.which("pipx") is not None
    if mode == "uv":
        return shutil.which("uv") is not None
    return mode == "pip"
