"""The update check: what counts as newer, and what may replace this install.

Two things here are worth more than the rest. `is_newer` is the guard against
offering someone an older build than the one they are running -- which the CLI
check it grew out of would do, because it asks whether two strings differ. And
`detect_install` is the guard against a forced reinstall landing on a working
tree or inside a Docker image, neither of which can be undone by the person it
happens to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherlock_project.updater import (
    CHECK_INTERVAL_SECONDS,
    UNKNOWN_VERSION,
    cache_path,
    checked_recently,
    detect_install,
    install_command,
    is_newer,
    manual_instruction,
    parse_version,
    record_check,
    run_install,
    updater_available,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v0.2.0", (0, 2, 0)),
        ("0.2.0", (0, 2, 0)),
        ("V1.0", (1, 0)),
        ("  v2.3.4  ", (2, 3, 4)),
        # Anything that is not plain dotted digits has no ordering we can
        # trust, and guessing is how a release candidate installs over a
        # stable build.
        ("v2.0-rc1", None),
        ("release-3", None),
        ("", None),
        ("v", None),
        ("1.2.x", None),
    ],
)
def test_parse_version_reads_only_what_it_can_order(text, expected):
    assert parse_version(text) == expected


@pytest.mark.parametrize(
    ("remote", "installed", "expected"),
    [
        ("v0.2.0", "0.1.0", True),
        ("v1.0.0", "0.9.9", True),
        ("v0.2.1", "0.2.0", True),
        # Equal is not newer, which is the common case on every launch after
        # the first.
        ("v0.1.0", "0.1.0", False),
        # The defect this replaces. `tag[1:] != __version__` answers
        # "different", so a retracted tag or a local build ahead of the last
        # release both offered to install something older.
        ("v0.1.0", "0.2.0", False),
        ("v0.1.0", "0.1.1", False),
        # Unorderable on either side means no offer.
        ("release-3", "0.1.0", False),
        ("v0.2.0", "not-a-version", False),
        # A build that cannot read its own version has no business deciding
        # it is out of date.
        ("v9.9.9", UNKNOWN_VERSION, False),
    ],
)
def test_is_newer_compares_order_not_difference(remote, installed, expected):
    assert is_newer(remote, installed) is expected


def test_shorter_versions_compare_as_lower():
    """`0.2` is older than `0.2.1`, not equal to it.

    Tuple comparison gives this for free, but only because `parse_version`
    returns tuples rather than padding to a fixed length -- worth pinning, as
    a well-meant "normalise to three parts" would make 0.2 and 0.2.0 differ.
    """
    assert is_newer("v0.2.1", "0.2") is True
    assert is_newer("v0.2", "0.2.1") is False


# -- what may replace this install --------------------------------------------


def test_docker_is_detected_from_the_variable_the_image_already_sets():
    """`SHERLOCK_ENV=docker` is set by the Dockerfile and, until this feature,
    read by nothing. An image cannot rewrite itself: the install would vanish
    with the container and the next `docker run` would be the old build."""
    assert detect_install(environ={"SHERLOCK_ENV": "docker"}) == "docker"
    assert install_command("docker", "v0.2.0") is None
    assert "docker build" in manual_instruction("docker", "v0.2.0")


def test_a_checkout_is_detected_by_its_pyproject_and_refused(tmp_path: Path):
    """A source checkout or an editable install IS the working tree. The probe
    is the same one `get_version()` uses for its pyproject fallback, and it is
    load-bearing here for a different reason: a forced reinstall over a tree
    someone is editing replaces the thing they are editing."""
    package = tmp_path / "sherlock_project"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")

    mode = detect_install(environ={}, module_file=str(package / "updater.py"))

    assert mode == "source"
    assert install_command("source", "v0.2.0") is None
    assert "git pull" in manual_instruction("source", "v0.2.0")


def test_an_installed_copy_is_not_mistaken_for_a_checkout(tmp_path: Path):
    """No pyproject two directories up means this is an installed copy, and
    installed copies are the ones that may be replaced."""
    package = tmp_path / "site-packages" / "sherlock_project"
    package.mkdir(parents=True)

    mode = detect_install(environ={}, module_file=str(package / "updater.py"))

    assert mode == "pip"
    assert install_command(mode, "v0.2.0") is not None


@pytest.mark.parametrize(
    ("segment", "expected"),
    [("pipx/venvs", "pipx"), ("uv/tools", "uv"), ("elsewhere", "pip")],
)
def test_the_installer_is_read_off_the_path(tmp_path: Path, segment, expected):
    package = tmp_path / segment / "sherlock-rm" / "sherlock_project"
    package.mkdir(parents=True)

    mode = detect_install(environ={}, module_file=str(package / "updater.py"))

    assert mode == expected


@pytest.mark.parametrize("mode", ["pipx", "uv", "pip"])
def test_every_runnable_command_pins_the_tag(mode):
    """Pinned, not tracking a branch: the release is what was offered, and
    resolving to whatever the default branch holds by the time the download
    starts would install something nobody was shown."""
    command = install_command(mode, "v0.2.0")

    assert command is not None
    assert any("@v0.2.0" in part for part in command)
    # Without --force a resolver that sees a satisfied requirement does
    # nothing and reports success.
    assert any("force" in part for part in command)


def test_pip_installs_with_the_running_interpreter():
    """`sys.executable -m pip`, never a bare `pip`: the bare name may be a
    different environment's, which is how an update lands somewhere other than
    the install being updated."""
    import sys

    command = install_command("pip", "v0.2.0")

    assert command is not None
    assert command[0] == sys.executable


def test_modes_that_cannot_install_report_no_tool():
    assert updater_available("source") is False
    assert updater_available("docker") is False
    assert updater_available("pip") is True


# -- the throttle -------------------------------------------------------------


def test_the_stamp_suppresses_a_second_check_but_not_a_later_one(tmp_path: Path):
    environ = {"SHERLOCK_CACHE": str(tmp_path / "update-check.json")}

    assert checked_recently(environ=environ) is False

    record_check(now=1000.0, environ=environ)

    assert checked_recently(now=1000.0, environ=environ) is True
    assert checked_recently(now=1000.0 + 60, environ=environ) is True
    assert (
        checked_recently(now=1000.0 + CHECK_INTERVAL_SECONDS + 1, environ=environ)
        is False
    )


def test_an_unreadable_stamp_means_check_rather_than_crash(tmp_path: Path):
    """Every failure to read the stamp costs one HTTP request. Every failure to
    HANDLE it costs a UI that will not open, so the fallback is to check."""
    broken = tmp_path / "update-check.json"
    broken.write_text("{not json", encoding="utf-8")

    assert checked_recently(environ={"SHERLOCK_CACHE": str(broken)}) is False


def test_a_stamp_from_the_future_does_not_suppress_forever(tmp_path: Path):
    """A clock that jumped -- or a file copied between machines -- must not
    disable checking until the date catches up."""
    stamp = tmp_path / "update-check.json"
    stamp.write_text(json.dumps({"checked_at": 9_999_999_999.0}), encoding="utf-8")

    assert checked_recently(now=1000.0, environ={"SHERLOCK_CACHE": str(stamp)}) is False


def test_the_cache_path_follows_the_override(tmp_path: Path):
    target = tmp_path / "somewhere.json"

    assert cache_path({"SHERLOCK_CACHE": str(target)}) == target


# -- running the installer ----------------------------------------------------


async def test_a_missing_installer_is_an_exit_code_not_a_traceback():
    """`pipx` absent arrives as OSError from exec, not as a non-zero exit. It
    has to become the same shape as any other failure or the dialog's error
    path never runs."""
    code, last = await run_install(["definitely-not-a-real-binary-xyz"])

    assert code != 0
    assert "could not run" in last


async def test_output_is_streamed_line_by_line_and_the_last_is_returned():
    code, last = await run_install(
        ["python3", "-c", "print('first'); print('second')"],
        on_line=(seen := []).append,
    )

    assert code == 0
    assert seen == ["first", "second"]
    assert last == "second"


async def test_a_failing_installer_reports_its_own_exit_code():
    code, _ = await run_install(["python3", "-c", "raise SystemExit(3)"])

    assert code == 3
