"""The settings screen, driven by real keypresses through Textual's pilot.

The field rules live in settings.py and are tested there without a terminal.
What is worth testing here is the wiring: that a keypress reaches the right
value, that saving writes what the screen shows, and that the app refuses to
draw where drawing is wrong.
"""

from io import StringIO
from pathlib import Path

from rich.console import Console

from sherlock_project.ai_config import (
    AISettings,
    ScanSettings,
    SherlockSettings,
    load_settings,
    save_settings,
)
from sherlock_project.settings import SETTING_FIELDS
from sherlock_project.settings_tui import SettingsApp, render_value, run_settings


def _index_of(key: str) -> int:
    return next(i for i, f in enumerate(SETTING_FIELDS) if f.key == key)


def _seed(path: Path, **scan) -> None:
    save_settings(
        SherlockSettings(
            ai=AISettings(base_url="http://127.0.0.1:1234", model="vendor/m"),
            scan=ScanSettings(**scan),
        ),
        path=path,
        environ={},
    )


async def test_arrow_keys_change_a_bounded_value_and_saving_writes_it(
    tmp_path: Path,
):
    path = tmp_path / "config.toml"
    _seed(path, concurrency=30)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        await pilot.press("right")
        assert app._values["scan.concurrency"] == 50
        assert app.dirty is True

        await pilot.press("ctrl+s")
        assert app.dirty is False

    assert load_settings(path=path, environ={}).scan.concurrency == 50


async def test_stepping_stops_at_the_ends_instead_of_wrapping(tmp_path: Path):
    """Arrowing past the maximum onto the minimum is how people end up
    scanning at concurrency 1 when they meant 100."""
    path = tmp_path / "config.toml"
    _seed(path, concurrency=100)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        for _ in range(5):
            await pilot.press("right")
        assert app._values["scan.concurrency"] == 100


async def test_a_value_the_spinner_does_not_offer_steps_to_a_neighbour(
    tmp_path: Path,
):
    """A hand-edited 77 is not wrong, so it must not be rejected or reset."""
    path = tmp_path / "config.toml"
    _seed(path, concurrency=77)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        await pilot.press("left")
        assert app._values["scan.concurrency"] == 75


async def test_escape_warns_before_discarding_unsaved_changes(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed(path, concurrency=30)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        await pilot.press("right")
        await pilot.press("escape")

        assert app.is_running is True
        assert "Unsaved" in app._status

    assert load_settings(path=path, environ={}).scan.concurrency == 30


async def test_reset_restores_the_saved_value_not_the_builtin_default(
    tmp_path: Path,
):
    path = tmp_path / "config.toml"
    _seed(path, concurrency=75)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        await pilot.press("left")
        await pilot.press("r")

        assert app._values["scan.concurrency"] == 75


async def test_toggles_flip_and_persist(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed(path, nsfw=False)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.nsfw")):
            await pilot.press("down")
        await pilot.press("right")
        await pilot.press("ctrl+s")

    assert load_settings(path=path, environ={}).scan.nsfw is True


async def test_an_unset_ai_field_says_so_instead_of_stepping(tmp_path: Path):
    """With no [ai] stored there is nothing to step, and silence would read
    as a broken key."""
    path = tmp_path / "config.toml"
    save_settings(SherlockSettings(), path=path, environ={})
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("ai.temperature")):
            await pilot.press("down")
        await pilot.press("right")

        assert app._values["ai.temperature"] is None
        assert "unset" in app._status


def test_render_marks_which_rows_respond_to_arrows():
    spin = next(f for f in SETTING_FIELDS if f.key == "scan.concurrency")
    text = next(f for f in SETTING_FIELDS if f.key == "scan.proxy")

    assert "‹" in render_value(spin, 30)
    assert "‹" not in render_value(text, None)
    assert render_value(text, None) == "not set"


async def test_settings_refuses_to_draw_without_a_terminal(tmp_path: Path):
    """A full-screen app that takes over a CI log is worse than no app."""
    path = tmp_path / "config.toml"
    _seed(path, concurrency=50)
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, no_color=True,
                      color_system=None, width=100)

    code = await run_settings(
        [], config_path=path, console=console, interactive=False
    )
    printed = stream.getvalue()

    assert code == 0
    assert "concurrency" in printed and "50" in printed
    assert "No terminal to draw on" in printed


async def test_show_prints_without_opening_the_editor(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed(path, nsfw=True)
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, no_color=True,
                      color_system=None, width=100)

    code = await run_settings(
        ["--show"], config_path=path, console=console, interactive=True
    )
    printed = stream.getvalue()

    assert code == 0
    # "on", not "True": the same setting must not read differently per surface.
    assert "on" in printed
    assert str(path) in printed
