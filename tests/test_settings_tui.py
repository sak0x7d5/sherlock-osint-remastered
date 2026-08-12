"""The settings screen, driven by real keypresses through Textual's pilot.

The field rules live in settings.py and are tested there without a terminal.
What is worth testing here is the wiring: that a keypress reaches the right
value, that saving writes what the screen shows, and that the app refuses to
draw where drawing is wrong.
"""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from sherlock_project.ai_config import (
    AISettings,
    ScanSettings,
    SherlockSettings,
    load_settings,
    save_settings,
)
from sherlock_project.ai_provider import AIModelInfo
from sherlock_project.settings import SETTING_FIELDS
from sherlock_project.settings_tui import (
    SettingsApp,
    format_context,
    render_value,
    run_settings,
    thinking_label,
)


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


async def test_the_editor_is_launched_from_inside_the_running_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """`sherlock settings` is reached from inside main()'s event loop.

    App.run() calls asyncio.run() internally, so the synchronous form dies with
    "asyncio.run() cannot be called from a running event loop" and leaves an
    un-awaited coroutine behind. Every other test drove the app through
    run_test(), which is already async -- so the one line that launches it for
    real was the only line not covered.
    """
    path = tmp_path / "config.toml"
    _seed(path, concurrency=50)
    launched: list[str] = []

    async def fake_run_async(self, *_args, **_kwargs):
        launched.append("async")

    def forbidden_run(self, *_args, **_kwargs):
        raise AssertionError("App.run() cannot be used inside a running loop")

    monkeypatch.setattr(SettingsApp, "run_async", fake_run_async)
    monkeypatch.setattr(SettingsApp, "run", forbidden_run)

    code = await run_settings([], config_path=path, interactive=True)

    assert code == 0
    assert launched == ["async"]


async def test_ai_settings_can_be_created_when_no_section_exists_yet(
    tmp_path: Path,
):
    """The state of every install that has not run `setup ai`.

    These edits used to be dropped in silence while the save still reported
    "Saved to ..." -- the worst failure available, because nothing on screen
    said the AI half had not been written.
    """
    path = tmp_path / "config.toml"
    save_settings(SherlockSettings(scan=ScanSettings(concurrency=20)),
                  path=path, environ={})
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        app._values["ai.model"] = "vendor/model"
        await pilot.press("ctrl+s")
        assert "Cannot save" in app._status
        assert load_settings(path=path, environ={}).ai is None

        app._values["ai.base_url"] = "http://127.0.0.1:1234"
        await pilot.press("ctrl+s")
        assert "Saved" in app._status

    stored = load_settings(path=path, environ={})
    assert stored.ai is not None
    assert stored.ai.model == "vendor/model"
    assert stored.scan.concurrency == 20


async def test_a_stale_message_is_cleared_when_the_cursor_moves(tmp_path: Path):
    """"Saved to ..." lingering under an unsaved marker is a contradiction."""
    path = tmp_path / "config.toml"
    _seed(path, concurrency=30)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+s")
        assert "Saved" in app._status
        await pilot.press("down")
        assert app._status == ""


def test_context_windows_read_as_people_say_them():
    """131072 is a number to decode; 128K is comparable at a glance."""
    assert format_context(131072) == "128K"
    assert format_context(32768) == "32K"
    assert format_context(1048576) == "1024K"
    assert format_context(None) == "?"
    assert format_context(1000) == "1000"


def test_thinking_column_distinguishes_three_states():
    """"Cannot be stopped" and "has none at all" are different facts.

    Pass one is written for reasoning-off, so this column is the difference
    between a model that fits and one that is merely allowed.
    """
    def model(reasoning):
        return AIModelInfo(
            key="k", display_name="k", quantization=None, params=None,
            loaded=False, max_context_length=None, reasoning_options=reasoning,
        )

    assert thinking_label(model(())) == "none"
    assert thinking_label(model(("off", "on"))) == "optional"
    assert thinking_label(model(("on",))) == "always"
