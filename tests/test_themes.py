"""Colour themes: what the setting stores, and what switching it repaints.

The palettes themselves are taste and are not asserted beyond the one property
that defines a family -- AMOLED means black. What is worth pinning is the
plumbing: a stored name always opens SOMETHING, the default looks exactly as it
did before themes existed, a preview does not outlive the screen it was tried
on, and the text built in Python follows the stylesheet instead of staying cyan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from textual.app import App

from sherlock_project.ai_config import (
    DEFAULT_THEME,
    THEME_CHOICES,
    AISettings,
    OutputSettings,
    SherlockSettings,
    load_settings,
    save_settings,
)
from sherlock_project.result import QueryStatus
from sherlock_project.settings import SETTING_FIELDS, field_values
from sherlock_project.settings_tui import SettingsApp
from sherlock_project.tui.app import SherlockUI
from sherlock_project.tui.theme import (
    STATUS_ORDER,
    progress_bar,
    status_cell,
    tone,
)
from sherlock_project.tui.themes import (
    SHERLOCK_THEMES,
    register_themes,
    resolve_theme,
)


def _seed(path: Path, theme: str | None = None) -> None:
    """A config that can be saved from the screen: one with a model in it."""
    save_settings(
        SherlockSettings(
            ai=AISettings(base_url="http://127.0.0.1:8080", model="vendor/m"),
            output=OutputSettings(theme=theme or DEFAULT_THEME),
        ),
        path=path,
        environ={},
    )


def _index_of(key: str) -> int:
    return next(i for i, f in enumerate(SETTING_FIELDS) if f.key == key)


# -- the setting --------------------------------------------------------------


def test_the_default_theme_is_the_original_look():
    assert OutputSettings().theme == DEFAULT_THEME == "sherlock"
    assert field_values(SherlockSettings())["output.theme"] == "sherlock"


def test_a_config_written_before_themes_existed_still_loads(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("version = 1\n[output]\ncolor = true\nverbose = false\n")

    assert load_settings(path=path, environ={}).output.theme == DEFAULT_THEME


def test_a_theme_this_build_does_not_know_still_loads(tmp_path: Path):
    """Refusing the file would cost every other setting in it over a colour."""
    path = tmp_path / "config.toml"
    path.write_text('version = 1\n[output]\ntheme = "from-the-future"\n')

    assert load_settings(path=path, environ={}).output.theme == "from-the-future"


def test_every_offered_theme_exists():
    """The settings row offers names; each must be one the app can apply.

    The names live in ai_config, which must not import Textual, and the
    palettes live in tui/themes.py, which must. This is what holds them
    together.
    """
    app = App()
    register_themes(app)

    missing = [name for name in THEME_CHOICES if name not in app.available_themes]
    assert missing == []
    assert THEME_CHOICES[0] == DEFAULT_THEME


def test_an_unknown_name_falls_back_to_the_default():
    app = App()
    register_themes(app)

    assert resolve_theme(app, "from-the-future") == DEFAULT_THEME
    assert resolve_theme(app, None) == DEFAULT_THEME
    assert resolve_theme(app, "amoled") == "amoled"


def test_amoled_themes_are_black():
    amoled = [theme for theme in SHERLOCK_THEMES if theme.name.startswith("amoled")]

    assert len(amoled) == 3
    for theme in amoled:
        assert theme.background == theme.surface == "#000000", theme.name


def test_reading_settings_does_not_import_textual():
    """Every CLI run loads the config. The theme names are in it, so they must
    not drag the UI toolkit in with them."""
    probe = (
        "import sys, sherlock_project.settings; "
        "print('textual' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.stdout.strip() == "False"


# -- tone ---------------------------------------------------------------------


def test_tone_outside_an_app_is_the_classic_ansi():
    assert tone("bold accent") == "bold cyan"
    assert tone("secondary") == "cyan"
    assert tone("dim warning") == "dim yellow"
    # ANSI names from older call sites read as their role.
    assert tone("green") == "green"
    # Everything else passes through.
    assert tone("dim italic") == "dim italic"
    assert tone("bold #123456") == "bold #123456"


async def test_tone_resolves_against_a_chosen_theme(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed(path, theme="amoled")
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == "amoled"
        assert tone("bold accent") == "bold #00E5FF"
        assert tone("bold cyan") == "bold #00E5FF"
        assert tone("secondary") == "#FF2BD6"
        assert tone("success") == "#00FF9C"


async def test_the_default_theme_keeps_its_ansi_colours(tmp_path: Path):
    """Mapped through textual-dark's palette, cyan would turn amber -- and the
    wordmark, the progress bar and every spinner would change colour for
    people who never touched the setting."""
    path = tmp_path / "config.toml"
    _seed(path)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == DEFAULT_THEME
        assert tone("bold accent") == "bold cyan"
        assert str(progress_bar(1, 2, 10).spans[0].style) == "bold cyan"


async def test_a_theme_changes_colours_but_never_the_glyphs_or_words(
    tmp_path: Path,
):
    """Colour is never the only signal -- whatever the palette."""
    path = tmp_path / "config.toml"
    _seed(path)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        await pilot.pause()
        before = [status_cell(status).plain for status in STATUS_ORDER]
        found_before = str(status_cell(QueryStatus.CLAIMED).spans[0].style)
        app.theme = "amoled-violet"
        await pilot.pause()
        after = [status_cell(status).plain for status in STATUS_ORDER]
        found_after = str(status_cell(QueryStatus.CLAIMED).spans[0].style)

    assert before == after
    assert found_before == "bold green"
    assert found_after == "bold #69F0AE"


# -- the settings row -----------------------------------------------------------


async def test_stepping_the_theme_row_previews_it_and_saving_keeps_it(
    tmp_path: Path,
):
    path = tmp_path / "config.toml"
    _seed(path)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("output.theme")):
            await pilot.press("down")
        await pilot.press("right")
        await pilot.pause()

        assert app._values["output.theme"] == THEME_CHOICES[1]
        assert app.theme == THEME_CHOICES[1]

        await pilot.press("ctrl+s")
        assert app.dirty is False

    assert load_settings(path=path, environ={}).output.theme == THEME_CHOICES[1]


async def test_leaving_without_saving_puts_the_saved_theme_back(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed(path, theme="amoled")
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("output.theme")):
            await pilot.press("down")
        await pilot.press("right")
        await pilot.pause()
        assert app.theme == "amoled-violet"

        # First escape warns about the unsaved edit, the second discards it.
        await pilot.press("escape")
        await pilot.press("escape")
        await pilot.pause()
        assert app.theme == "amoled"

    assert load_settings(path=path, environ={}).output.theme == "amoled"


# -- the unified UI -------------------------------------------------------------


async def test_the_ui_opens_in_the_stored_theme(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.toml"
    _seed(path, theme="amoled-mono")
    monkeypatch.setenv("SHERLOCK_CONFIG", str(path))
    app = SherlockUI()

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == "amoled-mono"


async def test_the_ui_opens_on_an_unknown_theme_in_the_default(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "config.toml"
    path.write_text('version = 1\n[output]\ntheme = "from-the-future"\n')
    monkeypatch.setenv("SHERLOCK_CONFIG", str(path))
    app = SherlockUI()

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.theme == DEFAULT_THEME


async def test_switching_theme_repaints_the_wordmark(tmp_path: Path, monkeypatch):
    """The appbar is drawn once and on resize. Without the theme hook it kept
    the old palette's wordmark under the new palette's chrome."""
    path = tmp_path / "config.toml"
    _seed(path)
    monkeypatch.setenv("SHERLOCK_CONFIG", str(path))
    app = SherlockUI()

    def wordmark_colour() -> str:
        rendered = app.query_one("#appbar").render()
        return str(rendered.spans[0].style)

    async with app.run_test() as pilot:
        await pilot.pause()
        before = wordmark_colour()
        app.theme = "amoled"
        for _ in range(5):
            await pilot.pause()
        after = wordmark_colour()

    assert before != after
    # Textual's Content reports the colour as rgb(), not as the hex it was given.
    assert "rgb(0,229,255)" in after


async def test_the_command_palette_has_no_second_theme_picker(
    tmp_path: Path, monkeypatch
):
    """Textual's own picker saves nothing, so its choice was undone by the next
    tab switch. The settings row is the one place a theme is chosen."""
    path = tmp_path / "config.toml"
    _seed(path)
    monkeypatch.setenv("SHERLOCK_CONFIG", str(path))
    app = SherlockUI()

    async with app.run_test() as pilot:
        await pilot.pause()
        titles = [command.title for command in app.get_system_commands(app.screen)]

    assert "Theme" not in titles
    assert "Quit" in titles
