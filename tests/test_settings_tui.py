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
from textual.widgets import Static

from sherlock_project.ai_config import (
    DEFAULT_AI_CONTEXT_LENGTH,
    DEFAULT_AI_TEMPERATURE,
    DEFAULT_LLAMACPP_BASE_URL,
    AISettings,
    ScanSettings,
    SherlockSettings,
    load_settings,
    save_settings,
)
from sherlock_project.ai_provider import AIModelInfo
from sherlock_project.llama_server import ServerStatus
from sherlock_project.settings import SETTING_FIELDS, SettingField
from sherlock_project.settings_tui import (
    SettingsApp,
    format_context,
    plain_value,
    render_value,
    run_settings,
    thinking_label,
)


def _index_of(key: str) -> int:
    return next(i for i, f in enumerate(SETTING_FIELDS) if f.key == key)


def _seed(path: Path, **scan) -> None:
    save_settings(
        SherlockSettings(
            ai=AISettings(base_url="http://127.0.0.1:8080", model="vendor/m"),
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


def _rendered(app: SettingsApp, selector: str) -> str:
    """The visible text of one widget.

    Textual 8 returns a Content object from render(); `.plain` is the text
    without its styling, which is what these assertions are about.
    """
    return app.query_one(selector, Static).render().plain


def _help_text(app: SettingsApp) -> str:
    return _rendered(app, "#help")


async def test_the_help_line_describes_whatever_row_the_cursor_is_on(
    tmp_path: Path,
):
    """The keyboard's answer to a hover tooltip.

    Textual does have real tooltips, but they only appear on mouse hover, and
    this screen is driven entirely from the keyboard -- the explanation would
    sit behind the one input device nobody here is using.
    """
    path = tmp_path / "config.toml"
    _seed(path)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        assert "llama-server runs whatever GGUF" in _help_text(app)

        for _ in range(_index_of("scan.timeout")):
            await pilot.press("down")
        assert "inconclusive" in _help_text(app)


async def test_toggling_the_browser_changes_the_flag_and_the_explanation(
    tmp_path: Path,
):
    """Two surfaces, one keypress: a terse flag that survives the cursor
    moving away, and the prose for the person deciding right now."""
    path = tmp_path / "config.toml"
    _seed(path, webbrowser=True)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.webbrowser")):
            await pilot.press("down")
        assert "can be trusted" in _help_text(app)

        await pilot.press("left")

        assert app._values["scan.webbrowser"] is False
        assert "reported as absent" in _help_text(app)
        # The flag rides on the row itself, so it is still there once the
        # cursor moves on.
        await pilot.press("down")
        assert "faster; inaccurate" in _rendered(
            app, f"#row-{_index_of('scan.webbrowser')}"
        )


async def test_reset_restores_the_builtin_default_not_the_saved_value(
    tmp_path: Path,
):
    """Deliberate reversal of the earlier behaviour, on the user's call.

    Restoring the SAVED value duplicated Esc, which already discards unsaved
    edits -- so the key did nothing that leaving the screen would not. Restoring
    what the tool ships is the thing no other key offers, and it is the way back
    out of a config someone has edited into a corner.
    """
    path = tmp_path / "config.toml"
    _seed(path, concurrency=75)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("scan.concurrency")):
            await pilot.press("down")
        await pilot.press("left")
        await pilot.press("r")

        assert app._values["scan.concurrency"] == 30
        assert "back to the default (30)" in app._status
        # Reset is an EDIT, not an undo: it leaves the screen dirty so ^S is
        # still what commits it, and Esc still walks away from it.
        assert app.dirty is True

    assert load_settings(path=path, environ={}).scan.concurrency == 75


async def test_reset_says_so_when_a_field_has_nothing_to_restore(tmp_path: Path):
    """Which model is right depends on what the user downloaded, so there is
    no shipped answer -- and blanking the field would be worse than refusing."""
    path = tmp_path / "config.toml"
    _seed(path)
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("ai.model")):
            await pilot.press("down")
        await pilot.press("r")

        assert app._values["ai.model"] == "vendor/m"
        assert "no default" in app._status
        assert app.dirty is False


async def test_reset_restores_the_conventional_endpoint(tmp_path: Path):
    """The schema marks the endpoint required, but "put it back to the usual
    LM Studio address" is still a real thing to want."""
    path = tmp_path / "config.toml"
    save_settings(
        SherlockSettings(
            ai=AISettings(base_url="http://10.0.0.5:9999", model="vendor/m"),
        ),
        path=path,
        environ={},
    )
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        for _ in range(_index_of("ai.base_url")):
            await pilot.press("down")
        await pilot.press("r")

        assert app._values["ai.base_url"] == "http://127.0.0.1:8080"


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


async def test_ai_spinners_step_from_their_default_with_no_ai_stored(
    tmp_path: Path,
):
    """Before `setup ai`, the AI spinners are usable rather than blank.

    Reversal of the earlier behaviour, on purpose. These rows used to hold None
    with nothing stored, so they rendered the literal word `None` between their
    arrows and refused to step -- a value nobody chose, that means nothing, and
    that the arrows could not move. They now start at the build's own default
    and step normally. Absence is still shown, on `ai.model`, which is the only
    AI field with no shipped answer and the only one the user must supply.
    """
    path = tmp_path / "config.toml"
    save_settings(SherlockSettings(), path=path, environ={})
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        assert app._values["ai.temperature"] == DEFAULT_AI_TEMPERATURE
        assert app._values["ai.context_length"] == DEFAULT_AI_CONTEXT_LENGTH
        assert app._values["ai.base_url"] == DEFAULT_LLAMACPP_BASE_URL
        # The one that genuinely has no default still reads as absent.
        assert app._values["ai.model"] is None

        for _ in range(_index_of("ai.temperature")):
            await pilot.press("down")
        await pilot.press("right")

        assert app._values["ai.temperature"] > DEFAULT_AI_TEMPERATURE


def test_render_marks_which_rows_respond_to_arrows():
    spin = next(f for f in SETTING_FIELDS if f.key == "scan.concurrency")
    text = next(f for f in SETTING_FIELDS if f.key == "scan.proxy")

    assert "‹" in render_value(spin, 30)
    assert "‹" not in render_value(text, None)
    assert render_value(text, None) == "not set"


def test_unit_and_zero_label_render_on_both_surfaces():
    """`‹ 5 ›` does not say five of what, and `‹ 0 ›` says the opposite of never.

    Asserted on both surfaces because they are the thing that must not drift:
    the row and the plain listing read the same value out of one place.

    Built on a synthetic field rather than a real one. It used to run against
    `ai.unload_after_minutes`, which the llama.cpp migration removed -- nothing
    in SETTING_FIELDS opts into `unit`/`zero_label` any more. Deleting the test
    with the setting would have left that rendering path live and unwatched
    for whichever setting adopts it next.
    """
    field = SettingField(
        "scan", "example", "example", "spin",
        (0, 5, 30),
        unit="min",
        zero_label="never",
    )

    assert "5 min" in render_value(field, 5)
    assert "never" in render_value(field, 0)
    assert "0" not in render_value(field, 0)
    assert plain_value(field, 30) == "30 min"
    assert plain_value(field, 0) == "never"


def test_units_stay_off_the_rows_that_never_asked_for_one():
    """The unit is opt-in per field; every other spinner renders as it did."""
    concurrency = next(f for f in SETTING_FIELDS if f.key == "scan.concurrency")
    webbrowser = next(f for f in SETTING_FIELDS if f.key == "scan.webbrowser")

    assert plain_value(concurrency, 30) == "30"
    assert plain_value(webbrowser, False) == "off"


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

    Naming the model is now ENOUGH. It used to also require typing an endpoint,
    which was friction for nothing: the endpoint arrives pre-filled with
    llama-server's own address, and anyone running it anywhere else already
    knows to change it.
    """
    path = tmp_path / "config.toml"
    save_settings(SherlockSettings(scan=ScanSettings(concurrency=20)),
                  path=path, environ={})
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        app._values["ai.model"] = "vendor/model"
        await pilot.press("ctrl+s")
        assert "Saved" in app._status

    stored = load_settings(path=path, environ={})
    assert stored.ai is not None
    assert stored.ai.model == "vendor/model"
    assert stored.ai.base_url == DEFAULT_LLAMACPP_BASE_URL
    assert stored.scan.concurrency == 20


async def test_saving_without_a_model_says_only_the_model_is_missing(
    tmp_path: Path,
):
    """The one AI field with no default is the one the message must name.

    The old wording said "AI needs both an endpoint and a model", which sent
    people hunting for a second problem after the endpoint started defaulting.
    """
    path = tmp_path / "config.toml"
    save_settings(SherlockSettings(), path=path, environ={})
    app = SettingsApp(config_path=path)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+s")

        assert "Cannot save" in app._status
        assert "a model" in app._status
        assert "an endpoint" not in app._status

    assert load_settings(path=path, environ={}).ai is None


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


async def test_the_model_picker_starts_a_server_instead_of_asking_you_to(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """The last surface that still told the user to run llama-server.

    It said "Start llama-server, then press ^R" on exactly the first run where
    the list matters most -- handing back a job the rest of the tool had
    already taken over. The picker now starts one itself, and stops it again
    when the screen closes.
    """
    from sherlock_project.tui import settings_pane as pane_module

    started: list[str] = []
    stopped: list[str] = []

    class RecordingServer:
        def __init__(self, settings, **_kwargs) -> None:
            self._settings = settings

        async def ensure_running(self):
            started.append(self._settings.models_dir or "<auto>")
            return ServerStatus(running=True, started_by_us=True, detail="ok")

        async def stop(self) -> None:
            stopped.append("stopped")

    class FakeProvider:
        def __init__(self, _settings, **_kwargs) -> None:
            pass

        async def list_models(self) -> list[AIModelInfo]:
            return [
                AIModelInfo(
                    key="Some-Model-Q4_K_M",
                    display_name="Some-Model",
                    quantization="Q4_K_M",
                    params="8B",
                    loaded=False,
                    max_context_length=8192,
                    reasoning_options=(),
                )
            ]

        async def close(self) -> None:
            return None

    monkeypatch.setattr(pane_module, "ManagedLlamaServer", RecordingServer)
    monkeypatch.setattr(pane_module, "LlamaCppProvider", FakeProvider)

    screen = pane_module.ModelPickerScreen(
        "http://127.0.0.1:8080",
        current=None,
        models_dir=str(tmp_path),
    )
    app = SettingsApp(config_path=tmp_path / "config.toml")

    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        status = screen.query_one("#picker-status", Static).render().plain

        # Never an instruction to go and run something.
        assert "Start llama-server" not in status
        assert "1 available" in status
        assert started == [str(tmp_path)]

        screen.action_cancel()
        await pilot.pause()

    assert stopped == ["stopped"]
