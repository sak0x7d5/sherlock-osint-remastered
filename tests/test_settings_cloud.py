"""The SETTINGS tab with a hosted provider: switching, the key, the picker."""

from pathlib import Path

import keyring
import keyring.backends.fail
from textual.widgets import Static

from sherlock_project.ai_config import (
    DEFAULT_GEMINI_BASE_URL,
    AISettings,
    SherlockSettings,
    load_settings,
    save_settings,
)
from sherlock_project.ai_provider import AIModelInfo, AIProviderAuthenticationError
from sherlock_project.ai_secrets import KEYRING_SERVICE
from sherlock_project.settings import SETTING_FIELDS, apply_values, field_values
from sherlock_project.settings_tui import SettingsApp
from sherlock_project.tui import settings_pane as pane_module
from sherlock_project.tui.confirm_screen import ConfirmScreen
from sherlock_project.tui.settings_pane import (
    CloudModelPickerScreen,
    TextEditScreen,
)

SECRET = "AIza-test-key-0123456789"


def _index_of(key: str) -> int:
    return next(i for i, f in enumerate(SETTING_FIELDS) if f.key == key)


def _seed_local(path: Path) -> None:
    save_settings(
        SherlockSettings(
            ai=AISettings(base_url="http://127.0.0.1:8080", model="vendor/m.gguf")
        ),
        path=path,
        environ={},
    )


def _seed_gemini(path: Path) -> None:
    save_settings(
        SherlockSettings(
            ai=AISettings(
                provider="gemini",
                base_url=DEFAULT_GEMINI_BASE_URL,
                model="gemini-2.5-flash",
            )
        ),
        path=path,
        environ={},
    )


def _status(app: SettingsApp) -> str:
    return app.query_one("#status", Static).render().plain


async def _go_to(pilot, key: str) -> None:
    for _ in range(_index_of(key)):
        await pilot.press("down")


def _model(key: str) -> AIModelInfo:
    return AIModelInfo(
        key=key,
        display_name=key,
        quantization=None,
        params=None,
        loaded=True,
        max_context_length=None,
        reasoning_options=(),
    )


# -- settings model ---------------------------------------------------------


def test_the_key_row_can_never_reach_the_config_model(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    stored = load_settings(path=path, environ={})
    values = field_values(stored)
    assert values["ai.api_key"] is None
    values["ai.api_key"] = SECRET
    updated = apply_values(stored, values)
    assert SECRET not in updated.model_dump_json()


def test_switching_provider_without_a_model_cannot_be_saved(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed_local(path)
    stored = load_settings(path=path, environ={})
    values = field_values(stored)
    values["ai.provider"] = "gemini"
    values["ai.model"] = None
    try:
        apply_values(stored, values)
    except ValueError as error:
        assert "model" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a provider switch saved without a model")


# -- switching provider -----------------------------------------------------


async def test_switching_to_gemini_asks_first_and_cancel_changes_nothing(
    tmp_path: Path,
):
    path = tmp_path / "config.toml"
    _seed_local(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        pane = app.query_one(pane_module.SettingsPane)
        await _go_to(pilot, "ai.provider")
        await pilot.press("right")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert pane._values["ai.provider"] == "llamacpp"
        assert "nothing changed" in _status(app)


async def test_confirming_gemini_sets_its_endpoint_and_pace(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed_local(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        pane = app.query_one(pane_module.SettingsPane)
        await _go_to(pilot, "ai.provider")
        await pilot.press("right")
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert pane._values["ai.provider"] == "gemini"
        assert pane._values["ai.base_url"] == DEFAULT_GEMINI_BASE_URL
        assert pane._values["ai.requests_per_minute"] == 10
        # The local GGUF name means nothing to Gemini.
        assert pane._values["ai.model"] is None

        await pilot.press("ctrl+s")
        await pilot.pause()
        assert "needs a model" in _status(app)
        assert load_settings(path=path, environ={}).ai.provider == "llamacpp"


async def test_local_only_rows_refuse_edits_on_gemini(tmp_path: Path):
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        await _go_to(pilot, "ai.models_dir")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, TextEditScreen)
        assert "not used by Gemini" in _status(app)


# -- the key ------------------------------------------------------------------


async def test_an_entered_key_goes_to_the_keychain_and_never_to_the_file(
    tmp_path: Path, memory_keychain
):
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        await _go_to(pilot, "ai.api_key")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, TextEditScreen)
        assert app.screen.query_one("#edit").password is True
        await pilot.press(*SECRET)
        await pilot.press("enter")
        await pilot.pause()

        assert memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] == SECRET
        assert "Key saved" in _status(app)
        row = app.query_one(f"#row-{_index_of('ai.api_key')}", Static).render().plain
        assert "set ·" in row
        assert SECRET not in row

        # Saving the rest of the settings still never writes the key.
        await _go_to(pilot, "ai.temperature")
        await pilot.press("right")
        await pilot.press("ctrl+s")
        await pilot.pause()
    assert SECRET not in path.read_text(encoding="utf-8")


async def test_an_empty_key_removes_the_stored_one_after_asking(
    tmp_path: Path, memory_keychain
):
    memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] = SECRET
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        await _go_to(pilot, "ai.api_key")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")  # empty
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.click("#confirm-yes")
        await pilot.pause()
        assert (KEYRING_SERVICE, "gemini") not in memory_keychain.passwords
        assert "removed" in _status(app)


async def test_with_no_keychain_the_row_points_at_the_variable(tmp_path: Path):
    keyring.set_keyring(keyring.backends.fail.Keyring())
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        await _go_to(pilot, "ai.api_key")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, TextEditScreen)
        assert "GEMINI_API_KEY" in _status(app)


# -- the hosted model picker -----------------------------------------------


class FakeCloudProvider:
    error: Exception | None = None

    def __init__(self, settings, **_kwargs) -> None:
        self.settings = settings

    async def list_models(self):
        if type(self).error is not None:
            raise type(self).error
        return [_model("gemini-2.5-flash"), _model("text-embedding-004")]

    async def close(self) -> None:
        return None


async def test_the_model_row_on_gemini_opens_the_hosted_picker(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(pane_module, "OpenAICompatibleProvider", FakeCloudProvider)
    FakeCloudProvider.error = None
    path = tmp_path / "config.toml"
    _seed_gemini(path)
    app = SettingsApp(config_path=path)
    async with app.run_test() as pilot:
        pane = app.query_one(pane_module.SettingsPane)
        pane._values["ai.model"] = None
        await _go_to(pilot, "ai.model")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, CloudModelPickerScreen)
        status = screen.query_one("#picker-status", Static).render().plain
        assert "Key works. 1 models" in status
        await pilot.press("enter")
        await pilot.pause()
        assert pane._values["ai.model"] == "gemini-2.5-flash"


async def test_the_hosted_picker_reports_a_rejected_key(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pane_module, "OpenAICompatibleProvider", FakeCloudProvider)
    FakeCloudProvider.error = AIProviderAuthenticationError("rejected the API key")
    screen = CloudModelPickerScreen(
        AISettings(provider="gemini", base_url=DEFAULT_GEMINI_BASE_URL, model="x")
    )
    app = SettingsApp(config_path=tmp_path / "config.toml")
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        status = screen.query_one("#picker-status", Static).render().plain
        assert "Key problem" in status
    FakeCloudProvider.error = None
