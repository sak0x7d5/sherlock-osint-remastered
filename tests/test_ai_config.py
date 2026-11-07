from pathlib import Path

import pytest

from sift_osint.ai_config import (
    CONFIG_VERSION,
    AIConfigError,
    AIConfigNotFound,
    AISettings,
    SiftSettings,
    ai_config_path,
    load_ai_settings,
    load_settings_or_default,
    save_ai_settings,
)


def test_config_round_trip_and_environment_url_override(tmp_path: Path):
    path = tmp_path / "config.toml"
    original = AISettings(
        base_url="http://localhost:8080",
        model="example/model",
        temperature=0.1,
        context_length=8192,
    )

    saved_to = save_ai_settings(original, path=path, environ={})
    loaded = load_ai_settings(
        path=path,
        environ={"LLAMA_SERVER_BASE_URL": "http://localhost:8000/"},
    )

    assert saved_to == path
    assert loaded.model == "example/model"
    assert loaded.base_url == "http://localhost:8000"
    assert loaded.temperature == 0.1
    serialized = path.read_text(encoding="utf-8")
    assert f"version = {CONFIG_VERSION}" in serialized
    assert "LM_API_TOKEN" not in serialized


def test_config_path_honors_explicit_environment_override(tmp_path: Path):
    configured = tmp_path / "custom.toml"

    assert ai_config_path({"SIFT_CONFIG": str(configured)}) == configured


def test_config_path_keeps_using_pre_rename_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "sift_osint.ai_config.user_config_path",
        lambda app, **_: tmp_path / "cfg" / app,
    )
    legacy = tmp_path / "cfg" / "sherlock" / "config.toml"
    legacy.parent.mkdir(parents=True)
    legacy.touch()

    assert ai_config_path({}) == legacy
    assert ai_config_path({"SHERLOCK_CONFIG": str(tmp_path / "x")}) == tmp_path / "x"

    new = tmp_path / "cfg" / "sift-osint" / "config.toml"
    new.parent.mkdir(parents=True)
    new.touch()

    assert ai_config_path({}) == new


def test_saved_theme_from_before_the_rename_still_loads():
    from sift_osint.ai_config import DEFAULT_THEME, OutputSettings

    assert DEFAULT_THEME == "sift"
    assert OutputSettings(theme="sherlock").theme == DEFAULT_THEME
    assert OutputSettings(theme="nord").theme == "nord"


def test_missing_config_has_actionable_error(tmp_path: Path):
    with pytest.raises(AIConfigError, match="sift-osint setup ai"):
        load_ai_settings(path=tmp_path / "missing.toml", environ={})


@pytest.mark.parametrize(
    "content",
    [
        "not toml",
        'version = 1\n[ai]\nbase_url = "invalid"\nmodel = "x"\n',
        # A version this build does not know, i.e. written by a newer Sherlock.
        'version = 99\n[ai]\nbase_url = "http://localhost"\nmodel = "x"\n',
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(AIConfigError, match="configuration"):
        load_ai_settings(path=path, environ={})


def test_invalid_environment_url_is_rejected(tmp_path: Path):
    path = tmp_path / "config.toml"
    save_ai_settings(
        AISettings(base_url="http://localhost:8080", model="example/model"),
        path=path,
        environ={},
    )

    with pytest.raises(AIConfigError, match="LLAMA_SERVER_BASE_URL"):
        load_ai_settings(
            path=path,
            environ={"LLAMA_SERVER_BASE_URL": "not-a-url"},
        )


def test_absent_config_is_not_reported_as_a_read_failure(tmp_path: Path):
    """A fresh install has no config file, and that is not an error.

    This is the most-seen path in the tool: every first run reaches it. The
    loader used to hand back "AI is not configured. Run `sift-osint setup ai`
    first." for a file that had simply never been written, and the scan
    printed it as "Stored settings could not be read" -- warning about lost
    preferences that never existed, and about a model a plain scan does not
    use. Absent means "no stored preferences", which the defaults already are.
    """
    settings, error = load_settings_or_default(
        path=tmp_path / "missing.toml", environ={}
    )

    assert error is None
    assert settings == SiftSettings()


def test_unreadable_config_still_reports_why(tmp_path: Path):
    """The silence fixed above must not swallow a file that IS broken.

    A config written by a newer build, or corrupted, changes what the run does
    and has to say so -- that is the case this loader was added for, and it is
    the one thing distinguishing it from `try_load_settings`.
    """
    path = tmp_path / "config.toml"
    path.write_text("not toml === [[[", encoding="utf-8")

    settings, error = load_settings_or_default(path=path, environ={})

    assert settings == SiftSettings()
    assert error is not None
    assert str(path) in error


def test_missing_config_keeps_raising_for_the_ai_specific_loader(tmp_path: Path):
    """`--ai` must still refuse before scanning when nothing is configured.

    The new exception type is a subclass precisely so this path is unchanged:
    for the AI section specifically, an absent file really does mean "not
    configured", and the run cannot proceed.
    """
    with pytest.raises(AIConfigNotFound, match="sift-osint setup ai"):
        load_ai_settings(path=tmp_path / "missing.toml", environ={})

    assert issubclass(AIConfigNotFound, AIConfigError)
