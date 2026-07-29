from pathlib import Path

import pytest

from sherlock_project.ai_config import (
    AIConfigError,
    AISettings,
    ai_config_path,
    load_ai_settings,
    save_ai_settings,
)


def test_config_round_trip_and_environment_url_override(tmp_path: Path):
    path = tmp_path / "config.toml"
    original = AISettings(
        base_url="http://localhost:1234",
        model="example/model",
        temperature=0.1,
        context_length=8192,
    )

    saved_to = save_ai_settings(original, path=path, environ={})
    loaded = load_ai_settings(
        path=path,
        environ={"LM_STUDIO_BASE_URL": "http://localhost:8000/"},
    )

    assert saved_to == path
    assert loaded.model == "example/model"
    assert loaded.base_url == "http://localhost:8000"
    assert loaded.temperature == 0.1
    serialized = path.read_text(encoding="utf-8")
    assert "version = 1" in serialized
    assert "LM_API_TOKEN" not in serialized


def test_config_path_honors_explicit_environment_override(tmp_path: Path):
    configured = tmp_path / "custom.toml"

    assert ai_config_path({"SHERLOCK_CONFIG": str(configured)}) == configured


def test_missing_config_has_actionable_error(tmp_path: Path):
    with pytest.raises(AIConfigError, match="sherlock setup ai"):
        load_ai_settings(path=tmp_path / "missing.toml", environ={})


@pytest.mark.parametrize(
    "content",
    [
        "not toml",
        'version = 1\n[ai]\nbase_url = "invalid"\nmodel = "x"\n',
        'version = 2\n[ai]\nbase_url = "http://localhost"\nmodel = "x"\n',
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
        AISettings(base_url="http://localhost:1234", model="example/model"),
        path=path,
        environ={},
    )

    with pytest.raises(AIConfigError, match="LM_STUDIO_BASE_URL"):
        load_ai_settings(
            path=path,
            environ={"LM_STUDIO_BASE_URL": "not-a-url"},
        )
