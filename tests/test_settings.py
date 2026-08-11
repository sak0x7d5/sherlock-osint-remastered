"""Precedence and provenance for one run's effective settings.

The rule is flag > config > default, and the provenance half is not a
diagnostic nicety: a flag is visible in the command someone typed, a stored
default is not, so the scan has to say which values came from the file.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from sherlock_project.ai_config import (
    AIConfigError,
    AISettings,
    OutputSettings,
    ScanSettings,
    SherlockSettings,
    load_settings,
    save_ai_settings,
    save_settings,
    try_load_settings,
)
from sherlock_project.settings import resolve_runtime_settings, resolve_value


def _resolve(stored: SherlockSettings, **flags):
    defaults = {
        "concurrency": None,
        "timeout": None,
        "proxy": None,
        "nsfw": None,
        "no_color": None,
        "verbose": None,
    }
    return resolve_runtime_settings(stored=stored, **{**defaults, **flags})


def test_flag_beats_config_beats_default():
    stored = SherlockSettings(scan=ScanSettings(concurrency=50, timeout=15))

    resolved = _resolve(stored, concurrency=100)

    assert (resolved.concurrency.value, resolved.concurrency.source) == (
        100,
        "flag",
    )
    assert (resolved.timeout.value, resolved.timeout.source) == (15, "config")
    assert resolved.proxy.source == "default"


def test_a_flag_matching_the_default_still_counts_as_a_flag():
    """Otherwise `-c 30` against a stored 50 would silently lose to config.

    This is why every resolver-fed flag defaults to None in argparse rather
    than to its real default: absence has to be distinguishable from a value
    that happens to equal the default.
    """
    stored = SherlockSettings(scan=ScanSettings(concurrency=50))

    resolved = _resolve(stored, concurrency=30)

    assert resolved.concurrency.value == 30
    assert resolved.concurrency.source == "flag"


def test_only_config_sourced_settings_are_reported():
    """Flags are already on screen; defaults are what everyone else gets."""
    stored = SherlockSettings(
        scan=ScanSettings(concurrency=50, nsfw=True),
        output=OutputSettings(verbose=True),
    )

    reported = _resolve(stored, concurrency=10).from_config()

    assert reported == {"nsfw sites": True, "verbose": True}
    assert "concurrency" not in reported


def test_nothing_is_reported_when_nothing_was_stored():
    assert _resolve(SherlockSettings()).from_config() == {}


def test_no_color_flag_inverts_into_the_stored_positive():
    """The flag says --no-color; the file says color = true."""
    stored = SherlockSettings(output=OutputSettings(color=True))

    assert _resolve(stored, no_color=True).color.value is False
    assert _resolve(stored).color.value is True
    assert _resolve(SherlockSettings(output=OutputSettings(color=False))).color.source == "config"


def test_resolve_value_treats_false_as_an_answer():
    """`if flag:` would drop --nsfw=False; absence is None, not falsiness."""
    assert resolve_value(flag=False, config=True, default=False).source == "flag"
    assert resolve_value(flag=None, config=True, default=False).source == "config"


def test_saving_ai_settings_preserves_scan_preferences(tmp_path: Path):
    """Changing model must not silently discard someone's other settings."""
    path = tmp_path / "config.toml"
    save_settings(
        SherlockSettings(scan=ScanSettings(concurrency=77, timeout=12)),
        path=path,
        environ={},
    )

    save_ai_settings(
        AISettings(base_url="http://localhost:1234", model="vendor/model"),
        path=path,
        environ={},
    )
    reloaded = load_settings(path=path, environ={})

    assert reloaded.scan.concurrency == 77
    assert reloaded.scan.timeout == 12
    assert reloaded.ai is not None
    assert reloaded.ai.model == "vendor/model"


def test_scan_preferences_are_storable_without_any_ai_config(tmp_path: Path):
    """[ai] used to be required, which made scan settings depend on a model."""
    path = tmp_path / "config.toml"

    save_settings(
        SherlockSettings(scan=ScanSettings(concurrency=8)),
        path=path,
        environ={},
    )
    reloaded = load_settings(path=path, environ={})

    assert reloaded.ai is None
    assert reloaded.scan.concurrency == 8


def test_an_unreadable_config_falls_back_to_defaults(tmp_path: Path):
    """Stored preferences must never stop a scan from running."""
    path = tmp_path / "config.toml"
    path.write_text("this is not toml at all {{{", encoding="utf-8")

    stored = try_load_settings(path=path, environ={})

    assert stored.scan.concurrency == ScanSettings().concurrency
    assert stored.ai is None


def test_a_config_from_the_future_is_refused_with_an_explanation(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("version = 99\n", encoding="utf-8")

    with pytest.raises(AIConfigError, match="newer version"):
        load_settings(path=path, environ={})


@pytest.mark.parametrize("value", [0, -1])
def test_stored_concurrency_cannot_hang_the_scan(value: int):
    """Semaphore(0) does not raise, it blocks every acquire forever.

    The CLI already rejects it; the stored value has to be rejected in the
    schema or config becomes a way around that validator.
    """
    with pytest.raises(ValidationError):
        ScanSettings(concurrency=value)
