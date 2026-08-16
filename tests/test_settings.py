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
from sherlock_project.settings import (
    NO_DEFAULT,
    SETTING_FIELDS,
    field_default,
    field_description,
    field_note,
    resolve_runtime_settings,
    resolve_value,
)


def _resolve(stored: SherlockSettings, **flags):
    defaults = {
        "concurrency": None,
        "timeout": None,
        "proxy": None,
        "nsfw": None,
        "webbrowser": None,
        "no_webbrowser": None,
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


def test_no_webbrowser_flag_inverts_into_the_stored_positive():
    """The flag says --no-webbrowser; the file says webbrowser = true."""
    stored = SherlockSettings()

    assert _resolve(stored).webbrowser.value is True
    assert _resolve(stored, no_webbrowser=True).webbrowser.value is False
    assert _resolve(stored, no_webbrowser=True).webbrowser.source == "flag"


def test_the_browser_can_be_forced_back_on_for_one_run():
    """Without the positive flag, a stored `webbrowser = false` is a one-way
    door: every later scan is degraded and no command line can undo it for a
    single run, on the one setting that decides whether answers are right."""
    stored = SherlockSettings(scan=ScanSettings(webbrowser=False))

    assert _resolve(stored).webbrowser.value is False
    assert _resolve(stored, webbrowser=True).webbrowser.value is True
    assert _resolve(stored, webbrowser=True).webbrowser.source == "flag"


def test_a_stored_browserless_default_is_reported_at_scan_start():
    """The setting that changes what a scan FINDS is the one that must be echoed.

    Someone reading the output days later has no other way to know the browser
    was off -- it is not in the command line, because nobody typed it.
    """
    stored = SherlockSettings(scan=ScanSettings(webbrowser=False))

    resolved = _resolve(stored)

    assert resolved.webbrowser.source == "config"
    assert resolved.from_config() == {"web browser": False}


def test_defaults_come_from_the_schema_not_from_a_second_list():
    """A restated default is one that can silently disagree with the real one."""
    def field(key: str):
        return next(item for item in SETTING_FIELDS if item.key == key)

    assert field_default(field("scan.concurrency")) == 30
    assert field_default(field("scan.webbrowser")) is True
    assert field_default(field("output.color")) is True


def test_no_proxy_is_an_answer_but_no_model_is_an_absence():
    """The distinction the reset key turns on.

    `scan.proxy` really does ship as None, so resetting it to None is correct.
    `ai.model` has no shipped value at all -- which model is right depends on
    what the user downloaded -- so there is nothing to restore and blanking it
    would be worse than refusing.
    """
    def field(key: str):
        return next(item for item in SETTING_FIELDS if item.key == key)

    assert field_default(field("scan.proxy")) is None
    assert field_default(field("ai.model")) is NO_DEFAULT
    assert field_default(field("ai.base_url")) == "http://127.0.0.1:1234"


def test_both_sides_of_the_transport_trade_are_labelled():
    """The default needs a label too, or it reads as an oversight.

    "slower; accurate" beside the browser is what says the default was chosen,
    and it is the only thing making the faster mode discoverable to someone who
    never reads the help line. Only the risky side is flagged as a warning,
    because only one of the two should look like an alarm.
    """
    field = next(item for item in SETTING_FIELDS if item.key == "scan.webbrowser")

    on = field_note(field, True)
    off = field_note(field, False)

    assert (on.text, on.warning) == ("slower; accurate", False)
    assert (off.text, off.warning) == ("faster; inaccurate", True)


def test_no_setting_offers_a_placeholder_link():
    """A "read more" that goes nowhere is worse than none at all.

    `TRANSPORT_DOC_URL` was "https://example.com" while the real page did not
    exist, and it was printed on the settings help line, in the plain-text
    listing and in the scan warning -- an obviously fake link, in a tool whose
    job is telling you what is real. It is empty until there is a page; every
    surface already guards on it being non-empty.
    """
    for field in SETTING_FIELDS:
        assert "example.com" not in field.doc_url, (
            f"{field.key} points at a placeholder URL"
        )


def test_unremarkable_settings_carry_no_label():
    """Labelling all eleven rows is wallpaper, and buries the one that counts."""
    for key in ("scan.concurrency", "scan.timeout", "output.verbose"):
        field = next(item for item in SETTING_FIELDS if item.key == key)
        assert not field_note(field, 30)


def test_the_transport_description_changes_with_the_transport():
    """"The browser setting" is two different facts depending on its value.

    On, the sentence worth reading is why the results can be trusted; off, it
    is what has been given up. A single value-independent caption could only
    say one of them, and would be wrong half the time.
    """
    field = next(item for item in SETTING_FIELDS if item.key == "scan.webbrowser")

    assert "can be trusted" in field_description(field, True)
    assert "reported as absent" in field_description(field, False)


def test_the_transport_description_names_the_real_mechanism():
    """"Dynamic sites" is the tempting simplification and is wrong in the
    direction that matters -- a server-rendered site is dynamic and arrives
    complete, so the fast path handles it fine. The failure is client-side
    rendering specifically, and the term is named once so it can be looked up.
    """
    field = next(item for item in SETTING_FIELDS if item.key == "scan.webbrowser")

    assert "client-side rendering" in field_description(field, False)


def test_every_setting_has_something_to_say_for_itself():
    """The help line is worthless if it is blank on most rows.

    It also guards the next setting added: a field with no description silently
    produces an empty caption rather than an error.
    """
    for field in SETTING_FIELDS:
        assert field_description(field, None).strip(), field.key


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
