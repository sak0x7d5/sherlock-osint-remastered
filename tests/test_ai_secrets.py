import keyring
import keyring.backends.fail
import pytest

from sherlock_project.ai_config import AIConfigError
from sherlock_project.ai_secrets import (
    KEYRING_SERVICE,
    delete_api_key,
    describe_key_source,
    keychain_available,
    resolve_api_key,
    store_api_key,
)


def test_store_resolve_delete_round_trip(memory_keychain):
    store_api_key("gemini", "  stored-key  ", env_name="GEMINI_API_KEY")
    assert memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] == "stored-key"
    assert resolve_api_key("gemini", "GEMINI_API_KEY", {}) == ("stored-key", "keychain")
    assert delete_api_key("gemini") is True
    assert resolve_api_key("gemini", "GEMINI_API_KEY", {}) == (None, None)
    assert delete_api_key("gemini") is False


def test_the_environment_wins_over_the_keychain(memory_keychain):
    memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] = "stored-key"
    assert resolve_api_key(
        "gemini", "GEMINI_API_KEY", {"GEMINI_API_KEY": "env-key"}
    ) == ("env-key", "env")


def test_a_blank_variable_does_not_hide_a_stored_key(memory_keychain):
    memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] = "stored-key"
    assert resolve_api_key("gemini", "GEMINI_API_KEY", {"GEMINI_API_KEY": " "})[1] == (
        "keychain"
    )


def test_no_keychain_is_reported_plainly_and_never_raises_on_lookup():
    keyring.set_keyring(keyring.backends.fail.Keyring())
    assert keychain_available() is False
    assert resolve_api_key("gemini", "GEMINI_API_KEY", {}) == (None, None)
    with pytest.raises(AIConfigError, match="GEMINI_API_KEY"):
        store_api_key("gemini", "k", env_name="GEMINI_API_KEY")
    assert delete_api_key("gemini") is False


def test_a_failing_keychain_becomes_a_config_error(memory_keychain, monkeypatch):
    def broken(*_args):
        raise RuntimeError("locked")

    monkeypatch.setattr(memory_keychain, "set_password", broken)
    monkeypatch.setattr(memory_keychain, "get_password", broken)
    with pytest.raises(AIConfigError, match="refused"):
        store_api_key("gemini", "k", env_name="GEMINI_API_KEY")
    # Reading a locked keychain is "no key", not a crash.
    assert resolve_api_key("gemini", "GEMINI_API_KEY", {}) == (None, None)


def test_an_empty_key_is_refused():
    with pytest.raises(AIConfigError, match="empty"):
        store_api_key("gemini", "   ", env_name="GEMINI_API_KEY")


def test_the_description_names_the_source_and_never_the_key(memory_keychain):
    assert describe_key_source("gemini", "GEMINI_API_KEY", {}) == "not set"
    env = describe_key_source(
        "gemini", "GEMINI_API_KEY", {"GEMINI_API_KEY": "secret-value"}
    )
    assert env == "set · $GEMINI_API_KEY"
    memory_keychain.passwords[(KEYRING_SERVICE, "gemini")] = "secret-value"
    stored = describe_key_source("gemini", "GEMINI_API_KEY", {})
    assert stored.startswith("set · ")
    assert "secret" not in stored
