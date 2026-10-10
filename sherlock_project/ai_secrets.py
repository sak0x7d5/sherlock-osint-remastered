"""Where a hosted provider's API key lives: the environment, or the OS keychain.

Never the config file. `config.toml` is plain text in a per-user directory; it
travels with backups and gets pasted into bug reports via `setup ai --show`.
A key written there is a key published. So the file records only the NAME of an
environment variable, and a key typed into the UI goes to the operating
system's own credential store -- Windows Credential Manager, the macOS
Keychain, or the Secret Service on a Linux desktop -- through `keyring`.

Lookup order is environment first, keychain second. The variable is the
explicit, per-shell choice, and someone who exports a different key for one
run must get that key, not the one they stored months ago.

A machine with no credential store at all (Docker, a headless server) is
normal, not an error: storing reports that plainly and points at the variable,
and looking up simply finds nothing. Nothing here is allowed to crash a scan
or the UI because a keychain daemon is missing or locked.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from sherlock_project.ai_config import AIConfigError

KEYRING_SERVICE = "sherlock-rm"
KeySource = Literal["env", "keychain"]

# keyring's backends, keyed by (module, class), as someone would name the store.
_BACKEND_LABELS = {
    ("Windows", "WinVaultKeyring"): "Windows Credential Manager",
    ("macOS", "Keyring"): "macOS Keychain",
    ("SecretService", "Keyring"): "Secret Service",
    ("kwallet", "DBusKeyring"): "KWallet",
}


def _keyring():
    # Imported on use, not at module load: a broken backend plugin must cost
    # the key row, not every import of the AI stack.
    import keyring

    return keyring


def _active_backend():
    try:
        return _keyring().get_keyring()
    except Exception:
        return None


def keychain_available() -> bool:
    """Whether this machine has a credential store keyring can actually use."""
    backend = _active_backend()
    if backend is None:
        return False
    module = type(backend).__module__
    if module.endswith((".fail", ".null")):
        return False
    if module.endswith(".chainer") and not getattr(backend, "backends", None):
        return False
    return True


def keychain_label() -> str:
    """The credential store's name, for a row that says where a key is."""
    backend = _active_backend()
    if backend is None or not keychain_available():
        return "no keychain"
    kind = type(backend)
    label = _BACKEND_LABELS.get((kind.__module__.rsplit(".", 1)[-1], kind.__name__))
    return label or getattr(backend, "name", kind.__name__)


def no_keychain_message(env_name: str) -> str:
    return (
        "This machine has no OS keychain to store a key in. Set the "
        f"{env_name} environment variable instead; it is read at run time "
        "and never stored."
    )


def resolve_api_key(
    provider: str,
    env_name: str,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, KeySource | None]:
    """The key to use, and where it came from. (None, None) if there is none.

    A keychain that cannot be read -- locked, no daemon, a backend error -- is
    treated as holding nothing. The caller already says "no key: set one" in
    that case, which is the right advice, and raising here would turn a
    missing convenience into a failed scan.
    """
    environment = os.environ if environ is None else environ
    value = (environment.get(env_name) or "").strip()
    if value:
        return value, "env"
    if not keychain_available():
        return None, None
    try:
        stored = _keyring().get_password(KEYRING_SERVICE, provider)
    except Exception:
        return None, None
    stored = (stored or "").strip()
    return (stored, "keychain") if stored else (None, None)


def store_api_key(provider: str, key: str, *, env_name: str) -> None:
    key = key.strip()
    if not key:
        raise AIConfigError("An empty API key cannot be stored.")
    if not keychain_available():
        raise AIConfigError(no_keychain_message(env_name))
    try:
        _keyring().set_password(KEYRING_SERVICE, provider, key)
    except Exception as error:
        raise AIConfigError(
            f"The OS keychain refused the key ({type(error).__name__}). "
            + no_keychain_message(env_name).split(". ", 1)[1]
        ) from error


def delete_api_key(provider: str) -> bool:
    """Remove a stored key. True if one was there to remove."""
    if not keychain_available():
        return False
    keyring = _keyring()
    try:
        if keyring.get_password(KEYRING_SERVICE, provider) is None:
            return False
        keyring.delete_password(KEYRING_SERVICE, provider)
    except Exception as error:
        raise AIConfigError(
            f"The OS keychain could not remove the key ({type(error).__name__})."
        ) from error
    return True


def describe_key_source(
    provider: str,
    env_name: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Where the key comes from, in words. Never the key, nor any part of it."""
    _, source = resolve_api_key(provider, env_name, environ)
    if source == "env":
        return f"set · ${env_name}"
    if source == "keychain":
        return f"set · {keychain_label()}"
    return "not set"
