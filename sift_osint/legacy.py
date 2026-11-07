"""Compatibility with this project's previous name.

The tool used to be called `sherlock-rm`, and kept its data under an app
directory called `sherlock` and its settings in `SHERLOCK_*` environment
variables. Renaming those silently would have meant that anyone who had run a
scan opened the new build to an empty database and default settings -- the
evidence cache is the expensive thing here -- so the old names are still
honoured, in one place, and only as a fallback.

Nothing is moved or copied. A file at the old location is simply *used* until
a file exists at the new one, so there is no migration step that can half-finish
and no second copy to drift. New installs never see the old names.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

LEGACY_ENV_PREFIX = "SHERLOCK_"
ENV_PREFIX = "SIFT_"
LEGACY_APP_NAME = "sherlock"
APP_NAME = "sift-osint"


def env_value(environment: Mapping[str, str], name: str) -> str | None:
    """`SIFT_<name>`, else the legacy `SHERLOCK_<name>`, else None.

    An empty value counts as unset in both spellings, matching how every caller
    already treated `if override:`. A set-but-empty `SIFT_DB` therefore does not
    shadow a real `SHERLOCK_DB`.
    """
    return environment.get(ENV_PREFIX + name) or environment.get(
        LEGACY_ENV_PREFIX + name
    ) or None


def prefer_existing(new: Path, legacy: Path) -> Path:
    """`new`, unless only `legacy` exists on disk.

    Once the new file exists it wins permanently, so someone who deliberately
    starts fresh under the new name is not pulled back to the old data.
    """
    if new.exists() or not legacy.exists():
        return new
    return legacy
