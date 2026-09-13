""" Sherlock Module

This module contains the main logic to search for usernames at social
networks.

"""

import pathlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

import tomli

# The DISTRIBUTION name, which is not the import package name. They were the
# same string until this project stopped calling itself "sherlock-project" --
# and `pkg_version` takes the distribution, so leaving it as "sherlock_project"
# made every installed copy raise PackageNotFoundError, fall through to a
# pyproject.toml that is not shipped inside a wheel, and die on `--version`
# with a FileNotFoundError pointing into site-packages. Editable installs hid
# it completely, because there `parent.parent` really is the repository.
_DISTRIBUTION_NAME = "sherlock-rm"


def get_version() -> str:
    """Fetch the version number of the installed package."""
    try:
        return pkg_version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        # Running from a source checkout that was never installed. Reading the
        # version out of pyproject.toml is right there and nowhere else.
        pyproject_path: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        try:
            with pyproject_path.open("rb") as f:
                pyproject_data = tomli.load(f)
            return pyproject_data["tool"]["poetry"]["version"]
        except (OSError, KeyError, tomli.TOMLDecodeError):
            # Neither source available. `--version` reporting "unknown" is a
            # poor answer; refusing to import the package at all is a worse
            # one, and that is what raising here would do -- `__version__` is
            # evaluated at module scope.
            return "unknown"

# This variable is only used to check for ImportErrors induced by users running as script rather than as module or package
import_error_test_var = None

__shortname__   = "Sherlock"
__longname__    = "Sherlock: Find Usernames Across Social Networks"
__version__     = get_version()

# Update checks must resolve against this derivative's own releases. Pointing
# at upstream would compare our version to theirs and advertise their download.
forge_api_latest_release = "https://api.github.com/repos/sak0x7d5/sherlock-osint-remastered/releases/latest"
