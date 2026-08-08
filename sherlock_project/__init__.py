""" Sherlock Module

This module contains the main logic to search for usernames at social
networks.

"""

from importlib.metadata import version as pkg_version, PackageNotFoundError
import pathlib
import tomli


def get_version() -> str:
    """Fetch the version number of the installed package."""
    try:
        return pkg_version("sherlock_project")
    except PackageNotFoundError:
        pyproject_path: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            pyproject_data = tomli.load(f)
        return pyproject_data["tool"]["poetry"]["version"]

# This variable is only used to check for ImportErrors induced by users running as script rather than as module or package
import_error_test_var = None

__shortname__   = "Sherlock"
__longname__    = "Sherlock: Find Usernames Across Social Networks"
__version__     = get_version()

# Update checks must resolve against this derivative's own releases. Pointing
# at upstream would compare our version to theirs and advertise their download.
forge_api_latest_release = "https://api.github.com/repos/sak0x7d5/sherlock-osint-remastered/releases/latest"
