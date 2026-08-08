#! /usr/bin/env python3

"""
Sherlock: Find Usernames Across Social Networks Module

This module contains the main logic to search for usernames at social
networks.
"""

import sys

if __name__ == "__main__":
    # Check if the user is using the correct version of Python
    python_version = sys.version.split()[0]

    # Deliberately checks below the supported floor: this guard exists to give
    # a clear message on runtimes the package does not support, which is the
    # one case where the condition can be true. noqa: the block is not dead.
    if sys.version_info < (3, 13):  # noqa: UP036
        print(f"Sherlock requires Python 3.13+\nYou are using Python {python_version}, which is not supported by Sherlock.")
        sys.exit(1)

    from sherlock_project import sherlock
    sherlock.cli()
