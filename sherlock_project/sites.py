"""Sherlock Sites Information Module

This module supports storing information about websites.
This is the raw data that will be used to search for usernames.
"""
import json
import os
import secrets

import requests

from sherlock_project.wmn_adapter import adapt_wmn_manifest

MANIFEST_URL = "https://data.sherlockproject.xyz"
EXCLUSIONS_URL = "https://raw.githubusercontent.com/sherlock-project/sherlock/refs/heads/exclusions/false_positive_exclusions.txt"

# The bundled WhatsMyName dataset is the default manifest. It ships with the
# package, so the common path makes no network call at all -- neither for the
# manifest nor for exclusions. The exclusions list existed only to patch the
# legacy dataset's false positives; a two-sided rule that stops matching now
# reports UNKNOWN instead, so there is nothing left to subtract.
WMN_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "resources", "wmn-data.json")


def is_wmn_manifest(site_data) -> bool:
    """Whether parsed json is a WhatsMyName manifest rather than a legacy one.

    WMN is an object with a 'sites' array; the legacy manifest is a flat object
    keyed by site name. Sniffing lets --json keep accepting either.
    """
    return isinstance(site_data, dict) and isinstance(site_data.get("sites"), list)

class SiteInformation:
    def __init__(self, name, url_home, url_username_format, username_claimed,
                information, is_nsfw):
        """Create Site Information Object.

        Contains information about a specific website.

        Keyword Arguments:
        self                   -- This object.
        name                   -- String which identifies site.
        url_home               -- String containing URL for home of site.
        url_username_format    -- String containing URL for Username format
                                  on site.
                                  NOTE:  The string should contain the
                                         token "{}" where the username should
                                         be substituted.  For example, a string
                                         of "https://somesite.com/users/{}"
                                         indicates that the individual
                                         usernames would show up under the
                                         "https://somesite.com/users/" area of
                                         the website.
        username_claimed       -- String containing username which is known
                                  to be claimed on website.
        information            -- Dictionary containing all known information
                                  about website.
                                  NOTE:  Custom information about how to
                                         actually detect the existence of the
                                         username will be included in this
                                         dictionary.  This information will
                                         be needed by the detection method,
                                         but it is only recorded in this
                                         object for future use.
        is_nsfw                -- Boolean indicating if site is Not Safe For Work.

        Return Value:
        Nothing.
        """

        self.name = name
        self.url_home = url_home
        self.url_username_format = url_username_format

        self.username_claimed = username_claimed
        self.username_unclaimed = secrets.token_urlsafe(32)
        self.information = information
        self.is_nsfw  = is_nsfw


    def __str__(self):
        """Convert Object To String.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Nicely formatted string to get information about this object.
        """

        return f"{self.name} ({self.url_home})"


class SitesInformation:
    def __init__(
            self,
            data_file_path: str|None = None,
            honor_exclusions: bool = True,
            do_not_exclude: list[str] | None = None,
        ):
        """Create Sites Information Object.

        Contains information about all supported websites.

        Keyword Arguments:
        self                   -- This object.
        data_file_path         -- String which indicates path to data file.
                                  The file name must end in ".json".

                                  There are 3 possible formats:
                                   * Absolute File Format
                                     For example, "c:/stuff/data.json".
                                   * Relative File Format
                                     The current working directory is used
                                     as the context.
                                     For example, "data.json".
                                   * URL Format
                                     For example,
                                     "https://example.com/data.json", or
                                     "http://example.com/data.json".

                                  An exception will be thrown if the path
                                  to the data file is not in the expected
                                  format, or if there was any problem loading
                                  the file.

                                  If this option is not specified, then a
                                  default site list will be used.

        Return Value:
        Nothing.
        """

        self.rejected = []

        if not data_file_path:
            # Bundled by default: no network call, and no runtime dependency on
            # anyone else's infrastructure to run a scan.
            data_file_path = WMN_MANIFEST_PATH

        if data_file_path.lower().startswith("http"):
            # Reference is to a URL.
            try:
                response = requests.get(url=data_file_path, timeout=30)
            except Exception as error:
                raise FileNotFoundError(
                    f"Problem while attempting to access data file URL '{data_file_path}':  {error}"
                )

            if response.status_code != 200:
                raise FileNotFoundError(f"Bad response while accessing "
                                        f"data file URL '{data_file_path}'."
                                        )
            try:
                site_data = response.json()
            except Exception as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}':  {error}."
                )

        else:
            # Reference is to a file.
            try:
                with open(data_file_path, "r", encoding="utf-8") as file:
                    try:
                        site_data = json.load(file)
                    except Exception as error:
                        raise ValueError(
                            f"Problem parsing json contents at '{data_file_path}':  {error}."
                        )

            except FileNotFoundError:
                raise FileNotFoundError(f"Problem while attempting to access "
                                        f"data file '{data_file_path}'."
                                        )

        if is_wmn_manifest(site_data):
            self._load_wmn(site_data)
            return

        site_data.pop('$schema', None)

        if honor_exclusions:
            try:
                response = requests.get(url=EXCLUSIONS_URL, timeout=10)
                if response.status_code == 200:
                    exclusions = response.text.splitlines()
                    exclusions = [exclusion.strip() for exclusion in exclusions]

                    for site in do_not_exclude or []:
                        if site in exclusions:
                            exclusions.remove(site)

                    for exclusion in exclusions:
                        try:
                            site_data.pop(exclusion, None)
                        except KeyError:
                            pass

            except Exception:
                # If there was any problem loading the exclusions, just continue without them
                print("Warning: Could not load exclusions, continuing without them.")
                honor_exclusions = False

        self.sites = {}

        # Add all site information from the json file to internal site list.
        for site_name in site_data:
            try:

                self.sites[site_name] = \
                    SiteInformation(site_name,
                                    site_data[site_name]["urlMain"],
                                    site_data[site_name]["url"],
                                    site_data[site_name]["username_claimed"],
                                    site_data[site_name],
                                    site_data[site_name].get("isNSFW",False)

                                    )
            except KeyError as error:
                raise ValueError(
                    f"Problem parsing json contents at '{data_file_path}':  Missing attribute {error}."
                )
            except TypeError:
                print(f"Encountered TypeError parsing json contents for target '{site_name}' at {data_file_path}\nSkipping target.\n")


    def _load_wmn(self, site_data: dict):
        """Populate from a WhatsMyName manifest.

        Rejected rules are retained rather than raised so a caller can surface
        dataset rot without losing the sites that still work.
        """
        adapted, self.rejected = adapt_wmn_manifest(site_data)

        self.sites = {
            name: SiteInformation(
                name,
                record["urlMain"],
                record["url"],
                record["username_claimed"],
                record,
                record["isNSFW"],
            )
            for name, record in adapted.items()
        }

    def remove_nsfw_sites(self, do_not_remove: list | None = None):
        """
        Remove NSFW sites from the sites, if isNSFW flag is true for site

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        None
        """
        sites = {}
        do_not_remove = [site.casefold() for site in do_not_remove or []]
        for site in self.sites:
            if self.sites[site].is_nsfw and site.casefold() not in do_not_remove:
                continue
            sites[site] = self.sites[site]
        self.sites =  sites

    def site_name_list(self):
        """Get Site Name List.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        List of strings containing names of sites.
        """

        return sorted([site.name for site in self], key=str.lower)

    def __iter__(self):
        """Iterator For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Iterator for sites object.
        """

        for site_name in self.sites:
            yield self.sites[site_name]

    def __len__(self):
        """Length For Object.

        Keyword Arguments:
        self                   -- This object.

        Return Value:
        Length of sites object.
        """
        return len(self.sites)
