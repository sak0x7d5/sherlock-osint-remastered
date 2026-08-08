"""WhatsMyName Manifest Adapter

Normalizes the WhatsMyName (WMN) dataset into the internal site record shape
used throughout the scan pipeline.

WMN is a two-sided dataset: every entry describes both what a *hit* looks like
(``e_code`` + ``e_string``) and what a *miss* looks like (``m_code`` +
``m_string``). That is the property this project is adopting it for. The legacy
manifest could only describe absence, so "the user exists" and "our rule is
stale" produced identical output; with both sides recorded, a response matching
neither is an honest UNKNOWN rather than a silent false positive.

The adapter is deliberately total: a malformed or structurally undecidable entry
is rejected and reported, never raised. One bad site rule must not cost the
caller the other 719.

Upstream dataset: https://github.com/WebBreacher/WhatsMyName
Licensed CC BY-SA 4.0 (c) Micah Hoffman. The vendored copy in ``resources/``
is kept byte-for-byte intact, license block included -- see NOTICE.md.
"""

import json
from typing import Any
from urllib.parse import urlsplit

# The raw dataset, for the scheduled refresh workflow. Kept here rather than in
# sites.py because nothing in the scan path should reach for the network.
WMN_MANIFEST_URL = (
    "https://raw.githubusercontent.com/WebBreacher/WhatsMyName/main/wmn-data.json"
)

# WMN spells its username placeholder differently to the legacy manifest. No
# entry in the dataset contains a bare "{}", so the substitution is unambiguous.
WMN_PLACEHOLDER = "{account}"
INTERNAL_PLACEHOLDER = "{}"

# WMN carries NSFW as a category value rather than a boolean flag.
NSFW_CATEGORY = "xx NSFW xx"

REQUIRED_KEYS = ("name", "uri_check", "e_code", "e_string", "m_string", "m_code", "known", "cat")


class WmnRejection:
    """A site rule that could not be adapted, and why.

    Collected rather than raised so that callers can surface dataset rot
    without losing the entries that are still good.
    """

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def __str__(self) -> str:
        return f"{self.name}: {self.reason}"

    def __repr__(self) -> str:
        return f"WmnRejection({self.name!r}, {self.reason!r})"


def _site_origin(url: str) -> str:
    """Derive a site home URL from a check or profile URL."""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url
    return f"{parts.scheme}://{parts.netloc}/"


def _to_internal_placeholder(template: str) -> str:
    return template.replace(WMN_PLACEHOLDER, INTERNAL_PLACEHOLDER)


def normalize_username(username: str, strip_bad_char: str | None) -> str:
    """Drop characters a site refuses to accept in a username.

    WMN records these per site (``strip_bad_char``) because the site itself
    silently discards them -- "john.doe" and "johndoe" are the same account on
    those targets, so probing the literal string yields a false negative.
    """
    if not strip_bad_char:
        return username
    return "".join(char for char in username if char not in strip_bad_char)


def _validate(entry: dict[str, Any]) -> str | None:
    """Return a rejection reason, or None if the entry is usable."""
    missing = [key for key in REQUIRED_KEYS if key not in entry]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"

    if entry.get("valid") is False:
        return "marked valid=false upstream"

    post_body = entry.get("post_body")
    if WMN_PLACEHOLDER not in entry["uri_check"] and not post_body:
        return f"no {WMN_PLACEHOLDER} placeholder in uri_check and no post_body"

    if post_body and WMN_PLACEHOLDER not in post_body:
        return f"post_body has no {WMN_PLACEHOLDER} placeholder"

    if post_body and not entry.get("headers"):
        return "post_body without headers"

    if not isinstance(entry["e_code"], int) or not isinstance(entry["m_code"], int):
        return "non-integer status code"

    if not entry["known"]:
        return "no known usernames to validate against"

    # The whole point of the migration: a rule must be able to tell a hit from a
    # miss. Identical codes with no distinguishing string can only ever guess.
    if not entry["e_string"] and entry["e_code"] == entry["m_code"]:
        return "undecidable: no e_string and e_code == m_code"

    return None


def adapt_wmn_site(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one WMN entry into an internal site record.

    Raises ValueError if the entry is unusable; prefer adapt_wmn_manifest,
    which collects those instead.
    """
    reason = _validate(entry)
    if reason is not None:
        raise ValueError(reason)

    post_body = entry.get("post_body")
    check_url = _to_internal_placeholder(entry["uri_check"])

    # uri_pretty is the human-facing profile; uri_check may be an API endpoint.
    # Detection uses the latter, content extraction the former -- keeping them
    # apart is what lets an expensive page render happen only on a confirmed hit.
    pretty = entry.get("uri_pretty")
    profile_url = _to_internal_placeholder(pretty) if pretty else check_url

    record: dict[str, Any] = {
        "url": check_url,
        "urlProfile": profile_url,
        "urlMain": _site_origin(pretty or entry["uri_check"]),
        "username_claimed": entry["known"][0],
        "known": list(entry["known"]),
        "category": entry["cat"],
        "isNSFW": entry["cat"] == NSFW_CATEGORY,
        "request_method": "POST" if post_body else "GET",
        "detection": {
            "exists": {"code": entry["e_code"], "string": entry["e_string"]},
            "missing": {"code": entry["m_code"], "string": entry["m_string"]},
        },
    }

    if post_body:
        record["request_payload"] = _to_internal_placeholder(post_body)
    if entry.get("headers"):
        record["headers"] = dict(entry["headers"])
    if entry.get("strip_bad_char"):
        record["strip_bad_char"] = entry["strip_bad_char"]
    if entry.get("protection"):
        # Anti-automation measures the dataset knows about up front. Lets the
        # engine reserve the stealth browser for targets that need it instead
        # of discovering the block after the fact via response fingerprinting.
        record["protection"] = list(entry["protection"])

    return record


def adapt_wmn_manifest(
    raw: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[WmnRejection]]:
    """Adapt a parsed WMN manifest into ``{site_name: site_record}``.

    Returns the usable sites alongside the entries that were rejected, so a
    caller can report dataset rot without failing the scan.
    """
    sites: dict[str, dict[str, Any]] = {}
    rejected: list[WmnRejection] = []

    entries = raw.get("sites")
    if not isinstance(entries, list):
        raise ValueError("WMN manifest has no 'sites' array")

    for entry in entries:
        if not isinstance(entry, dict):
            rejected.append(WmnRejection("<malformed>", "entry is not an object"))
            continue

        name = entry.get("name") or "<unnamed>"
        try:
            record = adapt_wmn_site(entry)
        except ValueError as error:
            rejected.append(WmnRejection(name, str(error)))
            continue
        except Exception as error:
            rejected.append(WmnRejection(name, f"unexpected adapter failure: {error}"))
            continue

        if name in sites:
            rejected.append(WmnRejection(name, "duplicate site name"))
            continue

        sites[name] = record

    return sites, rejected


def load_wmn_manifest(
    data_file_path: str,
) -> tuple[dict[str, dict[str, Any]], list[WmnRejection]]:
    """Load and adapt a WMN manifest from a local path."""
    try:
        with open(data_file_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Problem while attempting to access data file '{data_file_path}'."
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Problem parsing json contents at '{data_file_path}':  {error}.")

    return adapt_wmn_manifest(raw)
