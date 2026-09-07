from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import TypeIs
from urllib.parse import urlsplit, urlunsplit

from trafilatura import extract, extract_metadata, html2txt

_HTML_METADATA_LABELS = {
    "description": "Description",
    "author": "Author",
    "og:title": "Open Graph title",
    "og:description": "Open Graph description",
    "twitter:title": "Twitter title",
    "twitter:description": "Twitter description",
    "twitter:creator": "Twitter creator",
}

_METADATA_DEDUPLICATION_GROUPS = {
    "description": "description",
    "og:description": "description",
    "twitter:description": "description",
    "og:title": "title",
    "twitter:title": "title",
    "og:url": "url",
}

_BREACH_ARTIFACT_MARKERS = (
    "combo_dump",
    "date_compromised",
    "device_id",
    "infostealer",
    "info-stealer",
    "malware_path",
    "stealer_family",
    "top_logins",
    "top_passwords",
)
_MISSING_PROFILE_TITLE_PATTERN = re.compile(
    r"(?i)^(?:"
    r"404(?:\s+error)?(?:\s*[-:]?\s*page)?\s+not\s+found"
    r"|(?:user|profile|account)(?:\s+was|\s+is|\s+has\s+been)?\s+"
    r"(?:not\s+found|unavailable|deleted|removed|does\s+not\s+exist)"
    r")(?:\s*[-|:•].*)?$"
)
_QUOTED_BIOGRAPHY_START = re.compile(
    r"^(?P<prefix>.+?:)\s*[\"“](?P<first>.+)$"
)
MAX_PROFILE_CONTENT_CHARS = 18_000


@dataclass(frozen=True, slots=True)
class ProfileContentDiagnostics:
    """Explain how one saved response became bounded Pass 1 model input."""

    content: str
    outcome: str
    response_chars: int
    saw_markup: bool
    metadata_fields: tuple[str, ...]
    main_content_method: str
    main_content_chars: int
    prepared_content_chars: int
    truncated: bool


class _HTMLMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.saw_markup = False
        self.title_parts: list[str] = []
        self.canonical_urls: list[str] = []
        self.values: dict[str, list[str]] = {}
        self._inside_title = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.saw_markup = True
        tag = tag.casefold()
        attributes = {
            name.casefold(): value
            for name, value in attrs
            if name and value is not None
        }

        if tag == "title":
            self._inside_title = True
            return

        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            content = attributes.get("content")
            if not key or not content:
                return

            key = key.strip().casefold()
            if key in _HTML_METADATA_LABELS or key.startswith("profile:"):
                self.values.setdefault(key, []).append(content)
            return

        if tag == "link":
            rel = attributes.get("rel", "").casefold().split()
            href = attributes.get("href")
            if "canonical" in rel and href:
                self.canonical_urls.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self.title_parts.append(data)


class _MetadataOutput:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.seen_values: set[str] = set()
        self._seen_grouped_values: set[tuple[str, str]] = set()

    def add(
        self,
        label: str,
        value: object,
        deduplication_group: str | None = None,
    ) -> None:
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value if item)
        if not isinstance(value, str):
            return

        cleaned = _clean_metadata_value(value)
        deduplication_key = _deduplication_key(cleaned)
        grouped_key = (deduplication_group or label.casefold(), deduplication_key)
        if not cleaned or grouped_key in self._seen_grouped_values:
            return

        self._seen_grouped_values.add(grouped_key)
        self.seen_values.add(deduplication_key)
        self.items.append((label, cleaned))


def _clean_metadata_value(value: str) -> str:
    lines = (" ".join(line.split()) for line in unescape(value).splitlines())
    return "\n".join(line for line in lines if line)


def _clean_main_content(value: str) -> str:
    lines = [line.rstrip() for line in value.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    cleaned_lines: list[str] = []
    previous_was_empty = False
    for line in lines:
        is_empty = not line
        if is_empty and previous_was_empty:
            continue
        cleaned_lines.append(line)
        previous_was_empty = is_empty

    return "\n".join(cleaned_lines)


def _deduplication_key(value: str) -> str:
    collapsed = " ".join(value.split())
    parsed_url = urlsplit(collapsed)
    if parsed_url.scheme.casefold() in {"http", "https"} and parsed_url.netloc:
        return urlunsplit(
            (
                parsed_url.scheme.casefold(),
                parsed_url.netloc.casefold(),
                parsed_url.path.rstrip("/"),
                parsed_url.query,
                parsed_url.fragment,
            )
        )
    return collapsed.casefold()


def _parse_html_metadata(response_text: str) -> _HTMLMetadataParser:
    parser = _HTMLMetadataParser()
    parser.feed(response_text)
    parser.close()
    return parser


def _extract_main_content(response_text: str) -> tuple[str, str]:
    method = "none"
    try:
        main_content = extract(
            response_text,
            output_format="markdown",
            favor_recall=True,
        )
    except ValueError:
        main_content = None

    if main_content:
        method = "trafilatura_extract"
    else:
        try:
            main_content = html2txt(response_text)
        except ValueError:
            main_content = None
        if main_content:
            method = "trafilatura_html2txt"

    cleaned = _clean_main_content(main_content or "")
    return cleaned, method if cleaned else "none"


def _add_html_metadata(
    output: _MetadataOutput,
    parser: _HTMLMetadataParser,
) -> None:
    output.add(
        "Title",
        "".join(parser.title_parts),
        deduplication_group="title",
    )

    for key, label in _HTML_METADATA_LABELS.items():
        for value in parser.values.get(key, []):
            output.add(
                label,
                value,
                deduplication_group=_METADATA_DEDUPLICATION_GROUPS.get(key),
            )

    for key in sorted(parser.values):
        if not key.startswith("profile:"):
            continue
        label = f"Profile {key.removeprefix('profile:').replace('_', ' ')}"
        for value in parser.values[key]:
            output.add(label, value)


def _add_trafilatura_metadata(
    output: _MetadataOutput,
    response_text: str,
) -> None:
    try:
        document = extract_metadata(response_text)
    except ValueError:
        return

    if document is None:
        return

    metadata = document.as_dict()
    for field, label, deduplication_group in (
        ("title", "Title", "title"),
        ("description", "Description", "description"),
        ("author", "Author", None),
    ):
        output.add(
            label,
            metadata.get(field),
            deduplication_group=deduplication_group,
        )


def _format_metadata(items: list[tuple[str, str]]) -> str:
    lines = ["## Page metadata"]
    for label, value in items:
        value_lines = value.splitlines()
        if label.casefold().endswith("description") and len(value_lines) > 1:
            match = _QUOTED_BIOGRAPHY_START.fullmatch(value_lines[0])
            if match is not None:
                biography_lines = [
                    match.group("first"),
                    *value_lines[1:],
                ]
                biography_lines[-1] = biography_lines[-1].rstrip("\"”")
                lines.append(f"- {label}: {match.group('prefix')}")
                lines.append("- Profile biography:")
                lines.extend(f"  - {line}" for line in biography_lines if line)
                continue
        if len(value_lines) == 1:
            lines.append(f"- {label}: {value}")
            continue

        lines.append(f"- {label}:")
        lines.extend(f"  {line}" for line in value_lines)
    return "\n".join(lines)


def _is_breach_artifact(response_text: str) -> bool:
    casefolded = response_text.casefold()
    matches = sum(
        marker in casefolded
        for marker in _BREACH_ARTIFACT_MARKERS
    )
    return matches >= 2


def _is_missing_profile_page(parser: _HTMLMetadataParser) -> bool:
    title = _clean_metadata_value("".join(parser.title_parts))
    return bool(title and _MISSING_PROFILE_TITLE_PATTERN.fullmatch(title))


def _truncate_profile_content(content: str) -> str:
    if len(content) <= MAX_PROFILE_CONTENT_CHARS:
        return content
    truncated = content[:MAX_PROFILE_CONTENT_CHARS]
    if "\n" in truncated[-500:]:
        truncated = truncated.rsplit("\n", maxsplit=1)[0]
    return truncated.rstrip()


# --- JSON responses -------------------------------------------------------
#
# A large share of the manifest checks an API endpoint rather than a rendered
# page: `wmn-data.json` alone points dozens of Mastodon instances at
# `/api/v1/accounts/lookup`. Those bodies are JSON, and running them through
# the HTML path does not merely waste effort, it destroys evidence. Trafilatura
# finds the `<p>` tags inside a JSON string value, concludes the document is
# HTML, and renders a hybrid: some markup resolved to text, the surrounding
# JSON left as literal punctuation, and every field before the first tag --
# `display_name` among them -- discarded as boilerplate. What reaches the model
# is the one thing it cannot do at 4B: parse a document format by hand.
#
# Parsing it here instead turns the format from a liability into the best input
# in the pipeline. A profile API hands over the field NAMES, which is precisely
# what Pass 1 spends its reasoning budget inventing on an HTML page.

_JSON_DOCUMENT_STARTS = ("{", "[")
_MAX_JSON_DEPTH = 3
_MAX_JSON_FIELDS = 60
_HTML_TAG_PATTERN = re.compile(r"<[A-Za-z/!][^>]*>")
_TIMESTAMP_VALUE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_OPAQUE_ID_VALUE_PATTERN = re.compile(
    r"(?i)^(?:\d{4,}|[0-9a-f]{16,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
    r"-[0-9a-f]{4}-[0-9a-f]{12})$"
)
# Anchored, and whitespace-free by construction: an asset URL is noise only
# when the value IS one. Matching anywhere inside the string instead deletes a
# biography for ending with a link to its author's own portfolio image.
_MEDIA_URL_VALUE_PATTERN = re.compile(
    r"(?i)^\S+\.(?:png|jpe?g|gif|webp|svg|bmp|ico|avif|mp4|webm|mov|mp3|ogg)"
    r"(?:[?#]\S*)?$"
)
_NAME_VALUE_PAIR_KEYS = ({"name", "value"}, {"key", "value"}, {"label", "value"})
# Fragment-level block tags. Deliberately not a document parser: trafilatura's
# document heuristics are what mangled these bodies, and a profile bio is a
# fragment, so the only structure worth keeping is where the line breaks fall.
_FRAGMENT_BREAK_TAGS = frozenset(
    {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
)


class _FragmentTextParser(HTMLParser):
    """Flatten an HTML fragment held inside a JSON string value to text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in _FRAGMENT_BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _identity_token(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _html_fragment_to_text(value: str) -> str:
    parser = _FragmentTextParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts)


def _clean_json_string(value: str) -> str:
    """Render one JSON string leaf as the text a reader would have seen.

    Mastodon splits a link across `<span class="invisible">https://</span>` and
    a visible remainder, so flattening the fragment reassembles the URL the
    profile actually advertises rather than the two halves the markup shows.
    """

    text = value
    if _HTML_TAG_PATTERN.search(text):
        text = _html_fragment_to_text(text)
    return _clean_metadata_value(unescape(text))


def _is_noise_json_value(value: str) -> bool:
    """Drop leaves by SHAPE, never by field name.

    Every rule here describes a kind of string no profile reader would quote:
    a timestamp, an opaque row id, an asset URL. Naming the fields instead
    would buy exactly the sites already seen -- the same trap the Pass 1 key
    and value denylists fell into.
    """

    if not value:
        return True
    if _TIMESTAMP_VALUE_PATTERN.fullmatch(value):
        return True
    if _OPAQUE_ID_VALUE_PATTERN.fullmatch(value):
        return True
    return bool(_MEDIA_URL_VALUE_PATTERN.fullmatch(value))


def _looks_like_json_document(response_text: str) -> bool:
    return response_text.lstrip().startswith(_JSON_DOCUMENT_STARTS)


def _parse_json_document(response_text: str) -> object | None:
    """Parse a body already known to declare itself JSON, or give up on it.

    Returning `None` here means unreadable, never "try HTML instead": the
    caller dispatches on the format the body claims, so a JSON body that will
    not parse settles as empty. Falling through would hand the plain-text
    branch a document with no markup in it, and Pass 1 would be asked to read
    18,000 characters of punctuation.
    """

    try:
        return json.loads(response_text.lstrip())
    except (ValueError, RecursionError):
        # RecursionError, not just ValueError: a deeply nested body raises it
        # from the C parser. Uncaught it reaches `ai_worker`'s blind except,
        # which records the site as pending -- retrying the same unparseable
        # response on every future run rather than settling it once.
        return None


def _looks_like_profile_record(candidate: object) -> bool:
    if not isinstance(candidate, dict):
        return False
    named_strings = sum(
        1
        for key, value in candidate.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and _clean_json_string(value)
    )
    # Two, not one: an error envelope such as `{"error": "Record not found"}`
    # is a dict of strings too, and must not read as somebody's profile.
    return named_strings >= 2


def _collect_profile_records(document: object) -> list[dict[str, object]]:
    if _looks_like_profile_record(document):
        return [document]  # type: ignore[list-item]

    candidates: list[object] = []
    if isinstance(document, list):
        candidates = document
    elif isinstance(document, dict):
        for value in document.values():
            if isinstance(value, list):
                candidates.extend(value)
    return [item for item in candidates if _looks_like_profile_record(item)]


def _record_identity_tokens(record: dict[str, object]) -> set[str]:
    tokens: set[str] = set()
    for value in record.values():
        if not isinstance(value, str):
            continue
        cleaned = _clean_json_string(value)
        if not cleaned or "\n" in cleaned:
            continue
        tokens.add(_identity_token(cleaned))
        # `acct` arrives federated as `user@host`; the local part is the handle.
        tokens.add(_identity_token(cleaned.split("@", maxsplit=1)[0]))
    return tokens - {""}


def _select_owner_record(
    records: list[dict[str, object]],
    searched_username: str,
) -> tuple[dict[str, object] | None, str]:
    """Pick the one record this response is about, or refuse to guess.

    A search endpoint answers with every account matching the query, and the
    pipeline's whole contract -- prompt included -- is "one website profile's
    owner". Handed several, a model does not pick one: it writes down the union,
    and a scan of `0day` comes back with a Florida pentester's name over a
    Madrid hackerspace's meeting times, as one person who does not exist.
    Returning nothing loses a real lead; returning a composite invents a
    fictional human and hands it to Pass 2 as evidence. Only one of those is
    recoverable.
    """

    if not records:
        return None, "no_extractable_content"
    if len(records) == 1:
        return records[0], "extracted"

    token = _identity_token(searched_username)
    matches = [record for record in records if token in _record_identity_tokens(record)]
    if len(matches) == 1:
        return matches[0], "extracted"
    if not matches:
        return None, "no_matching_profile_record"
    return None, "ambiguous_profile_records"


def _render_json_record(
    record: dict[str, object],
    *,
    label_prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key, value in record.items():
        if not isinstance(key, str) or len(fields) >= _MAX_JSON_FIELDS:
            break
        label = f"{label_prefix} {key}".strip()
        fields.extend(_render_json_value(value, label=label, depth=depth))
    return fields


def _render_json_value(
    value: object,
    *,
    label: str,
    depth: int,
) -> list[tuple[str, str]]:
    # Non-string scalars carry no owner evidence and every count, flag, and row
    # id in one rule: `followers_count`, `bot`, and `id` need no denylist entry
    # because none of them is text a profile states about its owner.
    if isinstance(value, str):
        cleaned = _clean_json_string(value)
        return [] if _is_noise_json_value(cleaned) else [(label, cleaned)]
    if depth >= _MAX_JSON_DEPTH:
        return []
    if isinstance(value, dict):
        return _render_json_record(value, label_prefix=label, depth=depth + 1)
    if isinstance(value, list):
        fields: list[tuple[str, str]] = []
        for item in value:
            fields.extend(
                _render_json_pair(item, label=label, depth=depth)
                if _is_name_value_pair(item)
                else _render_json_value(item, label=label, depth=depth + 1)
            )
        return fields
    return []


def _is_name_value_pair(item: object) -> TypeIs[dict[str, object]]:
    return isinstance(item, dict) and any(
        keys <= {key.casefold() for key in item if isinstance(key, str)}
        for keys in _NAME_VALUE_PAIR_KEYS
    )


def _render_json_pair(
    item: dict[str, object],
    *,
    label: str,
    depth: int,
) -> list[tuple[str, str]]:
    """Render an owner-authored `{name, value}` row under its own name.

    Mastodon's profile `fields` are the clearest owner evidence on the page and
    the label is written by the owner, so it is kept: `field hacktivity` says
    more than `fields` repeated four times.
    """

    pairs = {key.casefold(): value for key, value in item.items()}
    name = pairs.get("name") or pairs.get("key") or pairs.get("label")
    value = pairs.get("value")
    if not isinstance(name, str) or not isinstance(value, str):
        return []
    cleaned_name = _clean_json_string(name).replace("\n", " ")
    if not cleaned_name:
        return []
    singular = label.removesuffix("s") if label.endswith("s") else label
    return _render_json_value(
        value,
        label=f"{singular} {cleaned_name}".strip(),
        depth=depth,
    )


def _format_json_record(fields: list[tuple[str, str]]) -> str:
    lines = ["## Profile record"]
    for label, value in fields:
        value_lines = value.splitlines()
        if len(value_lines) == 1:
            lines.append(f"- {label}: {value}")
            continue
        lines.append(f"- {label}:")
        lines.extend(f"  {line}" for line in value_lines)
    return "\n".join(lines)


def _inspect_json_profile(
    document: object,
    *,
    response_chars: int,
    searched_username: str,
) -> ProfileContentDiagnostics:
    record, outcome = _select_owner_record(
        _collect_profile_records(document),
        searched_username,
    )
    if record is None:
        return ProfileContentDiagnostics(
            content="",
            outcome=outcome,
            response_chars=response_chars,
            saw_markup=False,
            metadata_fields=(),
            main_content_method="json_profile_record",
            main_content_chars=0,
            prepared_content_chars=0,
            truncated=False,
        )

    fields = _render_json_record(record)
    if not fields:
        return ProfileContentDiagnostics(
            content="",
            outcome="no_extractable_content",
            response_chars=response_chars,
            saw_markup=False,
            metadata_fields=(),
            main_content_method="json_profile_record",
            main_content_chars=0,
            prepared_content_chars=0,
            truncated=False,
        )

    rendered = _format_json_record(fields)
    content = _truncate_profile_content(rendered)
    return ProfileContentDiagnostics(
        content=content,
        outcome="extracted",
        response_chars=response_chars,
        saw_markup=False,
        metadata_fields=tuple(label for label, _ in fields),
        main_content_method="json_profile_record",
        main_content_chars=len(rendered),
        prepared_content_chars=len(content),
        truncated=len(content) < len(rendered),
    )


def inspect_profile_content(
    response_text: str,
    *,
    searched_username: str = "",
) -> ProfileContentDiagnostics:
    """Extract Pass 1 input together with stage-level diagnostics.

    `searched_username` disambiguates a response holding several profile
    records; without it such a response is refused rather than merged.
    """

    response_chars = len(response_text)
    if not response_text.strip():
        return ProfileContentDiagnostics(
            content="",
            outcome="empty_response",
            response_chars=response_chars,
            saw_markup=False,
            metadata_fields=(),
            main_content_method="none",
            main_content_chars=0,
            prepared_content_chars=0,
            truncated=False,
        )
    if _is_breach_artifact(response_text):
        return ProfileContentDiagnostics(
            content="",
            outcome="breach_artifact",
            response_chars=response_chars,
            saw_markup="<" in response_text,
            metadata_fields=(),
            main_content_method="skipped",
            main_content_chars=0,
            prepared_content_chars=0,
            truncated=False,
        )

    if _looks_like_json_document(response_text):
        return _inspect_json_profile(
            _parse_json_document(response_text),
            response_chars=response_chars,
            searched_username=searched_username,
        )

    parser = _parse_html_metadata(response_text)
    if _is_missing_profile_page(parser):
        return ProfileContentDiagnostics(
            content="",
            outcome="missing_profile_page",
            response_chars=response_chars,
            saw_markup=parser.saw_markup,
            metadata_fields=(),
            main_content_method="skipped",
            main_content_chars=0,
            prepared_content_chars=0,
            truncated=False,
        )
    metadata = _MetadataOutput()
    _add_html_metadata(metadata, parser)
    _add_trafilatura_metadata(metadata, response_text)

    main_content, main_content_method = _extract_main_content(response_text)
    if not main_content and not parser.saw_markup:
        main_content = _clean_main_content(response_text)
        if main_content:
            main_content_method = "plain_text"
    if _deduplication_key(main_content) in metadata.seen_values:
        main_content = ""
        main_content_method = "deduplicated"

    sections: list[str] = []
    if metadata.items:
        sections.append(_format_metadata(metadata.items))
    if main_content:
        sections.append(f"## Main content\n{main_content}")

    untruncated_content = "\n\n".join(sections)
    content = _truncate_profile_content(untruncated_content)
    return ProfileContentDiagnostics(
        content=content,
        outcome="extracted" if content else "no_extractable_content",
        response_chars=response_chars,
        saw_markup=parser.saw_markup,
        metadata_fields=tuple(label for label, _ in metadata.items),
        main_content_method=main_content_method,
        main_content_chars=len(main_content),
        prepared_content_chars=len(content),
        truncated=len(content) < len(untruncated_content),
    )


def extract_profile_content(
    response_text: str,
    *,
    searched_username: str = "",
) -> str:
    """Extract useful profile metadata and visible text for AI analysis."""

    return inspect_profile_content(
        response_text,
        searched_username=searched_username,
    ).content
