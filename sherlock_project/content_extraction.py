from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
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


def inspect_profile_content(response_text: str) -> ProfileContentDiagnostics:
    """Extract Pass 1 input together with stage-level diagnostics."""

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


def extract_profile_content(response_text: str) -> str:
    """Extract useful profile metadata and visible text for AI analysis."""

    return inspect_profile_content(response_text).content
