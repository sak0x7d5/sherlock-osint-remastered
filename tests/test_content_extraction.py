import pytest

from sherlock_project.content_extraction import (
    MAX_PROFILE_CONTENT_CHARS,
    extract_profile_content,
    inspect_profile_content,
)

INSTAGRAM_HTML = """
<html>
  <head>
    <meta property="og:type" content="profile">
    <meta property="og:image" content="https://scontent.cdninstagram.com/signed-profile-image.jpg?token=secret">
    <meta property="og:title" content="Example Person (@fixture_handle) - Instagram photos and videos">
    <meta property="og:url" content="https://www.instagram.com/fixture_handle">
    <meta property="og:description" content="48.2K Followers, 806 Following, 91 Posts - See Instagram photos and videos from Example Person (@fixture_handle)">
    <meta name="description" content="48.2K Followers, 806 Following, 91 Posts - Example Person (@fixture_handle) on Instagram: &quot;\U0001f511Serial Entrepreneur
\U0001f499Wildlife Rescue Volunteer - (@example_foundation)
\U0001f916Penetration Tester&quot;">
    <link rel="alternate" href="android-app://com.instagram.android/https/instagram.com/_u/fixture_handle/">
    <link rel="canonical" href="https://www.instagram.com/fixture_handle/">
  </head>
  <body></body>
</html>
"""


def test_extract_profile_content_reads_instagram_metadata():
    result = extract_profile_content(INSTAGRAM_HTML)

    assert "Example Person (@fixture_handle)" in result
    assert "48.2K Followers, 806 Following, 91 Posts" in result
    assert "See Instagram photos and videos" in result
    assert "Serial Entrepreneur" in result
    assert "Wildlife Rescue Volunteer - (@example_foundation)" in result
    assert "Penetration Tester" in result
    assert "Description:" in result
    assert "- Profile biography:" in result
    assert "  - \U0001f511Serial Entrepreneur" in result
    assert "  - \U0001f916Penetration Tester" in result
    assert "Open Graph description:" in result
    assert "https://www.instagram.com/fixture_handle" not in result
    assert "<meta" not in result
    assert "scontent.cdninstagram.com" not in result
    assert "android-app://" not in result


def test_extract_profile_content_combines_metadata_and_body_text():
    html = """
    <html>
      <head><meta name="description" content="Security researcher"></head>
      <body><main><h1>Alice Example</h1><p>Builds defensive tools.</p></main></body>
    </html>
    """

    result = extract_profile_content(html)

    assert "Security researcher" in result
    assert "Alice Example" in result
    assert "Builds defensive tools." in result
    assert "## Page metadata" in result
    assert "## Main content" in result


def test_extract_profile_content_deduplicates_metadata_values():
    html = """
    <html><head>
      <meta name="description" content="Same profile biography">
      <meta property="og:description" content="Same profile biography">
      <meta name="twitter:description" content="Same profile biography">
    </head><body></body></html>
    """

    result = extract_profile_content(html)

    assert result.count("Same profile biography") == 1


def test_extract_profile_content_includes_twitter_and_profile_metadata():
    html = """
    <html><head>
      <meta name="twitter:creator" content="@alice_sec">
      <meta property="profile:first_name" content="Alice">
      <meta property="profile:last_name" content="Example">
      <meta property="profile:username" content="alice">
    </head><body></body></html>
    """

    result = extract_profile_content(html)

    assert "Twitter creator: @alice_sec" in result
    assert "Profile first name: Alice" in result
    assert "Profile last name: Example" in result
    assert "Profile username: alice" in result


def test_extract_profile_content_preserves_multiline_unicode_metadata():
    result = extract_profile_content(INSTAGRAM_HTML)

    assert "\U0001f511Serial Entrepreneur" in result
    assert "\U0001f499Wildlife Rescue Volunteer" in result
    assert "\U0001f916Penetration Tester" in result


def test_profile_content_diagnostics_explain_instagram_metadata_only_input():
    diagnostics = inspect_profile_content(INSTAGRAM_HTML)

    assert diagnostics.outcome == "extracted"
    assert diagnostics.response_chars == len(INSTAGRAM_HTML)
    assert diagnostics.saw_markup is True
    assert diagnostics.metadata_fields == (
        "Description",
        "Open Graph title",
        "Open Graph description",
    )
    assert diagnostics.main_content_chars == 0
    assert diagnostics.main_content_method in {"none", "deduplicated"}
    assert diagnostics.prepared_content_chars == len(diagnostics.content)
    assert diagnostics.truncated is False


@pytest.mark.parametrize(
    ("response_text", "outcome"),
    [
        ("", "empty_response"),
        ("<title>404 - Page Not Found</title>", "missing_profile_page"),
        (
            "combo_dump device_id malware_path top_passwords",
            "breach_artifact",
        ),
        ("<html><body></body></html>", "no_extractable_content"),
    ],
)
def test_profile_content_diagnostics_explain_empty_output(
    response_text: str,
    outcome: str,
):
    diagnostics = inspect_profile_content(response_text)

    assert diagnostics.outcome == outcome
    assert diagnostics.content == ""
    assert diagnostics.prepared_content_chars == 0


def test_extract_profile_content_handles_malformed_html():
    html = (
        '<html><head><meta name="description" content="Security researcher">'
        "<body><main>Unclosed profile content"
    )

    result = extract_profile_content(html)

    assert "Security researcher" in result
    assert "Unclosed profile content" in result


def test_extract_profile_content_preserves_plain_text():
    text = "Avery Stone\nSecurity researcher and entrepreneur"

    result = extract_profile_content(text)

    assert text in result
    assert "<" not in result


def test_extract_profile_content_applies_a_profile_first_character_cap():
    text = "profile-start\n" + ("x" * 30_000) + "\nprofile-end"

    result = extract_profile_content(text)

    assert "profile-start" in result
    assert "profile-end" not in result
    assert len(result) <= MAX_PROFILE_CONTENT_CHARS


def test_extract_profile_content_rejects_breach_artifact_payloads():
    response_text = """
    <html><body><pre>{
      "message": "This username appears in an info-stealer result.",
      "stealers": [{
        "stealer_family": "Example",
        "malware_path": "Not Found",
        "top_passwords": ["redacted"],
        "top_logins": ["redacted@example.invalid"]
      }]
    }</pre></body></html>
    """

    assert extract_profile_content(response_text) == ""


def test_extract_profile_content_keeps_security_articles_about_malware():
    response_text = """
    <html><body><main>
      <h1>Alice Example</h1>
      <p>Researches malware analysis and defensive tooling.</p>
    </main></body></html>
    """

    result = extract_profile_content(response_text)

    assert "Alice Example" in result
    assert "malware analysis" in result


@pytest.mark.parametrize(
    "title",
    [
        "User Not Found | Example",
        "Profile has been deleted - Example",
        "404 Page Not Found",
    ],
)
def test_extract_profile_content_rejects_explicit_missing_profile_pages(
    title: str,
):
    response_text = (
        f"<html><head><title>{title}</title></head>"
        "<body><main>Navigation and platform footer</main></body></html>"
    )

    assert extract_profile_content(response_text) == ""


def test_extract_profile_content_keeps_articles_mentioning_missing_profiles():
    response_text = """
    <html><head>
      <title>Why profiles sometimes show user not found</title>
    </head><body><main>
      Alice Example researches account-recovery security.
    </main></body></html>
    """

    result = extract_profile_content(response_text)

    assert "Alice Example" in result


@pytest.mark.parametrize(
    "response_text",
    [
        "",
        "   ",
        "<html><head><script>window.data = {};</script></head><body></body></html>",
    ],
)
def test_extract_profile_content_returns_empty_when_nothing_is_extractable(
    response_text: str,
):
    assert extract_profile_content(response_text) == ""
