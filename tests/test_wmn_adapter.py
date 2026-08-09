import copy
import json
import os

import pytest
from jsonschema import validate

from sherlock_project.wmn_adapter import (
    NSFW_CATEGORY,
    adapt_wmn_manifest,
    adapt_wmn_site,
    load_wmn_manifest,
    normalize_username,
    preferred_transport,
)

WMN_DATA_RELATIVE = "../sherlock_project/resources/wmn-data.json"
WMN_SCHEMA_RELATIVE = "../sherlock_project/resources/wmn-data.schema.json"


def _resource(relative: str) -> str:
    return os.path.join(os.path.dirname(__file__), relative)


@pytest.fixture(scope="session")
def wmn_raw() -> dict:
    with open(_resource(WMN_DATA_RELATIVE), "r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def wmn_adapted(wmn_raw) -> tuple[dict, list]:
    return adapt_wmn_manifest(wmn_raw)


@pytest.fixture()
def entry() -> dict:
    """A minimal well-formed WMN entry, for mutation by individual tests."""
    return {
        "name": "ExampleSite",
        "uri_check": "https://example.com/user/{account}",
        "e_code": 200,
        "e_string": "profile-header",
        "m_string": "not found",
        "m_code": 404,
        "known": ["alice", "bob"],
        "cat": "social",
    }


class TestVendoredDataset:
    def test_matches_upstream_schema(self, wmn_raw):
        """The vendored copy must stay valid against the vendored schema."""
        with open(_resource(WMN_SCHEMA_RELATIVE), "r", encoding="utf-8") as file:
            schema = json.load(file)
        validate(instance=wmn_raw, schema=schema)

    def test_license_block_intact(self, wmn_raw):
        """CC BY-SA compliance: the license text ships verbatim."""
        assert "license" in wmn_raw
        assert any("Creative Commons" in line for line in wmn_raw["license"])
        assert any("Micah Hoffman" in line for line in wmn_raw["license"])

    def test_loads_from_disk(self):
        sites, rejected = load_wmn_manifest(_resource(WMN_DATA_RELATIVE))
        assert len(sites) > 700
        # Only the single upstream valid=false entry should be dropped.
        assert len(rejected) == 1

    def test_every_site_is_decidable(self, wmn_adapted):
        """No adapted rule may be unable to tell a hit from a miss."""
        sites, _ = wmn_adapted
        for name, record in sites.items():
            exists = record["detection"]["exists"]
            missing = record["detection"]["missing"]
            assert exists["string"] or exists["code"] != missing["code"], (
                f"{name} cannot distinguish a hit from a miss"
            )

    def test_every_site_can_be_probed(self, wmn_adapted):
        sites, _ = wmn_adapted
        for name, record in sites.items():
            has_target = "{}" in record["url"]
            has_payload = "{}" in record.get("request_payload", "")
            assert has_target or has_payload, f"{name} has nowhere to put the username"

    def test_rejections_are_reported_not_raised(self, wmn_adapted):
        _, rejected = wmn_adapted
        assert [r.name for r in rejected] == ["7cup"]
        assert "valid=false" in rejected[0].reason


class TestAdaptSite:
    def test_maps_core_fields(self, entry):
        record = adapt_wmn_site(entry)
        assert record["url"] == "https://example.com/user/{}"
        assert record["urlMain"] == "https://example.com/"
        assert record["username_claimed"] == "alice"
        assert record["known"] == ["alice", "bob"]
        assert record["category"] == "social"
        assert record["request_method"] == "GET"

    def test_builds_two_sided_detection_rule(self, entry):
        record = adapt_wmn_site(entry)
        assert record["detection"] == {
            "exists": {"code": 200, "string": "profile-header"},
            "missing": {"code": 404, "string": "not found"},
        }

    def test_profile_url_defaults_to_check_url(self, entry):
        record = adapt_wmn_site(entry)
        assert record["urlProfile"] == record["url"]

    def test_pretty_url_separates_detection_from_extraction(self, entry):
        entry["uri_check"] = "https://example.com/api/exists?u={account}"
        entry["uri_pretty"] = "https://example.com/@{account}"
        record = adapt_wmn_site(entry)

        assert record["url"] == "https://example.com/api/exists?u={}"
        assert record["urlProfile"] == "https://example.com/@{}"
        # urlMain follows the human-facing URL when there is one.
        assert record["urlMain"] == "https://example.com/"

    def test_post_body_becomes_payload(self, entry):
        entry["uri_check"] = "https://example.com/api/check"
        entry["post_body"] = '{"username":"{account}"}'
        entry["headers"] = {"Content-Type": "application/json"}

        record = adapt_wmn_site(entry)
        assert record["request_method"] == "POST"
        assert record["request_payload"] == '{"username":"{}"}'
        assert record["headers"] == {"Content-Type": "application/json"}

    def test_nsfw_category_becomes_flag(self, entry):
        entry["cat"] = NSFW_CATEGORY
        record = adapt_wmn_site(entry)
        assert record["isNSFW"] is True

    def test_non_nsfw_category(self, entry):
        assert adapt_wmn_site(entry)["isNSFW"] is False

    def test_optional_fields_absent_when_unset(self, entry):
        record = adapt_wmn_site(entry)
        for key in ("headers", "request_payload", "strip_bad_char", "protection"):
            assert key not in record

    def test_protection_is_carried_through(self, entry):
        entry["protection"] = ["cloudflare", "captcha"]
        assert adapt_wmn_site(entry)["protection"] == ["cloudflare", "captcha"]

    def test_code_only_rule_is_allowed_when_codes_differ(self, entry):
        entry["e_string"] = ""
        entry["e_code"] = 302
        entry["m_code"] = 404
        record = adapt_wmn_site(entry)
        assert record["detection"]["exists"]["string"] == ""

    def test_input_entry_is_not_mutated(self, entry):
        before = copy.deepcopy(entry)
        adapt_wmn_site(entry)
        assert entry == before


class TestRejection:
    @pytest.mark.parametrize("field", ["name", "uri_check", "e_code", "known", "cat"])
    def test_missing_required_field(self, entry, field):
        del entry[field]
        with pytest.raises(ValueError, match="missing required field"):
            adapt_wmn_site(entry)

    def test_undecidable_rule(self, entry):
        entry["e_string"] = ""
        entry["m_code"] = entry["e_code"]
        with pytest.raises(ValueError, match="undecidable"):
            adapt_wmn_site(entry)

    def test_no_placeholder_anywhere(self, entry):
        entry["uri_check"] = "https://example.com/user/static"
        with pytest.raises(ValueError, match="no .* placeholder in uri_check"):
            adapt_wmn_site(entry)

    def test_post_body_without_placeholder(self, entry):
        entry["uri_check"] = "https://example.com/api/check"
        entry["post_body"] = '{"username":"fixed"}'
        entry["headers"] = {"Content-Type": "application/json"}
        with pytest.raises(ValueError, match="post_body has no"):
            adapt_wmn_site(entry)

    def test_post_body_without_headers(self, entry):
        entry["post_body"] = '{"username":"{account}"}'
        with pytest.raises(ValueError, match="post_body without headers"):
            adapt_wmn_site(entry)

    def test_valid_false_is_skipped(self, entry):
        entry["valid"] = False
        with pytest.raises(ValueError, match="valid=false"):
            adapt_wmn_site(entry)

    def test_empty_known_list(self, entry):
        entry["known"] = []
        with pytest.raises(ValueError, match="no known usernames"):
            adapt_wmn_site(entry)

    def test_non_integer_status_code(self, entry):
        entry["e_code"] = "200"
        with pytest.raises(ValueError, match="non-integer status code"):
            adapt_wmn_site(entry)

    def test_bad_entry_does_not_sink_the_manifest(self, entry):
        good = dict(entry)
        bad = dict(entry, name="BrokenSite", e_string="", m_code=entry["e_code"])

        sites, rejected = adapt_wmn_manifest({"sites": [good, bad, "not-an-object"]})

        assert list(sites) == ["ExampleSite"]
        assert {r.name for r in rejected} == {"BrokenSite", "<malformed>"}

    def test_duplicate_names_are_rejected(self, entry):
        sites, rejected = adapt_wmn_manifest({"sites": [dict(entry), dict(entry)]})
        assert len(sites) == 1
        assert rejected[0].reason == "duplicate site name"

    def test_manifest_without_sites_array(self):
        with pytest.raises(ValueError, match="no 'sites' array"):
            adapt_wmn_manifest({"license": []})


class TestPreferredTransport:
    """Only a POST may skip the browser.

    A plain HTTP request runs no JavaScript, so a client-rendered profile comes
    back without its marker and a login wall can supply the rule's miss marker
    instead. Threads and Instagram both failed that way while every site was
    being routed to the API.
    """

    def test_get_site_uses_the_browser(self, entry):
        assert preferred_transport(adapt_wmn_site(entry)) == "browser"

    def test_post_site_uses_the_api(self, entry):
        entry["uri_check"] = "https://example.com/api/check"
        entry["post_body"] = '{"username":"{account}"}'
        entry["headers"] = {"Content-Type": "application/json"}
        assert preferred_transport(adapt_wmn_site(entry)) == "api"

    def test_api_style_check_url_still_uses_the_browser(self, entry):
        """uri_pretty no longer routes: it was wrong too often to trade for."""
        entry["uri_check"] = "https://example.com/api/exists?u={account}"
        entry["uri_pretty"] = "https://example.com/@{account}"
        assert preferred_transport(adapt_wmn_site(entry)) == "browser"

    def test_protection_flag_does_not_force_a_transport(self, entry):
        entry["protection"] = ["cloudflare"]
        assert preferred_transport(adapt_wmn_site(entry)) == "browser"

    def test_only_post_sites_leave_the_browser_across_the_dataset(self, wmn_adapted):
        sites, _ = wmn_adapted
        on_api = [n for n, r in sites.items() if preferred_transport(r) == "api"]
        non_post = [n for n in on_api if sites[n]["request_method"] != "POST"]
        assert not non_post, f"non-POST sites bypassing the browser: {non_post}"


class TestNormalizeUsername:
    def test_strips_listed_characters(self):
        assert normalize_username("john.doe", ".") == "johndoe"

    def test_strips_multiple_characters(self):
        assert normalize_username("j.o-hn", "-.") == "john"

    def test_noop_without_rule(self):
        assert normalize_username("john.doe", None) == "john.doe"
        assert normalize_username("john.doe", "") == "john.doe"
