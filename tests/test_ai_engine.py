import asyncio
import json

import pytest

from sherlock_project.ai_config import AISettings
from sherlock_project.ai_engine import (
    AIRequestTrace,
    AIService,
    DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS,
    OSINTResponse,
    PASS_TWO_MAX_OUTPUT_TOKENS,
    PassOneKeyRegistry,
    StructuredResponseError,
    TargetDecision,
    pass_one_contract_hash,
)
from sherlock_project.ai_provider import (
    AICompletion,
    AIGenerationStats,
    AIProviderUnavailableError,
)
from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    InvestigationContext,
    SiteExtraction,
)


pytestmark = pytest.mark.asyncio


def _settings() -> AISettings:
    return AISettings(
        base_url="http://localhost:1234",
        model="example/model",
        temperature=0.1,
    )


def _completion(
    payload: object,
    *,
    native_reasoning: str = "",
    stats: AIGenerationStats | None = None,
) -> AICompletion:
    if (
        isinstance(payload, dict)
        and "extraction" in payload
        and "reasoning" not in payload
    ):
        payload = {
            "reasoning": "The supplied content was evaluated before extraction.",
            **payload,
        }
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return AICompletion(
        final_text=text,
        native_reasoning=native_reasoning,
        stats=stats or AIGenerationStats(
            input_tokens=40,
            output_tokens=20,
            reasoning_tokens=0,
            tokens_per_second=10,
            time_to_first_token_seconds=0.2,
        ),
        elapsed_seconds=2,
    )


def _decision(identity_status: str) -> dict[str, str]:
    return {"identity_status": identity_status}


class FakeProvider:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.settings = _settings()
        self.outcomes = outcomes or []
        self.generate_calls: list[dict[str, object]] = []
        self.ensure_calls = 0
        self.close_calls = 0

    async def list_models(self):
        return []

    async def ensure_model_loaded(self):
        self.ensure_calls += 1
        if self.outcomes and isinstance(self.outcomes[0], asyncio.CancelledError):
            raise self.outcomes.pop(0)
        return object()

    async def generate(self, **kwargs) -> AICompletion:
        self.generate_calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]

    async def close(self) -> None:
        self.close_calls += 1


def _service(
    outcomes: list[object] | None = None,
    *,
    traces: list[AIRequestTrace] | None = None,
) -> tuple[AIService, FakeProvider]:
    provider = FakeProvider(outcomes)
    service = AIService(
        provider=provider,
        settings=provider.settings,
        trace_callback=traces.append if traces is not None else None,
    )
    return service, provider


async def test_create_loads_model_and_closes_provider_on_cancellation():
    provider = FakeProvider([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await AIService.create(provider=provider)

    assert provider.ensure_calls == 1
    assert provider.close_calls == 1


async def test_extract_profile_validates_complete_json_and_emits_trace():
    traces: list[AIRequestTrace] = []
    service, provider = _service(
        [
            _completion(
                {
                    "extraction": {
                        "full_name": [" Jane Doe ", "Jane Doe", "JANE DOE"],
                        "conference_talks": ["Threat Modeling at LakeSec"],
                    },
                }
            )
        ],
        traces=traces,
    )

    response = await service.extract_profile(
        "sample_handle",
        "Example",
        "Jane Doe",
        known_profile_keys=["full_name", "conference_talks"],
    )

    assert response.extraction == {
        "full_name": ["Jane Doe", "JANE DOE"],
        "conference_talks": ["Threat Modeling at LakeSec"],
    }
    assert len(provider.generate_calls) == 1
    call = provider.generate_calls[0]
    system_prompt = str(call["system_prompt"])
    assert "JSON Schema" in system_prompt
    assert '"reasoning"' in system_prompt
    schema = OSINTResponse.model_json_schema()
    reasoning_schema = schema["properties"]["reasoning"]
    assert reasoning_schema["type"] == "string"
    assert "reason concisely through site_content evidence by evidence" in (
        reasoning_schema["description"]
    )
    extraction_schema = schema["properties"]["extraction"]
    assert extraction_schema["additionalProperties"] is False
    assert list(extraction_schema["patternProperties"]) == [
        "^[a-z][a-z0-9_]{0,63}$"
    ]
    assert call["payload"] == {
        "searched_username_do_not_extract": "sample_handle",
        "site_name": "Example",
        "known_profile_keys": ["full_name", "conference_talks"],
        "site_content": "Jane Doe",
    }
    assert call["reasoning_off"] is True
    assert call["max_tokens"] == DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS
    assert traces[0].attempt == 1
    assert traces[0].max_tokens == DEFAULT_STRUCTURED_RESPONSE_MAX_TOKENS
    assert traces[0].structured_reasoning
    assert traces[0].validated_output == response.model_dump(mode="json")
    assert traces[0].stats.reasoning_tokens == 0
    assert traces[0].provider == "lmstudio"
    assert traces[0].context_length == 8192


@pytest.mark.parametrize(
    "final_text",
    [
        "",
        "not-json",
        "{}",
        json.dumps({"extraction": {"Unsafe Key": ["value"]}}),
        json.dumps({"extraction": {"roles": "Researcher"}}),
        json.dumps({"extraction": {"roles": None}}),
        json.dumps({"extraction": {"roles": {"name": "Researcher"}}}),
        json.dumps({"extraction": {"roles": []}}),
        json.dumps({"extraction": {"roles": [""]}}),
        json.dumps({"extraction": {}, "commentary": "extra wrapper field"}),
        json.dumps(
            {
                "reasoning": {"facts": [], "exclusions": []},
                "extraction": {},
            }
        ),
    ],
)
async def test_extract_profile_rejects_invalid_final_response(
    final_text: str,
):
    private_output = final_text
    traces: list[AIRequestTrace] = []
    service, _ = _service([_completion(final_text)], traces=traces)

    with pytest.raises(StructuredResponseError) as error_info:
        await service.extract_profile(
            "sample_handle",
            "Example",
            "profile",
            known_profile_keys=[],
        )

    error = error_info.value
    assert error.validation_error
    if private_output:
        assert private_output not in str(error)
    assert traces[0].final_text == private_output
    assert traces[0].validation_error == error.validation_error
    assert traces[0].validated_output is None


async def test_extract_profile_accepts_empty_extraction():
    service, provider = _service(
        [_completion({"extraction": {}})],
    )

    response = await service.extract_profile(
        "sample_handle",
        "Example",
        "profile",
        known_profile_keys=[],
    )

    assert response.extraction == {}
    assert len(provider.generate_calls) == 1


async def test_pass_one_reasoning_is_not_used_as_a_draft_extraction():
    service, _ = _service(
        [
            _completion(
                {
                    "reasoning": (
                        "The biography contains three roles and an associated "
                        "organization; exclude the searched username and statistics."
                    ),
                    "extraction": {
                        "roles": [
                            "Serial Entrepreneur",
                            "Penetration Tester",
                        ],
                        "organizations": ["@sentinelfoundation"],
                    },
                }
            )
        ]
    )

    response = await service.extract_profile(
        "0day",
        "Instagram",
        (
            "Serial Entrepreneur\nChild Safety Warrior - "
            "(@sentinelfoundation)\nPenetration Tester"
        ),
        known_profile_keys=[],
    )

    assert response.extraction == {
        "roles": [
            "Serial Entrepreneur",
            "Penetration Tester",
        ],
        "organizations": ["@sentinelfoundation"],
    }


async def test_pass_one_keeps_owner_names_and_aliases_except_searched_username():
    service, _ = _service(
        [
            _completion(
                {
                    "extraction": {
                        "display_name": ["Ryan"],
                        "aliases": [
                            "0day",
                            "@0Day",
                            "0 Day",
                            "0-day",
                            "R. Montgomery",
                        ],
                        "other_usernames": ["@rmontgomery"],
                    }
                }
            )
        ]
    )

    response = await service.extract_profile(
        "0day",
        "Example",
        (
            "Profile owner\n"
            "Public name: Ryan\n"
            "Also known as R. Montgomery\n"
            "Uses @rmontgomery"
        ),
        known_profile_keys=[],
    )

    assert response.extraction == {
        "display_name": ["Ryan"],
        "aliases": ["R. Montgomery"],
        "other_usernames": ["@rmontgomery"],
    }


async def test_pass_one_semantic_gate_removes_breach_placeholders_and_hints():
    traces: list[AIRequestTrace] = []
    service, _ = _service(
        [
            _completion(
                {
                    "extraction": {
                        "username": ["0day"],
                        "current_course_id": ["Not Found"],
                        "total_xp": ["Not Found"],
                        "roles": ["Not Found"],
                        "statistics": ["Not Found"],
                        "avatar_url": ["Not Found"],
                        "learning_language": ["German"],
                        "from_language": ["English"],
                        "title": ["kaskuser"],
                        "total_plurks": ["792"],
                        "plurk_responses": ["864"],
                        "referrals": ["2"],
                        "trustscan_items": [
                            "7 Hours, 23 Minutes, 16 Seconds"
                        ],
                        "groups": ["No Followers", "HudsonRock", "Groups"],
                        "miscellaneous": [
                            "0 Threads",
                            "57 Posts",
                            "No awards",
                            "Active 4 minutes ago",
                            "Joined January 2024",
                            "true",
                        ],
                    }
                }
            )
        ],
        traces=traces,
    )

    response = await service.extract_profile(
        "0day",
        "HudsonRock",
        (
            '{"stealers":[{"malware_path":"Not Found",'
            '"top_passwords":["secret"]}]}'
        ),
        known_profile_keys=["learning_language", "current_course_id"],
    )

    assert response.extraction == {}
    assert traces[0].validated_output is not None
    assert traces[0].validated_output["extraction"] == {}


async def test_pass_one_semantic_gate_recovers_owner_name_and_drops_metrics():
    service, _ = _service(
        [
            _completion(
                {
                    "extraction": {
                        "username": ["@0day"],
                        "roles": [
                            "Serial Entrepreneur",
                            "Penetration Tester",
                        ],
                        "state": ["Child Safety Warrior"],
                        "total_followers": ["1M"],
                        "total_following": ["1,049"],
                        "total_posts": ["205"],
                    }
                }
            )
        ]
    )
    site_content = """## Page metadata
- Title: Ryan M. Montgomery (@0day) • Instagram photos and videos
- Description:
  1M Followers, 1,049 Following, 205 Posts
  Ryan M. Montgomery (@0day)
  Serial Entrepreneur
  Child Safety Warrior
  Penetration Tester
- Canonical URL: https://www.instagram.com/0day/

## Main content
Suggested accounts and recent posts
"""

    response = await service.extract_profile(
        "0day",
        "Instagram",
        site_content,
        known_profile_keys=["roles", "state", "total_followers"],
    )

    assert response.extraction == {
        "roles": ["Serial Entrepreneur", "Penetration Tester"],
        "full_name": ["Ryan M. Montgomery"],
    }


async def test_pass_one_semantic_gate_rejects_feed_posts_and_current_url():
    service, _ = _service(
        [
            _completion(
                {
                    "extraction": {
                        "username": ["@0day"],
                        "roles": [
                            "Serial Entrepreneur",
                            "Penetration Tester",
                        ],
                        "interests": ["Child Safety Warrior"],
                        "organizations": ["@sentinelfoundation"],
                        "location": ["Flipper zero"],
                        "publications": [
                            "This is how a Flipper zero can send malicious "
                            "Bluetooth connections to your phone."
                        ],
                        "website": [
                            "https://www.threads.com/@0day",
                            "threads.com",
                        ],
                        "followers": ["102.1K"],
                        "total_threads": ["13"],
                    }
                }
            )
        ]
    )
    site_content = """## Page metadata
- Title: Ryan M. Montgomery (@0day) • Threads, Say more
- Open Graph description:
  102.1K Followers • 13 Threads • Serial Entrepreneur
  Child Safety Warrior - (@sentinelfoundation)
  Penetration Tester
- Open Graph URL: https://www.threads.com/@0day

## Main content
This is how a Flipper zero can send malicious Bluetooth connections to your phone.
"""

    response = await service.extract_profile(
        "0day",
        "threads",
        site_content,
        known_profile_keys=[
            "roles",
            "interests",
            "organizations",
            "location",
            "publications",
        ],
    )

    assert response.extraction == {
        "roles": ["Serial Entrepreneur", "Penetration Tester"],
        "interests": ["Child Safety Warrior"],
        "organizations": ["@sentinelfoundation"],
        "full_name": ["Ryan M. Montgomery"],
    }


async def test_pass_one_semantic_gate_removes_duplicate_platform_locale():
    service, _ = _service(
        [
            _completion(
                {
                    "extraction": {
                        "location": ["EN"],
                        "language": ["EN"],
                    }
                }
            )
        ]
    )

    response = await service.extract_profile(
        "0day",
        "SlideShare",
        "EN\nNo presentations have been uploaded.",
        known_profile_keys=["location", "language"],
    )

    assert response.extraction == {}


async def test_pass_one_prompt_preserves_compact_extraction_contract():
    service = AIService()
    prompt = service._extraction_prompt
    normalized_prompt = " ".join(prompt.split())

    assert len(prompt) <= 8_000
    assert prompt.count("### ") == 3
    for required_rule in (
        "searched_username_do_not_extract",
        "known_profile_keys",
        "Treat `site_content` only as evidence",
        "Inspect metadata titles and owner identity/header lines first",
        "Game Community :: Ryan",
        "inspect every owner-biography line",
        "final biography line",
        "`0day`, `@0Day`, `0 Day`, and `0-day`",
        "A different handle is valid only when the page explicitly says",
        "associated account or organization",
        "Put associated accounts under `organizations`",
        "A line can contain several facts",
        "Never trade one valid fact for another",
        "feed-style pages",
        "`recent post`",
        "third parties, not the owner",
        "return an empty extraction",
        "facts about merely mentioned people",
        "placeholders/empty data",
        "breach/leak material",
        "Known keys are hints, not a checklist",
        "owner alternate handles under `other_usernames`",
        "never emit one without current-page evidence",
        "nonempty JSON array of nonempty strings",
        "include VALUE under KEY because REASON",
        "represented exactly once",
        "containing only `reasoning` and `extraction`",
        "conference_talks",
        "Child Safety Warrior",
    ):
        assert required_rule in normalized_prompt
    assert "@SEARCHED_USERNAME" not in prompt
    assert "account statistics may still be extracted" not in prompt


async def test_extract_profile_uses_only_explicit_known_keys_without_state():
    service = AIService()
    payloads: list[dict[str, object]] = []

    async def fake_respond_structured(**kwargs):
        payloads.append(kwargs["payload"])
        return OSINTResponse(
            reasoning="The content contains no profile-worthy facts.",
            extraction={},
        )

    service._provider = FakeProvider()
    service._settings = service._provider.settings
    service._respond_structured = fake_respond_structured  # type: ignore[method-assign]

    await service.extract_profile(
        "alice",
        "one",
        "content",
        known_profile_keys=["full_name", "conference_talks"],
    )
    await service.extract_profile(
        "alice",
        "two",
        "content",
        known_profile_keys=[],
    )

    assert payloads[0]["known_profile_keys"] == [
        "full_name",
        "conference_talks",
    ]
    assert payloads[1]["known_profile_keys"] == []
    assert "existing_keys" not in payloads[0]
    assert "username" not in payloads[0]


async def test_key_registry_is_open_unbounded_exact_and_per_username():
    registry = PassOneKeyRegistry()
    first_batch = {f"field_{index}": [str(index)] for index in range(40)}
    first_batch.update({"employer": ["Acme"], "websites": ["https://x.test"]})

    registry.seed(
        "alice",
        [first_batch, {"conference_talks": ["LakeSec"], "employer": ["Acme"]}],
    )
    registry.add(
        "alice",
        {"bug_bounty_programs": ["Example"]},
    )
    registry.add("bob", {"full_name": ["Bob Example"]})
    registry.add("alice", {"Bad Key": ["ignored"], "_private": ["ignored"]})

    assert registry.names("alice") == [
        *(f"field_{index}" for index in range(40)),
        "employer",
        "websites",
        "conference_talks",
        "bug_bounty_programs",
    ]
    assert registry.names("bob") == ["full_name"]
    assert registry.names("carol") == []


async def test_pass_one_contract_hash_is_deterministic_and_model_independent(
    monkeypatch,
):
    service, _ = _service()
    other_settings = AISettings(
        base_url="http://localhost:1234",
        model="different/model",
        temperature=0.8,
    )
    other_service = AIService(settings=other_settings)

    expected = pass_one_contract_hash()
    assert service.pass_one_contract_hash == expected
    assert other_service.pass_one_contract_hash == expected
    assert len(expected) == 64

    service._identity_prompt += "\nChanged Pass 2 only."
    assert service.pass_one_contract_hash == expected
    service._extraction_prompt += "\nChanged Pass 1."
    assert service.pass_one_contract_hash != expected

    with monkeypatch.context() as patch:
        patch.setattr(
            "sherlock_project.ai_engine.PASS_ONE_VALIDATION_POLICY_VERSION",
            "changed-for-test",
        )
        assert pass_one_contract_hash() != expected

    with monkeypatch.context() as patch:
        patch.setattr(
            "sherlock_project.ai_engine.PROFILE_CONTENT_EXTRACTION_POLICY_VERSION",
            "changed-for-test",
        )
        assert pass_one_contract_hash() != expected

    with monkeypatch.context() as patch:
        patch.setattr(
            "sherlock_project.ai_engine._compact_model_schema",
            lambda _model: '{"changed":true}',
        )
        assert pass_one_contract_hash() != expected


async def test_pass_two_prompt_allows_semantic_anchor_matches():
    service = AIService()
    schema = TargetDecision.model_json_schema()

    assert "semantic" in service._identity_prompt
    assert "equivalents both count" in service._identity_prompt
    assert "Penetration Tester" in service._identity_prompt
    assert "IT specialist" in service._identity_prompt
    assert "shared searched username" in service._identity_prompt
    assert "discovery context only" in service._identity_prompt
    assert "Use all supplied anchor fields" in service._identity_prompt
    assert "fixed list of identity fields" in service._identity_prompt
    assert "telemetry" in service._identity_prompt
    assert "Think through the evidence" in service._identity_prompt
    assert "internally before returning" in service._identity_prompt
    assert "a reasoning field" in service._identity_prompt
    assert "do not downgrade a clear semantic anchor" in service._identity_prompt
    assert schema["required"] == ["identity_status"]
    assert set(schema["properties"]) == {"identity_status"}
    assert schema["additionalProperties"] is False


async def test_anchorless_synthesis_uses_no_provider_and_collects_values():
    service = AIService()

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=1,
                site_name="One",
                extraction={"full_name": ["Jane Doe"]},
            ),
            SiteExtraction(
                site_id=2,
                site_name="Two",
                extraction={"full_name": ["Jane D."]},
            ),
        ],
        context=InvestigationContext(),
        input_hash="hash",
    )

    assert profile.mode == "aggregate"
    assert profile.strong_profile["full_name"] == ["Jane Doe", "Jane D."]
    assert profile.unsure_profile == {}


async def test_pass_two_uses_native_reasoning_and_status_only_output():
    traces: list[AIRequestTrace] = []
    service, provider = _service(
        [
            _completion(
                {"identity_status": "strong_match"},
                native_reasoning=(
                    "Jane Doe exactly matches the supplied full-name anchor."
                ),
            )
        ],
        traces=traces,
    )

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=1,
                site_name="Example",
                extraction={"full_name": ["Jane Doe"]},
            )
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Jane Doe")]
        ),
        input_hash="hash",
    )

    assert provider.generate_calls[0]["reasoning_off"] is False
    assert provider.generate_calls[0]["max_tokens"] == PASS_TWO_MAX_OUTPUT_TOKENS
    assert profile.source_decisions[0].disposition == "included"
    assert profile.source_decisions[0].identity_status == "strong_match"
    assert traces[0].native_reasoning == (
        "Jane Doe exactly matches the supplied full-name anchor."
    )
    assert traces[0].structured_reasoning == ""
    assert traces[0].validated_output == {"identity_status": "strong_match"}
    assert traces[0].max_tokens == PASS_TWO_MAX_OUTPUT_TOKENS


async def test_anchored_synthesis_uses_model_identity_decisions_unchanged():
    traces: list[AIRequestTrace] = []
    service, provider = _service(
        [
            _completion(
                _decision("strong_match"),
                native_reasoning=(
                    "The current name Jane Doe directly matches the anchor."
                ),
            ),
            _completion(
                _decision("reject"),
                native_reasoning=(
                    "The current name Other Person conflicts with Jane Doe."
                ),
            ),
        ],
        traces=traces,
    )
    context = InvestigationContext(
        anchors=[IdentityAnchor(field="full_name", value="Jane Doe")]
    )

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=2,
                site_name="Other",
                extraction={
                    "full_name": ["Other Person"],
                    "roles": ["Researcher"],
                },
            ),
            SiteExtraction(
                site_id=1,
                site_name="Match",
                extraction={
                    "full_name": ["Jane Doe"],
                    "roles": ["Researcher"],
                },
            ),
            SiteExtraction(site_id=3, site_name="Empty", extraction={}),
        ],
        context=context,
        input_hash="hash",
    )

    assert len(provider.generate_calls) == 2
    assert all(call["reasoning_off"] is False for call in provider.generate_calls)
    assert traces[0].native_reasoning == (
        "The current name Jane Doe directly matches the anchor."
    )
    assert all(trace.structured_reasoning == "" for trace in traces)
    assert profile.strong_profile == {
        "full_name": ["Jane Doe"],
        "roles": ["Researcher"],
    }
    assert profile.unsure_profile == {}
    assert {
        (item.field, item.value): item.source_site_ids
        for item in profile.provenance
    } == {
        ("full_name", "Jane Doe"): [1],
        ("roles", "Researcher"): [1],
    }
    assert [item.disposition for item in profile.source_decisions] == [
        "included",
        "excluded",
        "ignored",
    ]
    assert [item.identity_status for item in profile.source_decisions] == [
        "strong_match",
        "reject",
        None,
    ]
    assert "Other Person" not in profile.strong_profile["full_name"]
    assert all("reasoning" not in item.model_dump() for item in profile.source_decisions)


async def test_model_can_promote_semantic_role_anchor_to_strong_match():
    service, provider = _service(
        [
            _completion(
                _decision("strong_match"),
                native_reasoning=(
                    "Penetration Tester clearly matches the Ethical Hacker role."
                ),
            ),
        ]
    )

    profile = await service.synthesize(
        username="fixture_handle",
        extractions=[
            SiteExtraction(
                site_id=1,
                site_name="Training Profile",
                extraction={
                    "roles": ["Penetration Tester"],
                },
            )
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="role", value="Ethical Hacker")]
        ),
        input_hash="hash",
    )

    assert len(provider.generate_calls) == 1
    assert provider.generate_calls[0]["payload"] == {
        "username": "fixture_handle",
        "anchors": {"roles": ["Ethical Hacker"]},
        "current_site": {
            "site_name": "Training Profile",
            "extraction": {"roles": ["Penetration Tester"]},
        },
    }
    assert profile.strong_profile == {
        "roles": ["Penetration Tester"]
    }
    assert profile.unsure_profile == {}
    assert [
        item.model_dump()
        for item in profile.provenance
    ] == [
        {
            "field": "roles",
            "value": "Penetration Tester",
            "source_site_ids": [1],
            "origins": ["extraction"],
        }
    ]
    assert profile.source_decisions[0].disposition == "included"
    assert profile.source_decisions[0].identity_status == "strong_match"
    assert all(call["reasoning_off"] is False for call in provider.generate_calls)


async def test_pass_two_failure_is_partial_and_later_sites_continue():
    service, provider = _service(
        [
            _completion(_decision("strong_match")),
            _completion("malformed"),
            _completion(_decision("reject")),
        ]
    )

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=index,
                site_name=f"Site {index}",
                extraction={"full_name": [name]},
            )
            for index, name in enumerate(
                ["Jane Doe", "Unknown", "Other Person"],
                start=1,
            )
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Jane Doe")]
        ),
        input_hash="hash",
    )

    assert len(provider.generate_calls) == 3
    assert profile.completeness == "partial"
    assert [item.disposition for item in profile.source_decisions] == [
        "included",
        "failed",
        "excluded",
    ]
    assert profile.strong_profile["full_name"] == ["Jane Doe"]
    warning = next(
        warning for warning in profile.warnings if "site id 2" in warning
    )
    assert "validation=json_invalid" in warning
    assert "final_chars=9" in warning
    assert "malformed" not in warning


async def test_sweep_promotion_rebuilds_disjoint_profiles():
    service, provider = _service(
        [
            _completion(_decision("unsure")),
            _completion(_decision("strong_match")),
            _completion(_decision("strong_match")),
        ]
    )

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=1,
                site_name="Early Handle",
                extraction={
                    "other_usernames": ["@jane_research"],
                    "roles": ["Researcher"],
                },
            ),
            SiteExtraction(
                site_id=2,
                site_name="Direct Name",
                extraction={
                    "full_name": ["Jane Doe"],
                    "other_usernames": ["@jane_research"],
                },
            ),
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Jane Doe")]
        ),
        input_hash="hash",
    )

    assert len(provider.generate_calls) == 3
    assert profile.unsure_profile == {}
    assert profile.strong_profile == {
        "other_usernames": ["@jane_research"],
        "roles": ["Researcher"],
        "full_name": ["Jane Doe"],
    }
    assert {
        (item.field, item.value): item.source_site_ids
        for item in profile.provenance
    } == {
        ("full_name", "Jane Doe"): [2],
        ("other_usernames", "@jane_research"): [1, 2],
        ("roles", "Researcher"): [1],
    }
    assert [
        decision.identity_status for decision in profile.source_decisions
    ] == ["strong_match", "strong_match"]


async def test_provider_wide_failure_marks_remaining_sources_without_more_calls():
    service, provider = _service(
        [AIProviderUnavailableError("provider unavailable")]
    )

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=index,
                site_name=f"Site {index}",
                extraction={"full_name": f"Person {index}"},
            )
            for index in range(1, 4)
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="full_name", value="Jane Doe")]
        ),
        input_hash="hash",
    )

    assert len(provider.generate_calls) == 1
    assert profile.completeness == "partial"
    assert all(
        item.disposition == "failed" for item in profile.source_decisions
    )
    assert len(profile.warnings) == 1


async def test_oversized_pass_two_input_becomes_failed_source(monkeypatch):
    monkeypatch.setattr(
        "sherlock_project.ai_engine.PASS_TWO_MAX_INPUT_BYTES",
        100,
    )
    service, provider = _service([])

    profile = await service.synthesize(
        username="sample_handle",
        extractions=[
            SiteExtraction(
                site_id=1,
                site_name="Large",
                extraction={"bio": "x" * 1000},
            )
        ],
        context=InvestigationContext(
            anchors=[IdentityAnchor(field="roles", value="Researcher")]
        ),
        input_hash="hash",
    )

    assert provider.generate_calls == []
    assert profile.completeness == "partial"
    assert profile.source_decisions[0].disposition == "failed"


async def test_synthesis_fingerprint_tracks_provider_settings_and_schema():
    service, _ = _service()

    fingerprints = service.synthesis_prompt_fingerprints

    assert fingerprints["provider"] == "lmstudio"
    assert fingerprints["native_reasoning"] == "on"
    assert fingerprints["max_output_tokens"] == str(PASS_TWO_MAX_OUTPUT_TOKENS)
    assert fingerprints["temperature"] == "0.1"
    assert fingerprints["context_length"] == "8192"
    assert len(fingerprints["identity"]) == 64
    assert len(fingerprints["target_schema"]) == 64
