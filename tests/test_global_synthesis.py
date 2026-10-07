import json

from sherlock_project.ai_config import DEFAULT_GEMINI_BASE_URL, AISettings
from sherlock_project.ai_engine import (
    PASS_TWO_GLOBAL_MAX_OUTPUT_TOKENS,
    AIService,
    GlobalDecisions,
)
from sherlock_project.ai_provider import (
    AICompletion,
    AIGenerationStats,
    AIQuotaExhaustedError,
)
from sherlock_project.global_synthesis import (
    Verdict,
    chunk_refs,
    deterministic_round,
    site_refs,
)
from sherlock_project.profile_synthesis import (
    IdentityAnchor,
    InvestigationContext,
    SiteExtraction,
)

USERNAME = "sample_handle"


def _cloud_settings() -> AISettings:
    return AISettings(
        provider="gemini",
        base_url=DEFAULT_GEMINI_BASE_URL,
        model="gemini-2.5-flash",
    )


def _local_settings() -> AISettings:
    return AISettings(base_url="http://localhost:8080", model="example/model")


def _completion(payload: object) -> AICompletion:
    return AICompletion(
        final_text=json.dumps(payload),
        native_reasoning="",
        stats=AIGenerationStats(),
        elapsed_seconds=1,
    )


def _decisions(*items: tuple[str, str, list[tuple[str, str]]]) -> AICompletion:
    return _completion(
        {
            "decisions": [
                {
                    "ref": ref,
                    "identity_status": status,
                    "matched": [
                        {"site_value": value, "reference": reference}
                        for value, reference in citations
                    ],
                }
                for ref, status, citations in items
            ]
        }
    )


class ScriptedProvider:
    """Answers each request from a function of its payload."""

    def __init__(self, answer, settings: AISettings | None = None) -> None:
        self.settings = settings or _cloud_settings()
        self.answer = answer
        self.calls: list[dict] = []

    async def list_models(self):
        return []

    async def ensure_model_loaded(self):
        return object()

    async def generate(self, **kwargs) -> AICompletion:
        self.calls.append(kwargs)
        result = self.answer(kwargs["payload"], len(self.calls))
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self) -> None:
        return None


def _service(answer, *, settings: AISettings | None = None):
    provider = ScriptedProvider(answer, settings)
    return AIService(provider=provider, settings=provider.settings), provider


def _site(site_id: int, extraction: dict, url: str | None = None) -> SiteExtraction:
    return SiteExtraction(
        site_id=site_id,
        site_name=f"Site{site_id}",
        site_url=url,
        extraction=extraction,
    )


def _context(**anchors: str) -> InvestigationContext:
    return InvestigationContext(
        anchors=[IdentityAnchor(field=key, value=value) for key, value in anchors.items()]
    )


async def _synthesize(service: AIService, extractions, context):
    return await service.synthesize(
        username=USERNAME,
        extractions=extractions,
        context=context,
        input_hash="hash",
    )


def _status(profile, site_id: int):
    return next(
        decision for decision in profile.source_decisions if decision.site_id == site_id
    )


# -- round 0 ---------------------------------------------------------------


def test_exact_anchor_match_needs_no_model():
    sites = site_refs([_site(1, {"full_name": ["Jane Doe"]})])
    verdicts: dict[str, Verdict] = {}
    deterministic_round(
        username=USERNAME, context=_context(name="jane doe"), sites=sites, verdicts=verdicts
    )
    assert verdicts["s1"].status == "strong_match"
    assert verdicts["s1"].origin == "deterministic"


def test_a_profile_linked_from_a_matched_one_is_matched_and_the_reverse_is_not():
    sites = site_refs(
        [
            _site(1, {"full_name": ["Jane Doe"], "links": ["https://social.example/jd"]}),
            _site(2, {"bio": ["Hiking"]}, url="https://www.social.example/jd/"),
            _site(3, {"links": ["https://site1.example/x"]}),
        ]
    )
    verdicts: dict[str, Verdict] = {}
    deterministic_round(
        username=USERNAME, context=_context(full_name="Jane Doe"), sites=sites, verdicts=verdicts
    )
    assert verdicts["s2"].status == "strong_match"
    assert "s3" not in verdicts


def test_a_shared_email_joins_the_matched_profile():
    sites = site_refs(
        [
            _site(1, {"full_name": ["Jane Doe"], "email": ["JD@example.org"]}),
            _site(2, {"email": ["jd@example.org"]}),
        ]
    )
    verdicts: dict[str, Verdict] = {}
    deterministic_round(
        username=USERNAME, context=_context(full_name="Jane Doe"), sites=sites, verdicts=verdicts
    )
    assert verdicts["s2"].status == "strong_match"


def test_the_searched_username_is_never_evidence():
    sites = site_refs([_site(1, {"usernames": [USERNAME]})])
    verdicts: dict[str, Verdict] = {}
    deterministic_round(
        username=USERNAME, context=_context(usernames=USERNAME), sites=sites, verdicts=verdicts
    )
    assert verdicts == {}


def test_chunks_respect_count_and_size():
    sites = site_refs([_site(index, {"bio": ["x" * 100]}) for index in range(1, 8)])
    chunks = chunk_refs(list(sites), sites, base_payload_bytes=0, max_sites=3)
    assert [len(chunk) for chunk in chunks] == [3, 3, 1]
    chunks = chunk_refs(list(sites), sites, base_payload_bytes=0, max_bytes=300)
    assert all(len(chunk) <= 2 for chunk in chunks)


# -- rounds through AIService ---------------------------------------------


async def test_one_request_decides_every_undecided_site():
    def answer(payload, call):
        refs = [site["ref"] for site in payload["sites"]]
        assert refs == ["s2", "s3"]
        assert "site_name" not in json.dumps(payload["sites"])
        return _decisions(
            ("s2", "unsure", [("Security engineer", "penetration tester")]),
            ("s3", "reject", []),
        )

    service, provider = _service(answer)
    profile = await _synthesize(
        service,
        [
            _site(1, {"full_name": ["Jane Doe"]}),
            _site(2, {"roles": ["Security engineer"]}),
            _site(3, {"full_name": ["Someone Else"]}),
            _site(4, {}),
        ],
        _context(full_name="Jane Doe", roles="penetration tester"),
    )

    # Round 0 settled site 1; round 1 asked once; round 2 had nothing new.
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["reasoning_off"] is False
    assert call["max_tokens"] == PASS_TWO_GLOBAL_MAX_OUTPUT_TOKENS
    assert call["json_schema"] == GlobalDecisions.model_json_schema()
    assert _status(profile, 1).identity_status == "strong_match"
    assert _status(profile, 2).identity_status == "unsure"
    assert _status(profile, 3).disposition == "excluded"
    assert _status(profile, 4).disposition == "ignored"
    assert profile.strong_profile == {"full_name": ["Jane Doe"]}
    assert profile.unsure_profile == {"roles": ["Security engineer"]}


async def test_a_decision_citing_facts_that_do_not_exist_is_downgraded():
    def answer(payload, call):
        return _decisions(
            ("s1", "strong_match", [("Invented Name", "Jane Doe")]),
            ("s2", "unsure", []),
        )

    service, _ = _service(answer)
    profile = await _synthesize(
        service,
        [_site(1, {"roles": ["Baker"]}), _site(2, {"roles": ["Pilot"]})],
        _context(full_name="Jane Doe"),
    )
    assert _status(profile, 1).identity_status == "unsure"
    assert _status(profile, 2).identity_status == "reject"
    assert any("downgraded" in warning for warning in profile.warnings)


async def test_round_two_revisits_a_reject_once_the_evidence_grows():
    def answer(payload, call):
        refs = [site["ref"] for site in payload["sites"]]
        if call == 1:
            assert refs == ["s1", "s2"]
            return _decisions(
                ("s1", "strong_match", [("Pen tester", "penetration tester")]),
                ("s2", "reject", []),
            )
        assert refs == ["s2"]
        assert payload["strong_profile"]["employer"] == ["Cloudflare"]
        return _decisions(("s2", "strong_match", [("Cloudflare", "Cloudflare")]))

    service, provider = _service(answer)
    profile = await _synthesize(
        service,
        [
            _site(1, {"roles": ["Pen tester"], "employer": ["Cloudflare"]}),
            _site(2, {"employer": ["Cloudflare"]}),
        ],
        _context(roles="penetration tester"),
    )
    assert len(provider.calls) == 2
    assert _status(profile, 2).identity_status == "strong_match"


async def test_a_skipped_ref_is_asked_again_on_its_own():
    def answer(payload, call):
        refs = [site["ref"] for site in payload["sites"]]
        if call == 1:
            return _decisions(("s1", "reject", []))
        assert refs == ["s2"]
        return _decisions(("s2", "reject", []))

    service, provider = _service(answer)
    profile = await _synthesize(
        service,
        [_site(1, {"roles": ["Baker"]}), _site(2, {"roles": ["Pilot"]})],
        _context(full_name="Jane Doe"),
    )
    assert len(provider.calls) == 2
    assert _status(profile, 2).disposition == "excluded"


async def test_a_failed_chunk_fails_only_its_sites(monkeypatch):
    monkeypatch.setattr("sherlock_project.ai_engine.chunk_refs", lambda refs, sites, **_: [[ref] for ref in refs])

    def answer(payload, call):
        ref = payload["sites"][0]["ref"]
        if ref == "s1":
            return _completion({"not": "the schema"})
        return _decisions((ref, "reject", []))

    service, _ = _service(answer)
    profile = await _synthesize(
        service,
        [_site(1, {"roles": ["Baker"]}), _site(2, {"roles": ["Pilot"]})],
        _context(full_name="Jane Doe"),
    )
    assert _status(profile, 1).disposition == "failed"
    assert _status(profile, 2).disposition == "excluded"
    assert profile.completeness == "partial"


async def test_exhausted_quota_stops_the_rounds_and_leaves_sites_failed():
    service, provider = _service(lambda payload, call: AIQuotaExhaustedError("quota"))
    profile = await _synthesize(
        service,
        [_site(1, {"roles": ["Baker"]}), _site(2, {"full_name": ["Jane Doe"]})],
        _context(full_name="Jane Doe"),
    )
    assert len(provider.calls) == 1
    assert _status(profile, 1).disposition == "failed"
    assert _status(profile, 2).identity_status == "strong_match"
    assert any("unavailable" in warning for warning in profile.warnings)


# -- strategy selection and caching ---------------------------------------


async def test_local_provider_keeps_the_sequential_sweeps():
    def answer(payload, call):
        assert "current_site" in payload
        return _completion({"identity_status": "reject"})

    service, provider = _service(answer, settings=_local_settings())
    await _synthesize(
        service, [_site(1, {"roles": ["Baker"]})], _context(full_name="Jane Doe")
    )
    assert len(provider.calls) == 1
    assert not service.uses_global_synthesis


def test_strategy_is_fingerprinted_for_the_global_path_only():
    cloud = AIService(settings=_cloud_settings()).synthesis_prompt_fingerprints
    local = AIService(settings=_local_settings()).synthesis_prompt_fingerprints
    assert cloud["strategy"].startswith("global")
    # Adding keys to the local fingerprint would invalidate every cached
    # local profile on upgrade.
    assert set(local) == {
        "identity",
        "target_schema",
        "max_output_tokens",
        "provider",
        "native_reasoning",
        "temperature",
        "context_length",
    }


async def test_unanchored_synthesis_still_needs_no_model():
    service, provider = _service(lambda payload, call: AssertionError("no call"))
    profile = await _synthesize(
        service, [_site(1, {"roles": ["Baker"]})], InvestigationContext()
    )
    assert provider.calls == []
    assert profile.mode == "aggregate"
