"""Pure tests for temporal domain enrichment."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.domain.cerebras import CerebrasDomainEnricher
from ingest.domain.gemini import GeminiDomainEnricher
from ingest.domain.models import (
    TOPICS_BY_DOMAIN,
    Domain,
    SceneSource,
    SegmentationResponse,
    Topic,
    stable_hash,
    validate_scenes,
    validate_segments,
)
from ingest.domain.repository import DomainRepository


def scenes(count: int) -> list[dict]:
    return [
        {
            "scene_id": idx,
            "script": f"scene {idx}",
            "start_frame": idx * 100,
            "end_frame": idx * 100 + 99,
            "start_time": float(idx * 10),
            "end_time": float(idx * 10 + 10),
            "scene_url": f"scene_{idx:03d}.mp4",
        }
        for idx in range(count)
    ]


def segment(
    start: int,
    end: int,
    domain: str,
    sub_domain: str,
    keywords: list[str] | None = None,
    topic_id: str = "other",
) -> dict:
    return {
        "start_scene_id": start,
        "end_scene_id": end,
        "domain": domain,
        "topic_id": topic_id,
        "sub_domain": sub_domain,
        "keywords": keywords or [sub_domain, f"{sub_domain} 2", f"{sub_domain} 3"],
        "summary": f"Tóm tắt {sub_domain}",
    }


def source() -> SceneSource:
    return SceneSource(
        bucket="bucket", key="Keyscence_L21_a/keyscence/L21_V002/scenes.json"
    )


def test_single_domain_normalization() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(
                    0,
                    4,
                    "Ẩm thực - Đời sống",
                    "Món ngon",
                    topic_id="food_cuisine",
                )
            ]
        },
        scenes(5),
    )
    assert not result.is_multi_domain
    assert not result.is_multi_topic
    assert not result.is_multi_segment
    assert result.primary_domain is Domain.FOOD_LIFESTYLE
    assert list(result.segments[0].scene_ids) == [0, 1, 2, 3, 4]


def test_multiple_topics_in_one_domain_are_not_multi_domain() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(0, 1, "Thể thao", "Bóng đá", topic_id="football"),
                segment(2, 4, "Thể thao", "Đua xe đạp", topic_id="cycling"),
            ]
        },
        scenes(5),
    )
    assert result.is_multi_topic
    assert result.is_multi_segment
    assert not result.is_multi_domain
    assert len(result.segments) == 2


def test_repeated_domain_after_another_domain_stays_separate() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(0, 1, "Thể thao", "Bóng đá"),
                segment(2, 3, "Giáo dục", "Dạy bơi"),
                segment(4, 5, "Thể thao", "Đua xe đạp BMX"),
            ]
        },
        scenes(6),
    )
    assert [list(item.scene_ids) for item in result.segments] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]


def test_multiple_stories_can_share_one_topic() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(
                    0,
                    1,
                    "Giao thông - Đô thị",
                    "Tai nạn Hà Nội",
                    topic_id="road_accident",
                ),
                segment(
                    2,
                    3,
                    "Giao thông - Đô thị",
                    "Tai nạn Đà Nẵng",
                    topic_id="road_accident",
                ),
            ]
        },
        scenes(4),
    )
    assert result.is_multi_segment
    assert not result.is_multi_topic
    assert len(result.segments) == 2


def test_adjacent_duplicate_topic_is_merged() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(0, 1, "Thể thao", "Đua xe", [" BMX ", "Xe đạp", "giải đấu"]),
                segment(2, 3, "Thể thao", "đua xe", ["bmx", "Olympic", "giải đấu"]),
            ]
        },
        scenes(4),
    )
    assert len(result.segments) == 1
    assert result.segments[0].keywords == [
        "bmx",
        "xe đạp",
        "giải đấu",
        "olympic",
    ]


def test_topic_must_belong_to_domain() -> None:
    with pytest.raises(ValueError, match="không thuộc domain"):
        validate_segments(
            {
                "segments": [
                    segment(
                        0,
                        1,
                        "Giáo dục",
                        "Đua xe",
                        topic_id="cycling",
                    )
                ]
            },
            scenes(2),
        )


def test_lion_dance_has_one_canonical_cultural_topic() -> None:
    result = validate_segments(
        {
            "segments": [
                segment(
                    0,
                    1,
                    "Văn hóa - Du lịch - Di tích",
                    "Biểu diễn múa lân",
                    topic_id="traditional_performing_arts",
                )
            ]
        },
        scenes(2),
    )

    assert result.segments[0].topic_id is Topic.TRADITIONAL_PERFORMING_ARTS


def test_keywords_are_bounded_for_compact_output() -> None:
    with pytest.raises(ValueError, match="at most 4 items"):
        validate_segments(
            {
                "segments": [
                    segment(
                        0,
                        0,
                        "Ẩm thực - Đời sống",
                        "Nấu ăn",
                        keywords=["một", "hai", "ba", "bốn", "năm"],
                        topic_id="food_cuisine",
                    )
                ]
            },
            scenes(1),
        )


@pytest.mark.parametrize(
    "segments,error",
    [
        ([segment(0, 1, "Thể thao", "A"), segment(3, 3, "Giáo dục", "B")], "bắt đầu"),
        ([segment(0, 2, "Thể thao", "A"), segment(2, 3, "Giáo dục", "B")], "bắt đầu"),
        ([segment(0, 4, "Thể thao", "A")], "vượt scene cuối"),
        ([segment(0, 1, "Thể thao", "A")], "cần phủ"),
    ],
)
def test_invalid_ranges_are_rejected(segments: list[dict], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_segments({"segments": segments}, scenes(4))


def test_scene_contract_is_validated_and_sorted() -> None:
    raw = list(reversed(scenes(3)))
    assert [scene.scene_id for scene in validate_scenes(raw)] == [0, 1, 2]
    raw[0]["scene_id"] = 7
    with pytest.raises(ValueError, match="scene_id"):
        validate_scenes(raw)


def test_pydantic_schema_is_strict_without_custom_cleanup() -> None:
    schema = SegmentationResponse.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ProposedSegment"]["additionalProperties"] is False
    assert schema["$defs"]["Domain"]["enum"] == [domain.value for domain in Domain]
    assert schema["$defs"]["Topic"]["enum"] == [topic.value for topic in Topic]
    assert set(TOPICS_BY_DOMAIN) == set(Domain)


class FakeCompletions:
    def __init__(self, contents: list[str], finish_reason: str = "stop") -> None:
        self.contents = iter(contents)
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="request-id",
            system_fingerprint="test-fingerprint",
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=next(self.contents)),
                )
            ],
            usage=SimpleNamespace(to_dict=lambda: {"total_tokens": 42}),
        )


class FakeGeminiModels:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=self.content,
            response_id="gemini-request-id",
            model_version="gemini-test-version",
            candidates=[SimpleNamespace(finish_reason=None)],
            usage_metadata=SimpleNamespace(
                model_dump=lambda **_kwargs: {"total_token_count": 42}
            ),
        )


def test_cerebras_semantic_retry_repairs_a_gap() -> None:
    invalid = {
        "segments": [
            segment(0, 0, "Thể thao", "A"),
            segment(2, 2, "Giáo dục", "B"),
        ]
    }
    valid = {
        "segments": [
            segment(0, 0, "Thể thao", "A"),
            segment(1, 2, "Giáo dục", "B"),
        ]
    }
    completions = FakeCompletions([json.dumps(invalid), json.dumps(valid)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    enricher = CerebrasDomainEnricher(client=client, semantic_retries=1)

    result = enricher.analyze("L21_V002", scenes(3))

    assert len(completions.calls) == 2
    assert list(result.segments[1].scene_ids) == [1, 2]
    assert result.inference and result.inference.semantic_attempts == 2
    assert completions.calls[0]["seed"] == 0
    assert completions.calls[0]["reasoning_effort"] == "medium"
    assert completions.calls[0]["reasoning_format"] == "hidden"
    assert completions.calls[0]["max_completion_tokens"] == 8192
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    schema = json.dumps(completions.calls[0]["response_format"]["json_schema"])
    assert "minItems" not in schema
    assert "maxItems" not in schema
    assert '"title"' not in schema


def test_cerebras_rejects_truncated_output() -> None:
    content = json.dumps({"segments": [segment(0, 0, "Thể thao", "Đua xe")]})
    completions = FakeCompletions([content], finish_reason="length")
    enricher = CerebrasDomainEnricher(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="gpt-oss-120b",
    )

    with pytest.raises(RuntimeError, match="output bị cắt"):
        enricher.analyze("L21_V002", scenes(1))


def test_gemini_uses_same_schema_and_normalized_output() -> None:
    expected = {"segments": [segment(0, 2, "Thể thao", "Đua xe đạp")]}
    models = FakeGeminiModels(json.dumps(expected))
    enricher = GeminiDomainEnricher(
        client=SimpleNamespace(models=models), model="gemini-3.7-flash"
    )

    result = enricher.analyze("L21_V002", scenes(3))

    assert result.primary_domain is Domain.SPORTS
    assert result.segments[0].domain_id == "sports"
    assert result.segments[0].topic_id is Topic.OTHER
    assert result.inference
    assert result.inference.request_id == "gemini-request-id"
    assert result.inference.model_version == "gemini-test-version"
    assert result.inference.usage == {"total_token_count": 42}
    call = models.calls[0]
    assert call["config"].response_json_schema == (
        SegmentationResponse.model_json_schema()
    )
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].automatic_function_calling.disable is True
    assert call["config"].thinking_config.thinking_level == "MEDIUM"
    assert call["config"].thinking_config.include_thoughts is False
    assert call["config"].max_output_tokens == 8192


def test_provider_is_part_of_inference_fingerprint() -> None:
    content = json.dumps({"segments": [segment(0, 0, "Thể thao", "Đua xe")]})
    cerebras = CerebrasDomainEnricher(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions([content]))
        ),
        model="same-model",
    )
    gemini = GeminiDomainEnricher(
        client=SimpleNamespace(models=FakeGeminiModels(content)), model="same-model"
    )

    assert cerebras.inference_fingerprint != gemini.inference_fingerprint


def test_hash_analysis_id_and_taxonomy_are_stable() -> None:
    digest = stable_hash(scenes(2))
    assert digest == stable_hash(scenes(2))
    analysis_id = DomainRepository.make_analysis_id("source", digest, "prompt", "run")
    assert analysis_id == DomainRepository.make_analysis_id(
        "source", digest, "prompt", "run"
    )
    assert [domain.id for domain in Domain] == [
        "politics_society",
        "economy_finance",
        "agriculture",
        "culture_travel_heritage",
        "science_technology",
        "health",
        "transport_urban",
        "environment_nature",
        "sports",
        "food_lifestyle",
        "law_security",
        "education",
        "general_news",
    ]


def test_failed_rerun_does_not_invalidate_active_analysis() -> None:
    class Jobs:
        def find_one(self, *_args, **_kwargs):
            return {
                "active": {
                    "content_hash": "content",
                    "inference_fingerprint": "fingerprint",
                },
                "last_attempt": {"status": "failed"},
            }

        def update_one(self, _selector, update):
            assert not any(key.startswith("active") for key in update["$set"])

    repository = object.__new__(DomainRepository)
    repository.db = SimpleNamespace(domain_jobs=Jobs())
    assert repository.is_active("source", "content", "fingerprint")
    repository.fail_attempt("source", "failure")


def test_repository_promotes_only_after_interval_mappings_are_written() -> None:
    log: list[tuple[str, str, object]] = []

    class Collection:
        def __init__(self, name: str) -> None:
            self.name = name

        def update_one(self, _selector, update, **_kwargs):
            log.append((self.name, "update", update))

        def bulk_write(self, operations, **_kwargs):
            log.append((self.name, "bulk", operations))

    repository = object.__new__(DomainRepository)
    repository.db = SimpleNamespace(
        domain_analyses=Collection("analyses"),
        scene_domain_map=Collection("scenes"),
        domain_jobs=Collection("jobs"),
    )
    parsed_scenes = validate_scenes(scenes(2))
    analysis = validate_segments(
        {"segments": [segment(0, 1, "Thể thao", "Đua xe đạp")]}, parsed_scenes
    )

    repository.save(
        source=source(),
        content_hash="content",
        inference_fingerprint="fingerprint",
        provider="cerebras",
        model="gpt-oss-120b",
        analysis=analysis,
        scenes=parsed_scenes,
    )

    assert [(name, operation) for name, operation, _ in log] == [
        ("analyses", "update"),
        ("scenes", "bulk"),
        ("analyses", "update"),
        ("jobs", "update"),
    ]
    mappings = log[1][2]
    assert isinstance(mappings, list)
    assert "keyframe_indices" not in mappings[0]._doc["$set"]
    assert "keyframe_urls" not in mappings[0]._doc["$set"]
    assert mappings[0]._doc["$set"]["topic_id"] == "other"
    assert "keywords" in mappings[0]._doc["$set"]
    assert "tags" not in mappings[0]._doc["$set"]
    final = log[-1][2]
    assert final["$set"]["active"]["analysis_id"]
    assert final["$set"]["last_attempt.status"] == "completed"


def test_repository_reads_only_active_interval() -> None:
    class Jobs:
        def find_one(self, *_args, **_kwargs):
            return {"active": {"analysis_id": "active-id"}}

    class Mappings:
        def find_one(self, query, **_kwargs):
            assert query["analysis_id"] == "active-id"
            return {"start_frame": 100, "end_frame": 199, "keywords": ["bmx"]}

    repository = object.__new__(DomainRepository)
    repository.db = SimpleNamespace(domain_jobs=Jobs(), scene_domain_map=Mappings())
    assert repository.find_scene_by_frame("source", 150)["keywords"] == ["bmx"]
    assert repository.find_scene_by_frame("source", -1) is None
