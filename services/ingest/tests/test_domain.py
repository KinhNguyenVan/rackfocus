"""Pure tests for temporal domain enrichment."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.domain.llm import LiteLLMDomainEnricher, split_model
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
from ingest.domain.service import discover_sources, normalize_group


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


class FakeCompletion:
    """Giả `litellm.completion`. Trả đúng shape litellm: choices/usage/id/model."""

    def __init__(self, contents: list[str], finish_reason: str = "stop") -> None:
        self.contents = iter(contents)
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="request-id",
            model="test-model-version",
            system_fingerprint="test-fingerprint",
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=next(self.contents)),
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda **_kwargs: {"total_tokens": 42}
            ),
        )


def test_semantic_retry_repairs_a_gap() -> None:
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
    completion = FakeCompletion([json.dumps(invalid), json.dumps(valid)])
    enricher = LiteLLMDomainEnricher(
        model="cerebras/gpt-oss-120b", semantic_retries=1, completion=completion
    )

    result = enricher.analyze("L21_V002", scenes(3))

    assert len(completion.calls) == 2
    assert list(result.segments[1].scene_ids) == [1, 2]
    assert result.inference and result.inference.semantic_attempts == 2

    call = completion.calls[0]
    assert call["model"] == "cerebras/gpt-oss-120b"
    assert call["seed"] == 0
    assert call["temperature"] == 0
    assert call["reasoning_effort"] == "medium"
    assert call["reasoning_format"] == "hidden"
    assert call["max_tokens"] == 32768
    assert call["response_format"]["type"] == "json_schema"
    # Cerebras strict mode từ chối các annotation này.
    schema = json.dumps(call["response_format"]["json_schema"])
    assert "minItems" not in schema
    assert "maxItems" not in schema
    assert '"title"' not in schema


def test_rejects_truncated_output() -> None:
    content = json.dumps({"segments": [segment(0, 0, "Thể thao", "Đua xe")]})
    enricher = LiteLLMDomainEnricher(
        model="cerebras/gpt-oss-120b",
        completion=FakeCompletion([content], finish_reason="length"),
    )

    with pytest.raises(RuntimeError, match="output bị cắt"):
        enricher.analyze("L21_V002", scenes(1))


def test_gemini_keeps_full_schema_and_maps_thinking() -> None:
    """Chỉ Cerebras cần cắt schema; provider khác giữ nguyên để model được hướng dẫn
    đầy đủ."""
    expected = {"segments": [segment(0, 2, "Thể thao", "Đua xe đạp")]}
    completion = FakeCompletion([json.dumps(expected)])
    enricher = LiteLLMDomainEnricher(
        model="gemini/gemini-3.7-flash", completion=completion
    )

    result = enricher.analyze("L21_V002", scenes(3))

    assert result.primary_domain is Domain.SPORTS
    assert result.segments[0].domain_id == "sports"
    assert result.segments[0].topic_id is Topic.OTHER
    assert result.inference
    assert result.inference.request_id == "request-id"
    assert result.inference.model_version == "test-model-version"
    assert result.inference.usage == {"total_tokens": 42}

    call = completion.calls[0]
    assert call["response_format"]["json_schema"]["schema"] == (
        SegmentationResponse.model_json_schema()
    )
    assert call["reasoning_effort"] == "medium"
    assert "seed" not in call


def test_provider_is_part_of_inference_fingerprint() -> None:
    content = json.dumps({"segments": [segment(0, 0, "Thể thao", "Đua xe")]})
    cerebras = LiteLLMDomainEnricher(
        model="cerebras/same-model", completion=FakeCompletion([content])
    )
    gemini = LiteLLMDomainEnricher(
        model="gemini/same-model", completion=FakeCompletion([content])
    )

    assert cerebras.provider == "cerebras"
    assert gemini.provider == "gemini"
    assert cerebras.inference_fingerprint != gemini.inference_fingerprint


def test_fingerprint_keeps_bare_model_and_legacy_options() -> None:
    """BẢO VỆ CACHE MONGO. `is_active` so `inference_fingerprint`, mà fingerprint gồm
    `provider` + `model` + `inference_options`.

    Nếu model thành "cerebras/gpt-oss-120b" thay vì "gpt-oss-120b", hoặc options khác
    dict mà SDK cũ sinh ra, thì fingerprint đổi cho MỌI video đã tag -> chạy lại toàn
    bộ qua LLM, tốn tiền theo số video. Test này khoá đúng hai thứ đó.
    """
    cerebras = LiteLLMDomainEnricher(
        model="cerebras/gpt-oss-120b", completion=FakeCompletion([])
    )
    assert cerebras.provider == "cerebras"
    assert cerebras.model == "gpt-oss-120b"
    assert cerebras.inference_options == {
        "seed": 0,
        "temperature": 0,
        "reasoning_effort": "medium",
        "reasoning_format": "hidden",
    }

    gemini = LiteLLMDomainEnricher(
        model="gemini/gemini-3.7-flash", completion=FakeCompletion([])
    )
    assert gemini.model == "gemini-3.7-flash"
    assert gemini.inference_options == {
        "thinking_level": "medium",
        "include_thoughts": False,
    }

    # Bản cũ chỉ bật reasoning cho đúng gpt-oss-120b / gemini-3.x.
    other = LiteLLMDomainEnricher(
        model="cerebras/llama-3.3-70b", completion=FakeCompletion([])
    )
    assert other.inference_options == {"seed": 0, "temperature": 0}
    old_gemini = LiteLLMDomainEnricher(
        model="gemini/gemini-2.0-flash", completion=FakeCompletion([])
    )
    assert old_gemini.inference_options == {}


def test_split_model_requires_provider_prefix() -> None:
    assert split_model("cerebras/gpt-oss-120b") == ("cerebras", "gpt-oss-120b")
    for bad in ("gpt-oss-120b", "/model", "provider/", ""):
        with pytest.raises(ValueError, match="provider/model"):
            _ = split_model(bad)


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


def test_active_domain_by_scene_bulk_reads_active_analysis_only() -> None:
    class Mappings:
        def aggregate(self, pipeline):
            assert pipeline[0]["$match"]["external_video_id"]["$in"] == [
                "L21_V001",
                "L21_V002",
            ]
            return [
                {"external_video_id": "L21_V001", "scene_id": 0, "domain_id": "sports"},
                {"external_video_id": "L21_V001", "scene_id": 1, "domain_id": "education"},
                {"external_video_id": "L21_V002", "scene_id": 0, "domain_id": "health"},
            ]

    repository = object.__new__(DomainRepository)
    repository.db = SimpleNamespace(scene_domain_map=Mappings())
    assert repository.active_domain_by_scene(["L21_V001", "L21_V002"]) == {
        "L21_V001": {0: "sports", 1: "education"},
        "L21_V002": {0: "health"},
    }


def test_active_domain_by_scene_empty_input_skips_query() -> None:
    repository = object.__new__(DomainRepository)
    repository.db = SimpleNamespace(scene_domain_map=None)  # would blow up if touched
    assert repository.active_domain_by_scene([]) == {}


# --------------------------- chia việc theo group ---------------------------
class FakeS3:
    """Bucket giả: chỉ cần CommonPrefixes + head_object cho discover_sources."""

    def __init__(self, videos: dict[str, list[str]]) -> None:
        self.videos = videos  # {"Keyscence_L26_b/": ["L26_V001", ...]}
        self.heads: list[str] = []

    def get_paginator(self, _name: str):
        outer = self

        class _P:
            def paginate(self, *, Bucket: str, Prefix: str, Delimiter: str):
                _ = Bucket, Delimiter
                if not Prefix:
                    return [{"CommonPrefixes": [{"Prefix": g} for g in outer.videos]}]
                for group, names in outer.videos.items():
                    if Prefix == f"{group}keyscence/":
                        return [
                            {
                                "CommonPrefixes": [
                                    {"Prefix": f"{Prefix}{n}/"} for n in names
                                ]
                            }
                        ]
                return [{}]

        return _P()

    def head_object(self, *, Bucket: str, Key: str):
        _ = Bucket
        self.heads.append(Key)
        return {"ETag": '"abc"'}


def _bucket() -> FakeS3:
    return FakeS3(
        {
            "Keyscence_L26_b/": ["L26_V001", "L26_V002"],
            "Keyscence_L26_c/": ["L26_V003"],
            "Keyscence_L29_a/": ["L29_V001"],
        }
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("L26_b", "Keyscence_L26_b/"),
        ("Keyscence_L26_b", "Keyscence_L26_b/"),
        ("Keyscence_L26_b/", "Keyscence_L26_b/"),
        ("  L26_b  ", "Keyscence_L26_b/"),
    ],
)
def test_normalize_group_accepts_bare_and_full_names(value: str, expected: str) -> None:
    """Notebook cắt frame gọi group là "L26_b"; CLI phải hiểu cùng tên đó."""
    assert normalize_group(value) == expected


def test_groups_selects_only_requested_groups() -> None:
    s3 = _bucket()
    sources = discover_sources(s3, "bucket", groups=["L26_b", "L26_c"])
    assert [s.key for s in sources] == [
        "Keyscence_L26_b/keyscence/L26_V001/scenes.json",
        "Keyscence_L26_b/keyscence/L26_V002/scenes.json",
        "Keyscence_L26_c/keyscence/L26_V003/scenes.json",
    ]


def test_groups_are_deduped() -> None:
    s3 = _bucket()
    sources = discover_sources(s3, "bucket", groups=["L26_b", "Keyscence_L26_b"])
    assert len(sources) == 2


def test_two_people_with_disjoint_groups_do_not_overlap() -> None:
    """Chia việc theo người: hai danh sách rời nhau thì không ai đụng video của ai."""
    s3 = _bucket()
    a = {s.key for s in discover_sources(s3, "bucket", groups=["L26_b"])}
    b = {s.key for s in discover_sources(s3, "bucket", groups=["L26_c", "L29_a"])}
    assert a and b and not (a & b)


def test_no_scope_walks_whole_bucket() -> None:
    s3 = _bucket()
    assert len(discover_sources(s3, "bucket")) == 4


def test_groups_take_precedence_over_prefix() -> None:
    s3 = _bucket()
    sources = discover_sources(
        s3, "bucket", prefix="Keyscence_L29_a", groups=["L26_c"]
    )
    assert [s.key for s in sources] == [
        "Keyscence_L26_c/keyscence/L26_V003/scenes.json"
    ]
