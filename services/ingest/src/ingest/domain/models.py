"""Domain taxonomy, source contracts, and deterministic validation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

PROMPT_VERSION = "domain-segmentation-v5"
TAXONOMY_VERSION = "vi-video-v3"

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]


class Domain(StrEnum):
    POLITICS_SOCIETY = "Chính trị - Xã hội"
    ECONOMY_FINANCE = "Kinh tế - Tài chính"
    AGRICULTURE = "Nông nghiệp"
    CULTURE_TRAVEL_HERITAGE = "Văn hóa - Du lịch - Di tích"
    SCIENCE_TECHNOLOGY = "Khoa học - Công nghệ"
    HEALTH = "Y tế - Sức khỏe"
    TRANSPORT_URBAN = "Giao thông - Đô thị"
    ENVIRONMENT_NATURE = "Môi trường - Thiên nhiên"
    SPORTS = "Thể thao"
    FOOD_LIFESTYLE = "Ẩm thực - Đời sống"
    LAW_SECURITY = "Pháp luật - An ninh"
    EDUCATION = "Giáo dục"
    GENERAL_NEWS = "Thời sự - Tổng hợp"

    @property
    def id(self) -> str:
        return self.name.lower()


class Topic(StrEnum):
    PUBLIC_POLICY = "public_policy"
    PUBLIC_ADMINISTRATION = "public_administration"
    LABOR_SOCIAL_WELFARE = "labor_social_welfare"
    DIPLOMACY = "diplomacy"
    COMMUNITY_SOCIAL_ISSUES = "community_social_issues"

    BANKING_INTEREST_RATES = "banking_interest_rates"
    REAL_ESTATE = "real_estate"
    TRADE_EXPORTS = "trade_exports"
    BUSINESS_MARKETS = "business_markets"
    CONSUMER_PRICES = "consumer_prices"

    CROP_FARMING = "crop_farming"
    LIVESTOCK_AQUACULTURE = "livestock_aquaculture"
    AGRICULTURAL_EXPORTS = "agricultural_exports"
    RURAL_DEVELOPMENT = "rural_development"

    HERITAGE_HISTORICAL_SITES = "heritage_historical_sites"
    TOURISM_DESTINATIONS = "tourism_destinations"
    ARTS_ENTERTAINMENT = "arts_entertainment"
    FESTIVALS_TRADITIONS = "festivals_traditions"
    TRADITIONAL_PERFORMING_ARTS = "traditional_performing_arts"

    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    ROBOTICS_AUTOMATION = "robotics_automation"
    BIOTECHNOLOGY = "biotechnology"
    SPACE_RESEARCH = "space_research"
    DIGITAL_TECHNOLOGY = "digital_technology"

    DISEASE_PREVENTION = "disease_prevention"
    HEALTHCARE_SERVICES = "healthcare_services"
    PHARMACEUTICALS_MEDICAL_DEVICES = "pharmaceuticals_medical_devices"
    PUBLIC_HEALTH = "public_health"
    NUTRITION_WELLNESS = "nutrition_wellness"

    ROAD_ACCIDENT = "road_accident"
    TRAFFIC_VIOLATION = "traffic_violation"
    PUBLIC_TRANSPORT = "public_transport"
    TRANSPORT_INFRASTRUCTURE = "transport_infrastructure"
    TRAFFIC_CONGESTION = "traffic_congestion"

    EXTREME_WEATHER = "extreme_weather"
    NATURAL_DISASTER = "natural_disaster"
    CONSERVATION_WILDLIFE = "conservation_wildlife"
    POLLUTION_WASTE = "pollution_waste"
    CLIMATE_ENVIRONMENT = "climate_environment"

    FOOTBALL = "football"
    CYCLING = "cycling"
    COMBAT_SPORTS = "combat_sports"
    ATHLETICS = "athletics"
    OTHER_SPORTS = "other_sports"

    FOOD_CUISINE = "food_cuisine"
    CONSUMER_LIFESTYLE = "consumer_lifestyle"
    FAMILY_DAILY_LIFE = "family_daily_life"
    FASHION_BEAUTY = "fashion_beauty"

    CRIME_INVESTIGATION = "crime_investigation"
    COURT_JUSTICE = "court_justice"
    LAW_ENFORCEMENT = "law_enforcement"
    PUBLIC_SECURITY = "public_security"

    SCHOOLS_EDUCATION = "schools_education"
    EXAMS_ADMISSIONS = "exams_admissions"
    SKILLS_TRAINING = "skills_training"
    EDUCATION_POLICY = "education_policy"

    BREAKING_NEWS = "breaking_news"
    MIXED_NEWS_DIGEST = "mixed_news_digest"
    HUMAN_INTEREST = "human_interest"
    OTHER = "other"


TOPICS_BY_DOMAIN: dict[Domain, tuple[Topic, ...]] = {
    Domain.POLITICS_SOCIETY: (
        Topic.PUBLIC_POLICY,
        Topic.PUBLIC_ADMINISTRATION,
        Topic.LABOR_SOCIAL_WELFARE,
        Topic.DIPLOMACY,
        Topic.COMMUNITY_SOCIAL_ISSUES,
        Topic.OTHER,
    ),
    Domain.ECONOMY_FINANCE: (
        Topic.BANKING_INTEREST_RATES,
        Topic.REAL_ESTATE,
        Topic.TRADE_EXPORTS,
        Topic.BUSINESS_MARKETS,
        Topic.CONSUMER_PRICES,
        Topic.OTHER,
    ),
    Domain.AGRICULTURE: (
        Topic.CROP_FARMING,
        Topic.LIVESTOCK_AQUACULTURE,
        Topic.AGRICULTURAL_EXPORTS,
        Topic.RURAL_DEVELOPMENT,
        Topic.OTHER,
    ),
    Domain.CULTURE_TRAVEL_HERITAGE: (
        Topic.HERITAGE_HISTORICAL_SITES,
        Topic.TOURISM_DESTINATIONS,
        Topic.ARTS_ENTERTAINMENT,
        Topic.FESTIVALS_TRADITIONS,
        Topic.TRADITIONAL_PERFORMING_ARTS,
        Topic.OTHER,
    ),
    Domain.SCIENCE_TECHNOLOGY: (
        Topic.ARTIFICIAL_INTELLIGENCE,
        Topic.ROBOTICS_AUTOMATION,
        Topic.BIOTECHNOLOGY,
        Topic.SPACE_RESEARCH,
        Topic.DIGITAL_TECHNOLOGY,
        Topic.OTHER,
    ),
    Domain.HEALTH: (
        Topic.DISEASE_PREVENTION,
        Topic.HEALTHCARE_SERVICES,
        Topic.PHARMACEUTICALS_MEDICAL_DEVICES,
        Topic.PUBLIC_HEALTH,
        Topic.NUTRITION_WELLNESS,
        Topic.OTHER,
    ),
    Domain.TRANSPORT_URBAN: (
        Topic.ROAD_ACCIDENT,
        Topic.TRAFFIC_VIOLATION,
        Topic.PUBLIC_TRANSPORT,
        Topic.TRANSPORT_INFRASTRUCTURE,
        Topic.TRAFFIC_CONGESTION,
        Topic.OTHER,
    ),
    Domain.ENVIRONMENT_NATURE: (
        Topic.EXTREME_WEATHER,
        Topic.NATURAL_DISASTER,
        Topic.CONSERVATION_WILDLIFE,
        Topic.POLLUTION_WASTE,
        Topic.CLIMATE_ENVIRONMENT,
        Topic.OTHER,
    ),
    Domain.SPORTS: (
        Topic.FOOTBALL,
        Topic.CYCLING,
        Topic.COMBAT_SPORTS,
        Topic.ATHLETICS,
        Topic.OTHER_SPORTS,
        Topic.OTHER,
    ),
    Domain.FOOD_LIFESTYLE: (
        Topic.FOOD_CUISINE,
        Topic.CONSUMER_LIFESTYLE,
        Topic.FAMILY_DAILY_LIFE,
        Topic.FASHION_BEAUTY,
        Topic.OTHER,
    ),
    Domain.LAW_SECURITY: (
        Topic.CRIME_INVESTIGATION,
        Topic.COURT_JUSTICE,
        Topic.LAW_ENFORCEMENT,
        Topic.PUBLIC_SECURITY,
        Topic.OTHER,
    ),
    Domain.EDUCATION: (
        Topic.SCHOOLS_EDUCATION,
        Topic.EXAMS_ADMISSIONS,
        Topic.SKILLS_TRAINING,
        Topic.EDUCATION_POLICY,
        Topic.OTHER,
    ),
    Domain.GENERAL_NEWS: (
        Topic.BREAKING_NEWS,
        Topic.MIXED_NEWS_DIGEST,
        Topic.HUMAN_INTEREST,
        Topic.OTHER,
    ),
}


class Scene(BaseModel):
    """Fields consumed from the stable scenes.json contract."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    scene_id: int = Field(ge=0)
    script: str = ""
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    scene_url: str | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> Scene:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame phải >= start_frame")
        if self.end_time < self.start_time:
            raise ValueError("end_time phải >= start_time")
        return self


class SceneSource(BaseModel):
    """Identity and S3 version metadata for one scenes.json."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    bucket: str
    key: str
    etag: str | None = None
    version_id: str | None = None
    last_modified: datetime | None = None

    @property
    def source_id(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    @property
    def group(self) -> str:
        return self.key.split("/", 1)[0]

    @property
    def video_id(self) -> str:
        parts = self.key.split("/")
        if len(parts) != 4 or parts[1] != "keyscence" or parts[-1] != "scenes.json":
            raise ValueError(f"S3 key không đúng format scenes.json: {self.key}")
        return parts[2]


class ProposedSegment(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    start_scene_id: int = Field(ge=0)
    end_scene_id: int = Field(ge=0)
    domain: Domain
    topic_id: Topic
    sub_domain: str
    keywords: list[str] = Field(min_length=3, max_length=4)
    summary: str


class SegmentationResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    segments: list[ProposedSegment]


class DomainSegment(ProposedSegment):
    segment_idx: int
    domain_id: str

    @property
    def scene_ids(self) -> range:
        return range(self.start_scene_id, self.end_scene_id + 1)


class InferenceMetadata(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    request_id: str | None = None
    system_fingerprint: str | None = None
    model_version: str | None = None
    usage: JsonObject = Field(default_factory=dict)
    semantic_attempts: int = 1


class VideoDomainAnalysis(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    is_multi_domain: bool
    is_multi_topic: bool
    is_multi_segment: bool
    primary_domain: Domain
    primary_domain_id: str
    segments: list[DomainSegment]
    inference: InferenceMetadata | None = None


_SCENES_JSON = TypeAdapter(list[Scene])


def stable_hash(data: object) -> str:
    payload = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _normalize_keywords(keywords: list[str]) -> list[str]:
    unique = dict.fromkeys(
        keyword for raw in keywords if (keyword := normalize_text(raw).casefold())
    )
    return list(unique)[:4]


def validate_scenes(
    raw_scenes: Sequence[Mapping[str, object] | Scene],
) -> list[Scene]:
    scenes = sorted(
        (
            scene if isinstance(scene, Scene) else Scene.model_validate(scene)
            for scene in raw_scenes
        ),
        key=lambda scene: scene.scene_id,
    )
    ids = [scene.scene_id for scene in scenes]
    if ids != list(range(len(scenes))):
        raise ValueError(f"scene_id phải liên tục từ 0..N-1, nhận {ids[:10]}...")
    return scenes


def parse_scenes_json(data: bytes | str) -> list[Scene]:
    return validate_scenes(_SCENES_JSON.validate_json(data))


def validate_segments(
    proposal: SegmentationResponse | Mapping[str, object],
    raw_scenes: Sequence[Mapping[str, object] | Scene],
) -> VideoDomainAnalysis:
    """Validate exact temporal coverage and derive all video-level fields."""
    scenes = validate_scenes(raw_scenes)
    if not scenes:
        return VideoDomainAnalysis(
            is_multi_domain=False,
            is_multi_topic=False,
            is_multi_segment=False,
            primary_domain=Domain.GENERAL_NEWS,
            primary_domain_id=Domain.GENERAL_NEWS.id,
            segments=[],
        )

    parsed = (
        proposal
        if isinstance(proposal, SegmentationResponse)
        else SegmentationResponse.model_validate(proposal)
    )
    proposed = sorted(parsed.segments, key=lambda segment: segment.start_scene_id)
    if not proposed:
        raise ValueError("Provider không trả segment nào")

    expected_start = 0
    normalized: list[DomainSegment] = []
    for raw in proposed:
        if raw.start_scene_id != expected_start:
            raise ValueError(
                f"segment phải bắt đầu ở scene {expected_start}, nhận {raw.start_scene_id}"
            )
        if raw.end_scene_id < raw.start_scene_id:
            raise ValueError(
                f"segment {raw.start_scene_id}..{raw.end_scene_id} có range ngược"
            )
        if raw.end_scene_id >= len(scenes):
            raise ValueError(
                f"end_scene_id {raw.end_scene_id} vượt scene cuối {len(scenes) - 1}"
            )

        if raw.topic_id not in TOPICS_BY_DOMAIN[raw.domain]:
            raise ValueError(
                f"topic_id {raw.topic_id.value} không thuộc domain {raw.domain.value}"
            )

        sub_domain = normalize_text(raw.sub_domain)
        summary = normalize_text(raw.summary)
        keywords = _normalize_keywords(raw.keywords)
        if not sub_domain or not summary:
            raise ValueError("sub_domain và summary không được rỗng")
        if len(keywords) < 3:
            raise ValueError("keywords phải có ít nhất 3 giá trị khác nhau")
        segment = DomainSegment(
            segment_idx=len(normalized),
            start_scene_id=raw.start_scene_id,
            end_scene_id=raw.end_scene_id,
            domain=raw.domain,
            domain_id=raw.domain.id,
            topic_id=raw.topic_id,
            sub_domain=sub_domain,
            keywords=keywords,
            summary=summary,
        )
        previous = normalized[-1] if normalized else None
        if (
            previous
            and previous.domain == segment.domain
            and previous.topic_id == segment.topic_id
            and previous.sub_domain.casefold() == segment.sub_domain.casefold()
        ):
            previous.end_scene_id = segment.end_scene_id
            previous.keywords = _normalize_keywords(
                previous.keywords + segment.keywords
            )
            if segment.summary and segment.summary not in previous.summary:
                previous.summary = normalize_text(
                    f"{previous.summary} {segment.summary}"
                )
        else:
            normalized.append(segment)
        expected_start = raw.end_scene_id + 1

    if expected_start != len(scenes):
        raise ValueError(
            f"segment chỉ phủ đến scene {expected_start - 1}, cần phủ đến {len(scenes) - 1}"
        )
    for idx, segment in enumerate(normalized):
        segment.segment_idx = idx

    durations: dict[Domain, float] = defaultdict(float)
    for segment in normalized:
        start = scenes[segment.start_scene_id].start_time
        end = scenes[segment.end_scene_id].end_time
        durations[segment.domain] += max(end - start, 0.001)
    primary = max(durations, key=durations.__getitem__)
    topics = {(segment.domain, segment.topic_id) for segment in normalized}
    return VideoDomainAnalysis(
        is_multi_domain=len(durations) > 1,
        is_multi_topic=len(topics) > 1,
        is_multi_segment=len(normalized) > 1,
        primary_domain=primary,
        primary_domain_id=primary.id,
        segments=normalized,
    )
