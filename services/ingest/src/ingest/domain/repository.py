"""Versioned MongoDB read model for domain enrichment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TypeAlias, cast

from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.database import Database

from ..config import config
from .models import (
    PROMPT_VERSION,
    TAXONOMY_VERSION,
    Scene,
    SceneSource,
    VideoDomainAnalysis,
    normalize_text,
    stable_hash,
)

MongoDocument: TypeAlias = dict[str, object]


class DomainRepository:
    """Persist immutable analyses and atomically promote one active version."""

    def __init__(self, uri: str | None = None, db_name: str | None = None) -> None:
        self.client: MongoClient[MongoDocument] = MongoClient(
            uri or config.mongo_uri,
            appname="rackfocus-domain-enrichment",
            serverSelectionTimeoutMS=5_000,
        )
        database = db_name or config.mongo_db
        self.db: Database[MongoDocument] = (
            self.client[database]
            if database
            else self.client.get_default_database("rackfocus")
        )
        self._ensure_indexes()

    def close(self) -> None:
        self.client.close()

    def _ensure_indexes(self) -> None:
        _ = self.db.domain_jobs.create_index(
            [("last_attempt.status", ASCENDING), ("source.group", ASCENDING)],
            name="idx_attempt_status_group",
        )
        _ = self.db.domain_jobs.create_index(
            [("external_video_id", ASCENDING)], name="idx_external_video_id"
        )
        _ = self.db.domain_analyses.create_index(
            [("source_id", ASCENDING), ("created_at", DESCENDING)],
            name="idx_source_created",
        )
        _ = self.db.scene_domain_map.create_index(
            [("analysis_id", ASCENDING), ("scene_id", ASCENDING)],
            name="uq_analysis_scene",
            unique=True,
        )
        _ = self.db.scene_domain_map.create_index(
            [("analysis_id", ASCENDING), ("start_frame", DESCENDING)],
            name="idx_analysis_start_frame",
        )
        _ = self.db.scene_domain_map.create_index(
            [("topic_id", ASCENDING), ("analysis_id", ASCENDING)],
            name="idx_topic_analysis",
        )
        _ = self.db.scene_domain_map.create_index(
            [("keywords", ASCENDING), ("topic_id", ASCENDING)],
            name="idx_keywords_topic",
        )

    @staticmethod
    def make_analysis_id(
        source_id: str,
        content_hash: str,
        inference_fingerprint: str,
        discriminator: str,
    ) -> str:
        return stable_hash(
            [source_id, content_hash, inference_fingerprint, discriminator]
        )

    @staticmethod
    def _source_doc(source: SceneSource) -> MongoDocument:
        return cast(MongoDocument, {**source.model_dump(), "group": source.group})

    def is_active(
        self,
        source_id: str,
        content_hash: str,
        inference_fingerprint: str,
        *,
        force: bool = False,
    ) -> bool:
        if force:
            return False
        job = self.db.domain_jobs.find_one(
            {"_id": source_id},
            {"active.content_hash": 1, "active.inference_fingerprint": 1},
        )
        active_value = job.get("active") if job else None
        if not isinstance(active_value, Mapping):
            return False
        active = cast(Mapping[str, object], active_value)
        return (
            active.get("content_hash") == content_hash
            and active.get("inference_fingerprint") == inference_fingerprint
        )

    def begin_attempt(
        self,
        source: SceneSource,
        content_hash: str,
        inference_fingerprint: str,
        provider: str,
        model: str,
    ) -> None:
        now = datetime.now(UTC)
        _ = self.db.domain_jobs.update_one(
            {"_id": source.source_id},
            {
                "$set": {
                    "external_video_id": source.video_id,
                    "source": self._source_doc(source),
                    "last_attempt": {
                        "status": "processing",
                        "content_hash": content_hash,
                        "inference_fingerprint": inference_fingerprint,
                        "provider": provider,
                        "model": model,
                        "started_at": now,
                        "finished_at": None,
                        "error": None,
                    },
                    "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    def fail_attempt(self, source_id: str, error: str) -> None:
        now = datetime.now(UTC)
        _ = self.db.domain_jobs.update_one(
            {"_id": source_id},
            {
                "$set": {
                    "last_attempt.status": "failed",
                    "last_attempt.error": error[:2_000],
                    "last_attempt.finished_at": now,
                    "updated_at": now,
                }
            },
        )

    def save(
        self,
        *,
        source: SceneSource,
        content_hash: str,
        inference_fingerprint: str,
        provider: str,
        model: str,
        analysis: VideoDomainAnalysis,
        scenes: list[Scene],
    ) -> str:
        result_hash = stable_hash(
            analysis.model_dump(exclude={"inference"}, mode="json")
        )
        request_id = analysis.inference.request_id if analysis.inference else None
        analysis_id = self.make_analysis_id(
            source.source_id,
            content_hash,
            inference_fingerprint,
            request_id or result_hash,
        )
        now = datetime.now(UTC)
        source_doc = self._source_doc(source)
        scene_by_id = {scene.scene_id: scene for scene in scenes}

        _ = self.db.domain_analyses.update_one(
            {"_id": analysis_id},
            {
                "$set": {
                    "source_id": source.source_id,
                    "external_video_id": source.video_id,
                    "source": source_doc,
                    "content_hash": content_hash,
                    "inference_fingerprint": inference_fingerprint,
                    "schema_version": 3,
                    "taxonomy_version": TAXONOMY_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "inference": (
                        analysis.inference.model_dump(mode="json")
                        if analysis.inference
                        else None
                    ),
                    "status": "staging",
                    "is_multi_domain": analysis.is_multi_domain,
                    "is_multi_topic": analysis.is_multi_topic,
                    "is_multi_segment": analysis.is_multi_segment,
                    "primary_domain": analysis.primary_domain.value,
                    "primary_domain_id": analysis.primary_domain_id,
                    "segments": [
                        segment.model_dump(mode="json") for segment in analysis.segments
                    ],
                    "num_scenes": len(scenes),
                    "created_at": now,
                    "updated_at": now,
                }
            },
            upsert=True,
        )

        mappings: list[UpdateOne] = []
        for segment in analysis.segments:
            for scene_id in segment.scene_ids:
                scene = scene_by_id[scene_id]
                mappings.append(
                    UpdateOne(
                        {"_id": f"{analysis_id}:scene:{scene_id}"},
                        {
                            "$set": {
                                "analysis_id": analysis_id,
                                "source_id": source.source_id,
                                "external_video_id": source.video_id,
                                "scene_id": scene_id,
                                "segment_idx": segment.segment_idx,
                                "domain": segment.domain.value,
                                "domain_id": segment.domain_id,
                                "topic_id": segment.topic_id.value,
                                "sub_domain": segment.sub_domain,
                                "keywords": segment.keywords,
                                "start_frame": scene.start_frame,
                                "end_frame": scene.end_frame,
                                "start_time": scene.start_time,
                                "end_time": scene.end_time,
                                "scene_url": scene.scene_url,
                                "updated_at": now,
                            }
                        },
                        upsert=True,
                    )
                )
        if mappings:
            _ = self.db.scene_domain_map.bulk_write(mappings, ordered=False)

        completed_at = datetime.now(UTC)
        _ = self.db.domain_analyses.update_one(
            {"_id": analysis_id},
            {"$set": {"status": "completed", "updated_at": completed_at}},
        )
        # Promotion is the only visibility point for readers.
        _ = self.db.domain_jobs.update_one(
            {"_id": source.source_id},
            {
                "$set": {
                    "external_video_id": source.video_id,
                    "source": source_doc,
                    "active": {
                        "analysis_id": analysis_id,
                        "content_hash": content_hash,
                        "inference_fingerprint": inference_fingerprint,
                        "provider": provider,
                        "model": model,
                        "activated_at": completed_at,
                    },
                    "last_attempt.status": "completed",
                    "last_attempt.finished_at": completed_at,
                    "last_attempt.error": None,
                    "is_multi_domain": analysis.is_multi_domain,
                    "is_multi_topic": analysis.is_multi_topic,
                    "is_multi_segment": analysis.is_multi_segment,
                    "primary_domain": analysis.primary_domain.value,
                    "primary_domain_id": analysis.primary_domain_id,
                    "num_scenes": len(scenes),
                    "num_segments": len(analysis.segments),
                    "updated_at": completed_at,
                }
            },
            upsert=True,
        )
        return analysis_id

    def find_scene_by_frame(
        self, source_id: str, frame_idx: int
    ) -> MongoDocument | None:
        if frame_idx < 0:
            return None
        job = self.db.domain_jobs.find_one(
            {"_id": source_id, "active.analysis_id": {"$exists": True}},
            {"active.analysis_id": 1},
        )
        active_value = job.get("active") if job else None
        if not isinstance(active_value, Mapping):
            return None
        active = cast(Mapping[str, object], active_value)
        analysis_id = active.get("analysis_id")
        if not isinstance(analysis_id, str):
            return None
        scene = self.db.scene_domain_map.find_one(
            {"analysis_id": analysis_id, "start_frame": {"$lte": frame_idx}},
            sort=[("start_frame", DESCENDING)],
        )
        end_frame = scene.get("end_frame") if scene else None
        return scene if isinstance(end_frame, int) and end_frame >= frame_idx else None

    def _find_active_mappings(
        self, match: MongoDocument, limit: int
    ) -> list[MongoDocument]:
        if limit < 1:
            return []
        pipeline: list[MongoDocument] = [
            {"$match": match},
            {
                "$lookup": {
                    "from": "domain_jobs",
                    "localField": "source_id",
                    "foreignField": "_id",
                    "as": "job",
                }
            },
            {"$unwind": "$job"},
            {"$match": {"$expr": {"$eq": ["$analysis_id", "$job.active.analysis_id"]}}},
            {"$unset": "job"},
            {"$limit": limit},
        ]
        return list(self.db.scene_domain_map.aggregate(pipeline))

    def active_domain_by_scene(
        self, video_ids: Sequence[str]
    ) -> dict[str, dict[int, str]]:
        """`{external_video_id: {scene_id: domain_id}}` cho analysis ĐANG ACTIVE.

        Dùng để gán tag=domain_id cho keyframe theo `payload.scene_idx` khi build
        `tags.npy` (xem `build_tags.py`) — cùng cơ chế lọc active với
        `_find_active_mappings`/`find_scene_by_frame`, chỉ khác là bulk theo nhiều
        video một lần thay vì single-frame lookup.
        """
        if not video_ids:
            return {}
        pipeline: list[MongoDocument] = [
            {"$match": {"external_video_id": {"$in": list(video_ids)}}},
            {
                "$lookup": {
                    "from": "domain_jobs",
                    "localField": "source_id",
                    "foreignField": "_id",
                    "as": "job",
                }
            },
            {"$unwind": "$job"},
            {"$match": {"$expr": {"$eq": ["$analysis_id", "$job.active.analysis_id"]}}},
            {"$project": {"external_video_id": 1, "scene_id": 1, "domain_id": 1}},
        ]
        result: dict[str, dict[int, str]] = defaultdict(dict)
        for doc in self.db.scene_domain_map.aggregate(pipeline):
            result[cast(str, doc["external_video_id"])][cast(int, doc["scene_id"])] = (
                cast(str, doc["domain_id"])
            )
        return result

    def find_scenes_by_topic(
        self, topic_id: str, limit: int = 50
    ) -> list[MongoDocument]:
        """Find active scene mappings by canonical topic."""
        normalized_topic = topic_id.casefold().strip()
        return (
            self._find_active_mappings({"topic_id": normalized_topic}, limit)
            if normalized_topic
            else []
        )

    def find_scenes_by_keyword(
        self, keyword: str, limit: int = 50
    ) -> list[MongoDocument]:
        """Find active scene mappings by an exact normalized keyword."""
        normalized_keyword = normalize_text(keyword).casefold()
        return (
            self._find_active_mappings({"keywords": normalized_keyword}, limit)
            if normalized_keyword
            else []
        )
