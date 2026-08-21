"""S3 discovery and per-video domain enrichment orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, cast

from boto3.session import Session
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from ..config import config
from .enricher import DomainEnricher
from .models import SceneSource, parse_scenes_json, stable_hash
from .repository import DomainRepository


class ProcessStatus(StrEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    EMPTY = "empty"
    INVALID = "invalid"
    ERROR = "error"


class _S3Session(Protocol):
    def client(self, service_name: Literal["s3"]) -> S3Client: ...


@dataclass(frozen=True, slots=True)
class ProcessResult:
    video_id: str
    status: ProcessStatus
    message: str = ""
    segment_count: int = 0


def create_s3_client() -> S3Client:
    session = cast(
        _S3Session,
        Session(
            region_name=config.aws_region,
            aws_access_key_id=config.aws_access_key or None,
            aws_secret_access_key=config.aws_secret_key or None,
        ),
    )
    return session.client("s3")


def _common_prefixes(client: S3Client, bucket: str, prefix: str) -> list[str]:
    pages = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, Delimiter="/"
    )
    result: list[str] = []
    for page in pages:
        for item in page.get("CommonPrefixes", []):
            if common_prefix := item.get("Prefix"):
                result.append(common_prefix)
    return result


def _normalize_key(value: str, bucket: str) -> str:
    value = value.strip()
    if value.startswith("s3://"):
        actual_bucket, separator, key = value[5:].partition("/")
        if not separator or actual_bucket != bucket:
            raise ValueError(f"S3 URI phải thuộc bucket {bucket}: {value}")
        return key
    return value.lstrip("/")


def _head_sources(
    client: S3Client, bucket: str, keys: list[str], workers: int = 16
) -> list[SceneSource]:
    def head(key: str) -> SceneSource | None:
        try:
            response = client.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return None
            raise
        return SceneSource(
            bucket=bucket,
            key=key,
            etag=response.get("ETag", "").strip('"') or None,
            version_id=response.get("VersionId"),
            last_modified=response.get("LastModified"),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sources = [source for source in pool.map(head, sorted(set(keys))) if source]
    return sorted(sources, key=lambda source: source.key)


def normalize_group(value: str) -> str:
    """`L26_b` hoặc `Keyscence_L26_b` hoặc `Keyscence_L26_b/` -> `Keyscence_L26_b/`.

    Nhận cả tên group trần để khớp cách `aic-frame-cut-2026.ipynb` gọi tên (GROUPS =
    ["L26_b", ...]) — chia việc theo người thì hai bên phải nói cùng một thứ tiếng.
    """
    group = value.strip().strip("/")
    if not group:
        raise ValueError("tên group rỗng")
    if not group.startswith("Keyscence_"):
        group = f"Keyscence_{group}"
    return f"{group}/"


def discover_sources(
    client: S3Client,
    bucket: str,
    *,
    prefix: str = "",
    groups: Sequence[str] = (),
    video_id: str = "",
    file_list: str = "",
) -> list[SceneSource]:
    """Discover scenes.json markers without listing scene media objects.

    `groups` nhận nhiều group để một người chạy một lệnh cho cả phần việc của mình,
    thay vì gọi lại lệnh cho từng group. Bỏ trống cả `prefix` lẫn `groups` = toàn bộ
    bucket.
    """
    if file_list:
        values = [
            line.strip()
            for line in Path(file_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        normalized = [_normalize_key(value, bucket) for value in values]
        direct_keys = [value for value in normalized if value.endswith("scenes.json")]
        wanted_ids = {
            value for value in normalized if not value.endswith("scenes.json")
        }
        sources = _head_sources(client, bucket, direct_keys)
        if wanted_ids:
            sources.extend(
                source
                for source in discover_sources(client, bucket)
                if source.video_id in wanted_ids
            )
        return sorted(
            {source.key: source for source in sources}.values(), key=lambda x: x.key
        )

    normalized_prefix = _normalize_key(prefix, bucket) if prefix else ""
    if normalized_prefix.endswith("scenes.json"):
        return _head_sources(client, bucket, [normalized_prefix])
    if "/keyscence/" in normalized_prefix:
        tail = normalized_prefix.rstrip("/").split("/")[-1]
        if tail.startswith("L"):
            key = f"{normalized_prefix.rstrip('/')}/scenes.json"
            return (
                _head_sources(client, bucket, [key])
                if not video_id or tail == video_id
                else []
            )

    if groups:
        # Giữ thứ tự người dùng gõ, bỏ trùng.
        selected = list(dict.fromkeys(normalize_group(value) for value in groups))
    elif normalized_prefix:
        selected = [f"{normalized_prefix.split('/')[0]}/"]
    else:
        selected = [
            value
            for value in _common_prefixes(client, bucket, "")
            if value.startswith("Keyscence_")
        ]

    keys = [
        f"{video_prefix}scenes.json"
        for group in selected
        for video_prefix in _common_prefixes(client, bucket, f"{group}keyscence/")
        if not video_id or video_prefix.rstrip("/").split("/")[-1] == video_id
    ]
    return _head_sources(client, bucket, keys)


def process_source(
    source: SceneSource,
    client: S3Client,
    enricher: DomainEnricher,
    repository: DomainRepository,
    *,
    force: bool = False,
) -> ProcessResult:
    attempt_started = False
    try:
        video_id = source.video_id
        response = client.get_object(Bucket=source.bucket, Key=source.key)
        scenes = parse_scenes_json(response["Body"].read())
        if not scenes:
            return ProcessResult(video_id, ProcessStatus.EMPTY)
        content_hash = stable_hash([scene.model_dump(mode="json") for scene in scenes])
        fingerprint = enricher.inference_fingerprint
        if repository.is_active(
            source.source_id, content_hash, fingerprint, force=force
        ):
            return ProcessResult(video_id, ProcessStatus.SKIPPED)

        repository.begin_attempt(
            source, content_hash, fingerprint, enricher.provider, enricher.model
        )
        attempt_started = True
        analysis = enricher.analyze(video_id, scenes)
        _ = repository.save(
            source=source,
            content_hash=content_hash,
            inference_fingerprint=fingerprint,
            provider=enricher.provider,
            model=enricher.model,
            analysis=analysis,
            scenes=scenes,
        )
        return ProcessResult(
            video_id, ProcessStatus.SUCCESS, segment_count=len(analysis.segments)
        )
    except (TypeError, ValueError) as exc:
        status = ProcessStatus.ERROR if attempt_started else ProcessStatus.INVALID
        message = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - isolate one video in a batch
        status = ProcessStatus.ERROR
        message = f"{type(exc).__name__}: {exc}"

    if attempt_started:
        try:
            repository.fail_attempt(source.source_id, message)
        except Exception as exc:  # noqa: BLE001 - preserve the original error
            message += f"; checkpoint={type(exc).__name__}: {exc}"
    return ProcessResult(source.key, status, message)
