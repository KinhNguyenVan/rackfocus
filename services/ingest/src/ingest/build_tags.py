"""Gán tag=domain_id cho snapshot đã build: đọc payload.parquet + map keyframe->scene đã
validate + MongoDB (domain enrichment), ghi tags.npy + tag_vocab.json vào snapshot_dir,
rồi đẩy lại lên SNAPSHOT_S3 để core thấy ngay khi tải snapshot lần đầu.

Chạy SAU khi `aic-embed-siglip-2026.ipynb` đã ghi snapshot (payload.parquet, idmap.npy,
manifest.json) và SAU khi `AIC_KeyframeSceneMap.ipynb` đã ghi
`Maps_<group>/maps/keyframe_scene_map.json` cho các group liên quan. Script này chỉ ĐỌC,
không đổi vector/idmap/faiss, nên giữ đúng row order đã có.

Join qua `(video_name, frame)` -> `Maps_*/maps/keyframe_scene_map.json` để lấy `scene_id`,
KHÔNG dùng thẳng cột `payload.scene_idx`: cột đó do `assign_scene_idx` trong notebook embed
tính bằng một vòng lặp không có bước validate nào, còn `AIC_KeyframeSceneMap.ipynb` bisect +
kiểm chéo theo cả frame lẫn timestamp, và assert `nearest == 0`/`unmatched == 0` trước khi
ghi file — đáng tin hơn để làm nguồn cho tag.

`core` không tự map domain->frame lúc chạy: nó chỉ ĐỌC `tags.npy` (đã map sẵn 1-1 theo
row) lúc load snapshot, không query Mongo. Nên chuỗi "domain -> scene -> frame" ba chặng
(qua map keyframe->scene rồi `DomainRepository.active_domain_by_scene`) phải chạy XONG, và
kết quả phải NẰM TRÊN S3 (`--upload-s3`), TRƯỚC khi core khởi động lần đầu với version đó.

    python -m ingest.build_tags --snapshot-dir /path/to/snapshots/v1 \\
        --maps-dir /path/to/maps --upload-s3 s3://bucket/snapshots/v1
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq

from . import storage
from .domain.models import Domain
from .domain.repository import DomainRepository

TAGS_FILE = "tags.npy"
TAG_VOCAB_FILE = "tag_vocab.json"
MANIFEST_FILE = "manifest.json"
PAYLOAD_FILE = "payload.parquet"
TAGS_SENTINEL = 65535  # khớp searchcore/snapshot.py::TAGS_SENTINEL

# Thứ tự cố định theo định nghĩa Domain — tag_id không đổi giữa các lần chạy
# miễn Domain enum không đổi thứ tự (đổi thứ tự = đổi tag_id của toàn corpus).
TAG_ORDER: list[Domain] = list(Domain)
TAG_ID_BY_DOMAIN: dict[str, int] = {d.id: i for i, d in enumerate(TAG_ORDER)}


def build_tag_vocab() -> dict[str, dict[str, str]]:
    return {
        str(tag_id): {"name": domain.id, "description": domain.value}
        for tag_id, domain in enumerate(TAG_ORDER)
    }


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_frame_to_scene(
    maps_dir: str, groups: list[str] | None = None
) -> dict[tuple[str, int], int]:
    """`{(video_id, frame): scene_id}` từ `Maps_<group>/maps/keyframe_scene_map.json`
    (do `AIC_KeyframeSceneMap.ipynb` sinh — operator tự tải/sync các thư mục `Maps_*`
    xuống `maps_dir` từ S3 trước khi chạy script này, giống cách `--snapshot-dir` cũng
    là một bản đã tải sẵn cục bộ).

    Bỏ qua keyframe có `scene_id: null` (video đó không có scene nào khớp — notebook map
    đã tự assert `unmatched == 0` trước khi ghi file, nên trường hợp còn lại chỉ là video
    chưa được notebook map đó xử lý tới, không phải lỗi).
    """
    pattern = os.path.join(maps_dir, "Maps_*", "maps", "keyframe_scene_map.json")
    paths = sorted(glob.glob(pattern))
    if groups:
        wanted = set(groups)
        paths = [
            p for p in paths
            if os.path.basename(os.path.dirname(os.path.dirname(p)))[len("Maps_"):]
            in wanted
        ]
    if not paths:
        raise RuntimeError(
            f"Không thấy Maps_*/maps/keyframe_scene_map.json nào dưới {maps_dir} "
            f"(groups={groups or 'tất cả'}) — chạy AIC_KeyframeSceneMap.ipynb trước."
        )

    out: dict[tuple[str, int], int] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for video_id, vmap in doc["videos"].items():
            for kf in vmap["keyframes"]:
                if kf["scene_id"] is not None:
                    out[(video_id, int(kf["frame"]))] = int(kf["scene_id"])
    return out


def compute_tags(
    video_names: list[str],
    frames: list[int],
    frame_to_scene: dict[tuple[str, int], int],
    repository: DomainRepository,
) -> np.ndarray:
    """uint16[N] tag_id theo domain_id đang active của scene chứa keyframe đó.

    Ba chặng mỗi row: (video_name, frame) -> `frame_to_scene` (map đã validate, xem
    `load_frame_to_scene`) -> scene_id -> `repository.active_domain_by_scene` -> domain_id
    -> tag_id. Keyframe không có trong map (video chưa map, hoặc map trả None) -> sentinel.
    """
    unique_videos = sorted(set(video_names))
    domain_by_scene = repository.active_domain_by_scene(unique_videos)

    tags = np.full(len(video_names), TAGS_SENTINEL, dtype=np.uint16)
    for row, (video_name, frame) in enumerate(zip(video_names, frames, strict=True)):
        scene_id = frame_to_scene.get((video_name, frame))
        if scene_id is None:
            continue
        domain_id = domain_by_scene.get(video_name, {}).get(scene_id)
        if domain_id is None:
            continue
        tag_id = TAG_ID_BY_DOMAIN.get(domain_id)
        if tag_id is None:
            raise ValueError(
                f"domain_id {domain_id!r} (video={video_name}, scene={scene_id}) "
                "không nằm trong Domain enum hiện tại — taxonomy đã đổi từ lúc "
                "phân tích, cần chạy lại domain enrichment trước khi build tag."
            )
        tags[row] = tag_id
    return tags


def _print_coverage(tags: np.ndarray) -> None:
    total = len(tags)
    assigned = tags[tags != TAGS_SENTINEL]
    covered = len(assigned)
    print(f"tag coverage: {covered}/{total} ({covered / total:.1%})" if total else "0 row")
    if not covered:
        return
    counts = np.bincount(assigned.astype(np.int64), minlength=len(TAG_ORDER))
    for tag_id, count in enumerate(counts):
        if count:
            print(f"  [{tag_id:>2}] {TAG_ORDER[tag_id].id:<24} {int(count):>8} "
                  f"({int(count) / covered:.1%})")


def _upload_to_s3(snapshot_dir: str, s3_uri: str, names: list[str]) -> None:
    """Đẩy lại đúng những file vừa đổi lên S3, KHÔNG re-upload cả snapshot (vài GB).

    Dùng bucket/prefix nêu trong chính `s3_uri` (giống `storage.upload_dir`) thay vì
    `storage.upload_file` — hàm đó khoá cứng bucket theo env AWS_BUCKET_NAME, sai chỗ nếu
    SNAPSHOT_S3 trỏ bucket khác. `core` chỉ đọc key mới ở lần LOAD ĐẦU của bản snapshot
    này (`download_dir` cache theo version, xem docs/search-design.md §7) — upload đây
    phải chạy TRƯỚC khi core start lần đầu với version đó, không phải patch một bản đang
    live.
    """
    bucket, prefix = storage.parse_s3_uri(s3_uri)
    client = storage.get_client()
    for name in names:
        key = f"{prefix}/{name}" if prefix else name
        client.upload_file(os.path.join(snapshot_dir, name), bucket, key)
        print(f"upload s3://{bucket}/{key}")


def _update_manifest_checksums(snapshot_dir: str, written: list[str]) -> None:
    manifest_path = os.path.join(snapshot_dir, MANIFEST_FILE)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    checksums = manifest.setdefault("checksums", {})
    for name in written:
        checksums[name] = _sha256(os.path.join(snapshot_dir, name))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sinh tags.npy + tag_vocab.json (tag=domain_id) cho một snapshot đã build."
    )
    _ = parser.add_argument("--snapshot-dir", required=True)
    _ = parser.add_argument(
        "--maps-dir", required=True,
        help="Thư mục cục bộ chứa Maps_<group>/maps/keyframe_scene_map.json "
        "(do AIC_KeyframeSceneMap.ipynb sinh, đã tải/sync về từ S3).",
    )
    _ = parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Chỉ đọc map của các group này (mặc định: mọi Maps_* tìm thấy dưới --maps-dir).",
    )
    _ = parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Không cập nhật manifest.checksums (chỉ ghi hai file, tự cập nhật thủ công sau).",
    )
    _ = parser.add_argument(
        "--upload-s3",
        default=os.environ.get("SNAPSHOT_S3", ""),
        help="s3://bucket/prefix — đẩy lại tags.npy/tag_vocab.json/manifest.json ngay sau "
        "khi ghi, để core thấy đủ 3 file khi tải snapshot lần đầu. Mặc định lấy env "
        "SNAPSHOT_S3 (đã dùng chung .env với core); truyền rỗng để chỉ ghi cục bộ.",
    )
    args = parser.parse_args()

    payload_path = os.path.join(args.snapshot_dir, PAYLOAD_FILE)
    table = pq.read_table(payload_path, columns=["video_name", "frame"])
    video_names = table.column("video_name").to_pylist()
    frames = table.column("frame").to_pylist()
    print(f"{len(video_names)} row, {len(set(video_names))} video")

    frame_to_scene = load_frame_to_scene(args.maps_dir, args.groups)
    print(f"{len(frame_to_scene)} keyframe trong map đã validate ({args.maps_dir})")

    repository = DomainRepository()
    try:
        tags = compute_tags(video_names, frames, frame_to_scene, repository)
    finally:
        repository.close()

    _print_coverage(tags)

    tags_path = os.path.join(args.snapshot_dir, TAGS_FILE)
    np.save(tags_path, tags)

    vocab_path = os.path.join(args.snapshot_dir, TAG_VOCAB_FILE)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(build_tag_vocab(), f, ensure_ascii=False, indent=2)
        f.write("\n")

    written = [TAGS_FILE, TAG_VOCAB_FILE]
    if not args.no_manifest:
        _update_manifest_checksums(args.snapshot_dir, written)
        written = [*written, MANIFEST_FILE]

    print(f"ghi {tags_path}")
    print(f"ghi {vocab_path}")

    if args.upload_s3:
        _upload_to_s3(args.snapshot_dir, args.upload_s3, written)
    else:
        print("--upload-s3 rỗng — chỉ ghi cục bộ, TỰ upload lên SNAPSHOT_S3 trước khi "
              "start/restart core với bản snapshot này.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
