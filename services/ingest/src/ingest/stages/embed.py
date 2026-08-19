"""Embed keyframe bằng SigLIP -> vector fp32 (nguồn sự thật, xem bất biến I5 trong
Handoff_core_be.md: fp32 trên S3 để rebuild, SQ8/fp16 là bản phái sinh dùng lúc serve).

Payload ghi kèm mỗi vector khớp trực tiếp field của `Payload` trong
`proto/searchcore/v1/common.proto`, để `build_index.py` dựng `Point` sau này mà
không cần suy diễn lại. Mới chỉ embed tier KEYFRAME (`INDEX_TIER_KEYFRAME=2`) —
tier SCENE (embedding trung bình theo scene) chưa làm. `objects`/`has_ocr` để
mặc định rỗng/False vì stage `objects.py`/`ocr.py` chưa code.

Không ép cứng số chiều vector: SigLIP (khác CLIP) không có projection layer riêng
nên `SiglipConfig` không có field `projection_dim` — dim được suy ra từ shape thật
của batch đầu tiên, ghi kèm cột `dim` trong parquet để đối chiếu với `VECTOR_DIM`
bên searchcore.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_MODEL = "google/siglip-so400m-patch14-384"
SHARD_SIZE = 50_000
TIER_KEYFRAME = 2  # IndexTier.INDEX_TIER_KEYFRAME, proto/searchcore/v1/common.proto


def load_siglip(name: str = DEFAULT_MODEL, device=None):
    """Nạp SiglipModel + processor (nạp 1 lần, tái dùng cho mọi video)."""
    from transformers import SiglipModel, SiglipProcessor

    model = SiglipModel.from_pretrained(name).to(device).eval()
    processor = SiglipProcessor.from_pretrained(name)
    return model, processor


def embed_keyframes(
    video_out_dir: str,
    keyframes: list[dict],
    model,
    processor,
    device,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode ảnh keyframe (theo `keyframe_url`) -> vector fp32 L2-normalized.

    Thứ tự hàng khớp với `keyframes`. Inference chạy fp16 (autocast) trên GPU, ép
    về fp32 numpy trước khi trả về vì fp32 là bản lưu trữ nguồn sự thật.

    Dim vector không đọc từ `model.config` (SigLIP không có `projection_dim` như
    CLIP) — batch đầu tiên quyết định dim, dựa theo shape thật của output.
    """
    import torch
    from PIL import Image

    if not keyframes:
        return np.empty((0, 0), dtype=np.float32)

    device_type = device.type if hasattr(device, "type") else str(device)
    vectors = None

    with torch.no_grad():
        for start in range(0, len(keyframes), batch_size):
            batch = keyframes[start : start + batch_size]
            images = [
                Image.open(os.path.join(video_out_dir, kf["keyframe_url"])).convert("RGB")
                for kf in batch
            ]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
                feats = model.get_image_features(**inputs)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats = feats.cpu().numpy()

            if vectors is None:
                vectors = np.empty((len(keyframes), feats.shape[1]), dtype=np.float32)
            vectors[start : start + len(batch)] = feats

    return vectors



# =====================================================================
# Payload (thuần Python, test được không cần model/GPU)
# =====================================================================
def assign_scene_idx(keyframes: list[dict], scenes: list[dict]) -> list[int]:
    """Map mỗi keyframe -> `scene_id` (chỉ số scene) chứa nó, theo `frame` nằm
    trong `[start_frame, end_frame]` của scene.

    Giả định `keyframes` và `scenes` đã sort tăng dần theo frame (đúng thứ tự
    pipeline sinh ra). Scene cuối bao trọn phần dư nếu frame vượt `end_frame`
    do làm tròn lúc trích keyframe.
    """
    result = []
    s = 0
    for kf in keyframes:
        while s < len(scenes) - 1 and kf["frame"] > scenes[s]["end_frame"]:
            s += 1
        result.append(scenes[s]["scene_id"])
    return result


def keyframe_point_id(video_id: int, frame: int) -> int:
    """Id tất định cho `Point` (tier KEYFRAME): video_id (32 bit cao) | frame (32 bit thấp).

    Không cần bảng DB riêng cho keyframe — `video_id` (Postgres bigserial) và
    `frame` (frame index trong 1 video) đều thừa dư so với 32 bit mỗi phần.
    """
    return (video_id << 32) | frame


def build_payload_rows(video_id: int, keyframes: list[dict], scenes: list[dict]) -> list[dict]:
    """Payload cho mỗi keyframe, field khớp `Payload` (common.proto), tier=KEYFRAME.

    `start_sec`/`end_sec` là thời gian của **scene chứa keyframe** (khớp cách BE
    hydrate `{s, e}` từ bảng `scenes`, xem mục 4.3 Handoff_core_be.md — để
    `Filter.min_start_sec/max_end_sec/*_duration_sec` lọc theo khoảng thời gian có
    ý nghĩa). `keyframe_time` là thời điểm riêng của keyframe, không trùng scene.
    """
    scene_idx_by_kf = assign_scene_idx(keyframes, scenes)
    rows = []
    for kf, scene_idx in zip(keyframes, scene_idx_by_kf):
        scene = scenes[scene_idx]
        rows.append({
            "point_id": keyframe_point_id(video_id, kf["frame"]),
            "video_id": video_id,
            "scene_idx": scene_idx,
            "keyframe_time": kf["timestamp"],
            "start_sec": scene["start_time"],
            "end_sec": scene["end_time"],
            "objects": [],
            "has_ocr": False,
            "has_speech": bool(scene.get("script")),
            "keyframe_key": f"{video_id}/{kf['keyframe_url']}",
            "clip_key": f"{video_id}/{scene['scene_url']}" if scene.get("scene_url") else None,
            "tier": TIER_KEYFRAME,
        })
    return rows


def dump_shards(
    video_id: int,
    video_name: str,
    keyframes: list[dict],
    scenes: list[dict],
    vectors: np.ndarray,
    out_dir: str,
    *,
    shard_size: int = SHARD_SIZE,
) -> list[str]:
    """Ghi vector + payload ra parquet, tối đa `shard_size` vector/file.

    Trả về danh sách đường dẫn parquet đã ghi (nằm trong `out_dir`).
    """
    if len(keyframes) != len(vectors):
        raise ValueError(
            f"Số keyframe ({len(keyframes)}) khác số vector ({len(vectors)})")

    os.makedirs(out_dir, exist_ok=True)
    payload_rows = build_payload_rows(video_id, keyframes, scenes)
    dim = int(vectors.shape[1]) if len(vectors) else 0

    paths = []
    for shard_id, start in enumerate(range(0, max(len(keyframes), 1), shard_size)):
        if start >= len(keyframes):
            break
        end = min(start + shard_size, len(keyframes))
        chunk_kf = keyframes[start:end]
        chunk_payload = payload_rows[start:end]
        table = pa.table({
            "point_id": [p["point_id"] for p in chunk_payload],
            "video_id": [p["video_id"] for p in chunk_payload],
            "video_name": [video_name] * (end - start),
            "frame": [kf["frame"] for kf in chunk_kf],
            "keyframe_url": [kf["keyframe_url"] for kf in chunk_kf],
            "scene_idx": [p["scene_idx"] for p in chunk_payload],
            "keyframe_time": [p["keyframe_time"] for p in chunk_payload],
            "start_sec": [p["start_sec"] for p in chunk_payload],
            "end_sec": [p["end_sec"] for p in chunk_payload],
            "objects": [p["objects"] for p in chunk_payload],
            "has_ocr": [p["has_ocr"] for p in chunk_payload],
            "has_speech": [p["has_speech"] for p in chunk_payload],
            "keyframe_key": [p["keyframe_key"] for p in chunk_payload],
            "clip_key": [p["clip_key"] for p in chunk_payload],
            "tier": [p["tier"] for p in chunk_payload],
            "dim": [dim] * (end - start),
            "vector": [vectors[i].tolist() for i in range(start, end)],
        })
        path = os.path.join(out_dir, f"embed_{shard_id:03d}.parquet")
        pq.write_table(table, path)
        paths.append(path)
    return paths
