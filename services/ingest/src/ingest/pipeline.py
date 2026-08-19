"""Orchestrator: đọc job queue, dispatch từng stage, checkpoint để resume.

Bản offline (GPU thuê/Kaggle): `process_video` chạy 1 video end-to-end và ghi 2 JSON
(`scene_<name>.json`, `keyframes.json`); `main()` duyệt các folder `Videos_K{xx}/video/`
và xử lý từng video, nạp model nặng đúng 1 lần.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from .db import get_conn, get_or_create_video_id, upsert_scenes
from .stages import (
    assign_script_to_scenes,
    cut_scenes,
    detect_shots,
    dump_shards,
    embed_keyframes,
    extract_keyframes,
    group_shots_into_scenes,
    load_asr_model,
    load_bassl,
    load_resnet_backbone,
    load_transnet,
    probe,
    transcribe,
)
from .storage import upload_video_outputs


def process_video(
    video_path: str,
    out_dir: str,
    *,
    asr_model,
    transnet_model,
    bassl_model,
    backbone,
    device,
    siglip_model=None,
    siglip_processor=None,
    db_conn=None,
    shot_threshold: float = 0.5,
    scene_threshold: float = 0.55,
    upload: bool = False,
) -> dict:
    """Chạy 1 video: shots -> keyframes -> scenes(BaSSL) -> ASR -> cắt scene -> embed -> 2 JSON.

    Trả về {"scenes": [...], "keyframes": [...]} và ghi ra file trong out_dir.
    Nếu `upload=True` (và có `siglip_model`), sau khi ghi xong sẽ đẩy toàn bộ
    keyframes/scenes/embed_*.parquet của video lên S3 (xem `storage.py`).
    """
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]

    width, height, fps, _ = probe(video_path)

    # 1) Shot boundaries + 2) keyframes (+ shots.csv mapping cho BaSSL).
    shots = detect_shots(video_path, transnet_model, threshold=shot_threshold)
    keyframes = extract_keyframes(
        video_path, shots, out_dir, fps=fps, width=width, height=height)

    # 3) Gom shot -> scene bằng BaSSL CRN.
    shots_df = pd.read_csv(os.path.join(out_dir, "shots.csv"))
    scenes = group_shots_into_scenes(
        shots_df, out_dir,
        model=bassl_model, backbone=backbone, device=device,
        threshold=scene_threshold)

    # 4) ASR toàn video -> gán script cho từng scene theo thời gian.
    segments = transcribe(video_path, asr_model)
    assign_script_to_scenes(scenes, segments)

    # 5) Cắt clip mp4 cho mỗi scene -> gán scene_url (relative).
    cut_scenes(video_path, scenes, out_dir)

    # 6) Đánh scene_id tuần tự + chọn field cho output.
    scenes_out = [
        {
            "scene_id": i,
            "script": sc.get("script", ""),
            "start_frame": sc["start_frame"],
            "end_frame": sc["end_frame"],
            "start_time": sc["start_time"],
            "end_time": sc["end_time"],
            "scene_url": sc.get("scene_url"),
        }
        for i, sc in enumerate(scenes)
    ]

    _dump(os.path.join(out_dir, f"scene_{name}.json"), scenes_out)
    _dump(os.path.join(out_dir, "keyframes.json"), keyframes)

    # 7) Embed keyframe (SigLIP, tier KEYFRAME) -> parquet fp32 + payload cho build_index.
    #    video_id/scene_idx lấy từ Postgres (videos/scenes) để payload khớp id thật dùng
    #    xuyên suốt hệ thống, không tự sinh hash.
    if siglip_model is not None:
        if db_conn is None:
            raise ValueError(
                "Cần db_conn (Postgres) để cấp video_id/scene_idx cho payload embedding")
        video_id = get_or_create_video_id(db_conn, name)
        upsert_scenes(db_conn, video_id, scenes_out)
        vectors = embed_keyframes(out_dir, keyframes, siglip_model, siglip_processor)
        dump_shards(video_id, name, keyframes, scenes_out, vectors, out_dir)

    # 8) Upload keyframes/scenes/embed_*.parquet lên S3 (sau khi mọi thứ đã ghi xong,
    #    để không upload rác nửa vời nếu có bước nào phía trên lỗi).
    if upload:
        upload_video_outputs(out_dir, name)

    return {"scenes": scenes_out, "keyframes": keyframes}


def _dump(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_videos(root: str) -> list[str]:
    """Tìm mọi video trong cấu trúc Videos_K{xx}/video/*.mp4 (fallback: mọi *.mp4)."""
    vids = sorted(glob.glob(os.path.join(root, "**", "video", "*.mp4"), recursive=True))
    if not vids:
        vids = sorted(glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True))
    return vids


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest offline: video -> scene/keyframe JSON")
    ap.add_argument("--root", required=True, help="Thư mục chứa các Videos_K{xx}/")
    ap.add_argument("--out-root", default="./out", help="Thư mục output gốc")
    ap.add_argument("--bassl-ckpt", required=True, help="Đường dẫn checkpoint .ckpt của BaSSL")
    ap.add_argument("--asr-model", default=None, help="Tên/đường dẫn model chunkformer")
    ap.add_argument("--siglip-model", default=None,
                     help="Thư mục (cục bộ hoặc s3://bucket/prefix) chứa bundle ONNX "
                          "(model.onnx + preprocessor_config.json, xem export_siglip_onnx.py) "
                          "để embed keyframe (bỏ trống = không embed)")
    ap.add_argument("--upload", action="store_true",
                     help="Đẩy keyframes/scenes/embed_*.parquet lên S3 sau mỗi video")
    ap.add_argument("--shot-threshold", type=float, default=0.5)
    ap.add_argument("--scene-threshold", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N video đầu (0 = tất cả)")
    ap.add_argument("--overwrite", action="store_true", help="Ghi đè cả khi đã có output")
    args = ap.parse_args()

    videos = find_videos(args.root)
    if args.limit:
        videos = videos[: args.limit]
    print(f"{len(videos)} video cần xử lý")
    if not videos:
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nạp model nặng đúng 1 lần.
    asr_model = load_asr_model(args.asr_model) if args.asr_model else load_asr_model()
    transnet_model = load_transnet()
    backbone = load_resnet_backbone(device)
    bassl_model = load_bassl(args.bassl_ckpt, device)
    siglip_model, siglip_processor = (None, None)
    if args.siglip_model is not None:
        from .stages import load_siglip

        siglip_model, siglip_processor = load_siglip(args.siglip_model, device)

    # Cần Postgres để cấp video_id/scene_idx cho payload khi có embed.
    db_conn = get_conn() if siglip_model is not None else None

    try:
        for i, video_path in enumerate(videos, 1):
            name = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(args.out_root, name)
            done = os.path.join(out_dir, f"scene_{name}.json")
            if os.path.exists(done) and not args.overwrite:
                print(f"[{i}/{len(videos)}] {name} — đã có, bỏ qua")
                continue
            try:
                res = process_video(
                    video_path, out_dir,
                    asr_model=asr_model, transnet_model=transnet_model,
                    bassl_model=bassl_model, backbone=backbone, device=device,
                    siglip_model=siglip_model, siglip_processor=siglip_processor,
                    db_conn=db_conn,
                    shot_threshold=args.shot_threshold,
                    scene_threshold=args.scene_threshold,
                    upload=args.upload)
                print(f"[{i}/{len(videos)}] {name}: "
                      f"{len(res['scenes'])} scenes, {len(res['keyframes'])} keyframes")
            except Exception as ex:  # noqa: BLE001 — 1 video lỗi không được chặn cả batch
                print(f"[{i}/{len(videos)}] {name} — LỖI: {ex}")
    finally:
        if db_conn is not None:
            db_conn.close()


if __name__ == "__main__":
    main()
