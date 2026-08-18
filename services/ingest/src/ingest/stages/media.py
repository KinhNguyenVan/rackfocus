"""Trích keyframe (webp) + cắt scene (mp4). ffmpeg + OpenCV.

Keyframe ghi tên theo **frame index toàn cục** `{frame:06d}.webp` (đây là tên cuối
dùng trong keyframes.json). Mapping shot -> frame index của 3 keyframe được ghi ra
`shots.csv` để stage BaSSL đọc lại đúng ảnh mà không cần đặt tên `shot_XXXX_j.webp`.
"""

from __future__ import annotations

import csv
import os
import subprocess

import cv2
import numpy as np

# Tên subfolder trong out_dir của mỗi video.
KEYFRAMES_DIR = "keyframes"
SCENES_DIR = "scenes"


def extract_keyframes(
    video_path: str,
    shots: list[tuple[int, int]],
    out_dir: str,
    *,
    fps: float,
    width: int,
    height: int,
    out_h: int = 360,      # 360 để tái dùng cho encoder ảnh (input 384)
    inset: float = 0.12,   # lùi vào 12% độ dài shot, tránh frame dissolve/fade ở biên
    quality: int = 90,     # WebP quality
) -> list[dict]:
    """Ghi 3 keyframe/shot (`{frame:06d}.webp`) + `shots.csv`. Một pass decode tuần tự.

    Trả về `keyframes`: list[{"frame", "timestamp", "keyframe_url"}] đã dedup theo
    frame index và sort tăng dần (dùng để ghi keyframes.json).
    """
    kf_dir = os.path.join(out_dir, KEYFRAMES_DIR)
    os.makedirs(kf_dir, exist_ok=True)

    out_w = round(width * out_h / height / 2) * 2   # giữ aspect ratio, chẵn

    # need: frame_idx -> True (ghi 1 lần dù nhiều shot dùng chung).
    # rows: một dòng/shot cho shots.csv, kèm 3 frame index kf0/kf1/kf2.
    need: dict[int, bool] = {}
    rows: list[list] = []
    for sid, (s, e) in enumerate(shots):
        s, e = int(s), int(e)
        off = int(max(e - s, 0) * inset)
        kf = [int(np.clip(i, s, e)) for i in (s + off, (s + e) // 2, e - off)]
        for i in kf:
            need[i] = True
        rows.append([sid, s, e, round(s / fps, 3), round(e / fps, 3), *kf])

    fsz = out_w * out_h * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-v", "quiet", "pipe:1"],
        stdout=subprocess.PIPE, bufsize=fsz * 8,
    )

    n, left = 0, len(need)
    written: set[int] = set()
    try:
        while left > 0:
            buf = proc.stdout.read(fsz)
            if len(buf) < fsz:
                break
            if n in need:
                img = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
                path = os.path.join(kf_dir, f"{n:06d}.webp")
                if cv2.imwrite(path, img, [cv2.IMWRITE_WEBP_QUALITY, quality]):
                    written.add(n)
                else:
                    print(f"  ! ghi hỏng frame {n}")
                left -= 1
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()

    with open(os.path.join(out_dir, "shots.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot_id", "start_frame", "end_frame",
                    "start_ts", "end_ts", "kf0", "kf1", "kf2"])
        w.writerows(rows)

    return [
        {
            "frame": idx,
            "timestamp": round(idx / fps, 3),
            "keyframe_url": f"{KEYFRAMES_DIR}/{idx:06d}.webp",
        }
        for idx in sorted(written)
    ]


def cut_scenes(video_path: str, scenes: list[dict], out_dir: str) -> None:
    """Cắt mỗi scene thành `scenes/scene_{id:03d}.mp4` và gán `scene_url` (relative).

    Re-encode libx264/crf23 + aac để cắt chính xác theo mốc thời gian (stream copy
    sẽ lệch về keyframe gần nhất). `-ss` đặt trước `-i` cho seek nhanh.
    """
    sc_dir = os.path.join(out_dir, SCENES_DIR)
    os.makedirs(sc_dir, exist_ok=True)

    for i, sc in enumerate(scenes):
        start_time = float(sc["start_time"])
        duration = float(sc["end_time"]) - start_time
        rel = f"{SCENES_DIR}/scene_{i:03d}.mp4"
        out_path = os.path.join(out_dir, rel)
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-ss", str(start_time), "-i", video_path, "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
            sc["scene_url"] = rel
        except subprocess.CalledProcessError as ex:
            print(f"  ! Lỗi khi cắt scene {i}: {ex}")
            sc["scene_url"] = None
