"""Build `transcript.sqlite` (FTS5) — artifact phụ của snapshot cho keyword search transcript.

Stack online (BE) tra keyword trên lời thoại scene rồi gợi ý kiểu Google; nhưng transcript
(`scene.script`, sinh ở `stages/asr.py`) KHÔNG có trong `Payload` snapshot (chỉ có cờ
`has_speech`) và online KHÔNG có Postgres. Nên đóng gói riêng thành 1 SQLite FTS5 read-only,
ship kèm snapshot, BE mở lúc khởi động (xem `services/be/src/app/services/transcript.py`).

Key `clip_key`/`keyframe_key` dựng ĐÚNG công thức của `stages/embed.py::build_payload_rows`
(`f"{video_id}/{...}"`) để trùng khít key trong parquet payload vector — click 1 gợi ý mở
đúng scene clip mà search vector cũng trỏ tới.

Tokenizer `unicode61 remove_diacritics 0`: GIỮ dấu tiếng Việt (mặc định của FTS5 lột dấu,
làm "khí" khớp cả "khi"). Tách token theo khoảng trắng/ký tự không phải chữ.

Tích hợp: `build_index.py` (dựng snapshot) gọi `build_transcript_db(...)` ghi
`transcript.sqlite` vào thư mục snapshot rồi thêm vào `manifest.json` checksums. CLI ở cuối
file dựng trực tiếp từ output ingest (`scene_<name>.json` + `keyframes.json`) để test/chạy tay.
"""

from __future__ import annotations

import sqlite3

# Tên bảng/artifact — BE đọc đúng tên này.
DB_FILENAME = "transcript.sqlite"
META_TABLE = "scenes_meta"
FTS_TABLE = "transcript_fts"


def _assign_scene_idx(keyframes: list[dict], scenes: list[dict]) -> list[int]:
    """Map mỗi keyframe -> `scene_id` chứa nó (frame trong [start_frame, end_frame]).

    Bản sao gọn của `stages.embed.assign_scene_idx` — không import `stages` để tránh kéo
    theo cv2/torch (media/scene_group) chỉ để chọn keyframe đại diện. Giữ đồng bộ logic
    với embed để keyframe đại diện khớp cách gán scene lúc embed.
    """
    result = []
    s = 0
    for kf in keyframes:
        while s < len(scenes) - 1 and kf["frame"] > scenes[s]["end_frame"]:
            s += 1
        result.append(scenes[s]["scene_id"])
    return result


def scene_transcript_rows(
    video_id: int,
    video_name: str,
    keyframes: list[dict],
    scenes: list[dict],
) -> list[dict]:
    """Mỗi scene CÓ script -> 1 row cho index. Bỏ scene script rỗng (khớp `has_speech`).

    `keyframe_key` là keyframe ĐẦU TIÊN thuộc scene (để hiện thumbnail gợi ý); scene không
    có keyframe nào -> "". `clip_key`/`keyframe_key` dựng như `embed.build_payload_rows` nên
    trùng key parquet payload.
    """
    # keyframe đại diện: keyframe đầu tiên map vào từng scene_id.
    first_kf: dict[int, dict] = {}
    if keyframes and scenes:
        for kf, sidx in zip(keyframes, _assign_scene_idx(keyframes, scenes)):
            first_kf.setdefault(sidx, kf)

    rows = []
    for scene in scenes:
        script = (scene.get("script") or "").strip()
        if not script:
            continue
        sidx = int(scene["scene_id"])
        scene_url = scene.get("scene_url")
        kf = first_kf.get(sidx)
        rows.append({
            "video_name": video_name,
            "scene_idx": sidx,
            "start_sec": float(scene["start_time"]),
            "end_sec": float(scene["end_time"]),
            "clip_key": f"{video_id}/{scene_url}" if scene_url else None,
            "keyframe_key": f"{video_id}/{kf['keyframe_url']}" if kf else "",
            "script": script,
        })
    return rows


def create_schema(conn: sqlite3.Connection) -> None:
    """Bảng thường giữ metadata + FTS5 external-content chỉ index cột `script`.

    External content (`content='scenes_meta'`): FTS không nhân đôi text, `snippet()` vẫn lấy
    được text gốc từ bảng meta theo rowid.
    """
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} ("
        "  video_name TEXT NOT NULL,"
        "  scene_idx INTEGER NOT NULL,"
        "  start_sec REAL NOT NULL,"
        "  end_sec REAL NOT NULL,"
        "  clip_key TEXT,"
        "  keyframe_key TEXT,"
        "  script TEXT NOT NULL)"
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5("
        f"  script, content='{META_TABLE}', content_rowid='rowid',"
        "  tokenize='unicode61 remove_diacritics 0')"
    )


def add_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Chèn rows vào bảng meta. Gọi `finalize` sau khi thêm hết để dựng FTS. Trả số row thêm."""
    conn.executemany(
        f"INSERT INTO {META_TABLE}"
        " (video_name, scene_idx, start_sec, end_sec, clip_key, keyframe_key, script)"
        " VALUES (:video_name, :scene_idx, :start_sec, :end_sec, :clip_key,"
        "         :keyframe_key, :script)",
        rows,
    )
    return len(rows)


def finalize(conn: sqlite3.Connection) -> None:
    """Dựng lại FTS index từ toàn bộ bảng meta (external content) rồi commit.

    'rebuild' rẻ hơn và ít lỗi hơn trigger khi build 1 lần dạng batch offline.
    """
    conn.execute(f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES('rebuild')")
    conn.commit()


def build_transcript_db(rows: list[dict], out_path: str) -> str:
    """Dựng file SQLite FTS5 hoàn chỉnh từ danh sách rows. Trả `out_path`.

    Idempotent theo file: mở lại file cũ vẫn append; muốn build sạch thì xoá file trước.
    """
    conn = sqlite3.connect(out_path)
    try:
        create_schema(conn)
        add_rows(conn, rows)
        finalize(conn)
    finally:
        conn.close()
    return out_path


# =====================================================================
# CLI: dựng transcript.sqlite từ output ingest (offline, có Postgres cấp video_id)
# =====================================================================
def _iter_video_outputs(out_root: str):
    """Sinh (video_name, keyframes, scenes) từ các thư mục out_root/<name>/ của ingest."""
    import glob
    import json
    import os

    for scene_path in sorted(glob.glob(os.path.join(out_root, "**", "scene_*.json"),
                                        recursive=True)):
        out_dir = os.path.dirname(scene_path)
        name = os.path.basename(scene_path)[len("scene_"):-len(".json")]
        kf_path = os.path.join(out_dir, "keyframes.json")
        with open(scene_path, encoding="utf-8") as f:
            scenes = json.load(f)
        keyframes = []
        if os.path.exists(kf_path):
            with open(kf_path, encoding="utf-8") as f:
                keyframes = json.load(f)
        yield name, keyframes, scenes


def main() -> None:
    import argparse
    import os

    from .db import get_conn, get_or_create_video_id

    ap = argparse.ArgumentParser(
        description="Dựng transcript.sqlite (FTS5) từ output ingest cho keyword search")
    ap.add_argument("--out-root", required=True, help="Thư mục output ingest (chứa <name>/scene_*.json)")
    ap.add_argument("--db", default=None, help=f"Đường dẫn file ra (mặc định out-root/{DB_FILENAME})")
    args = ap.parse_args()

    db_path = args.db or os.path.join(args.out_root, DB_FILENAME)
    if os.path.exists(db_path):
        os.remove(db_path)  # build sạch, tránh append trùng

    conn = sqlite3.connect(db_path)
    pg = get_conn()
    total = 0
    try:
        create_schema(conn)
        for name, keyframes, scenes in _iter_video_outputs(args.out_root):
            video_id = get_or_create_video_id(pg, name)
            rows = scene_transcript_rows(video_id, name, keyframes, scenes)
            total += add_rows(conn, rows)
            print(f"{name}: +{len(rows)} scene có transcript")
        finalize(conn)
    finally:
        conn.close()
        pg.close()
    print(f"Xong: {total} scene -> {db_path}")


if __name__ == "__main__":
    main()
