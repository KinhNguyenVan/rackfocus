"""Postgres: cấp id (bigserial) cho video/scene, dùng làm `Payload.video_id` và
`scene_idx` khi embed. Bảng khai báo ở `sql/init/01_schema.sql`.
"""

from __future__ import annotations

import os


def get_conn():
    """Kết nối Postgres (đọc `DATABASE_URL` từ env). Nạp 1 lần, tái dùng cho mọi video."""
    import psycopg

    return psycopg.connect(os.environ["DATABASE_URL"])


def get_or_create_video_id(conn, name: str) -> int:
    """Trả id (bigserial) của video theo tên; tạo mới nếu chưa có."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO videos (name) VALUES (%s) "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
            "RETURNING id",
            (name,),
        )
        video_id = cur.fetchone()[0]
    conn.commit()
    return video_id


def upsert_scenes(conn, video_id: int, scenes: list[dict]) -> None:
    """Ghi/cập nhật scenes theo `(video_id, scene_idx)`.

    `scenes`: list dict có `scene_id` (dùng làm `scene_idx`), `start_time`, `end_time`,
    `script`, `scene_url` (như `scenes_out` trong `pipeline.process_video`).
    """
    with conn.cursor() as cur:
        for sc in scenes:
            clip_key = f"{video_id}/{sc['scene_url']}" if sc.get("scene_url") else None
            cur.execute(
                "INSERT INTO scenes (video_id, scene_idx, start_sec, end_sec, script, clip_key) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (video_id, scene_idx) DO UPDATE SET "
                "start_sec = EXCLUDED.start_sec, end_sec = EXCLUDED.end_sec, "
                "script = EXCLUDED.script, clip_key = EXCLUDED.clip_key",
                (video_id, sc["scene_id"], sc["start_time"], sc["end_time"],
                 sc.get("script", ""), clip_key),
            )
    conn.commit()
