"""Test build transcript.sqlite (FTS5): rút row từ scene, key trùng embed, và truy vấn khớp.

Thuần logic + SQLite, không cần Postgres/S3/model.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.build_transcript_index import (
    FTS_TABLE,
    assign_segments_to_scenes,
    build_transcript_db,
    rows_from_payload_and_transcripts,
    scene_transcript_rows,
)


def _keyframes(frames):
    return [
        {"frame": f, "timestamp": round(f / 25.0, 3), "keyframe_url": f"keyframes/{f:06d}.webp"}
        for f in frames
    ]


def _scenes():
    return [
        {"scene_id": 0, "start_frame": 0, "end_frame": 100,
         "start_time": 0.0, "end_time": 4.0,
         "script": "biến đổi khí hậu ảnh hưởng nông nghiệp",
         "scene_url": "scenes/scene_000.mp4"},
        {"scene_id": 1, "start_frame": 101, "end_frame": 200,
         "start_time": 4.04, "end_time": 8.0, "script": "",
         "scene_url": "scenes/scene_001.mp4"},
        {"scene_id": 2, "start_frame": 201, "end_frame": 300,
         "start_time": 8.0, "end_time": 12.0,
         "script": "phỏng vấn trong studio về khí hậu",
         "scene_url": None},
    ]


# --------------------------- scene_transcript_rows ---------------------------
def test_only_scenes_with_script_become_rows():
    rows = scene_transcript_rows(7, "L21_V001", _keyframes([0, 150, 250]), _scenes())
    # scene 1 script rỗng -> bị bỏ; còn scene 0 và 2.
    assert [r["scene_idx"] for r in rows] == [0, 2]


def test_keys_match_embed_payload_formula():
    rows = scene_transcript_rows(7, "L21_V001", _keyframes([0, 150, 250]), _scenes())
    first = rows[0]
    # clip_key/keyframe_key dựng như embed.build_payload_rows: f"{video_id}/{...}".
    assert first["clip_key"] == "7/scenes/scene_000.mp4"
    assert first["keyframe_key"] == "7/keyframes/000000.webp"
    assert first["start_sec"] == 0.0 and first["end_sec"] == 4.0
    assert first["video_name"] == "L21_V001"
    # scene 2 không có scene_url -> clip_key None; keyframe đại diện là frame 250.
    assert rows[1]["clip_key"] is None
    assert rows[1]["keyframe_key"] == "7/keyframes/000250.webp"


def test_scene_without_keyframe_gets_empty_keyframe_key():
    rows = scene_transcript_rows(7, "V", [], _scenes())
    assert rows[0]["keyframe_key"] == ""


# --------------------------- build_transcript_db + query ---------------------------
def _query(db_path, match, limit=10):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT m.video_name, m.scene_idx,"
            f"       snippet({FTS_TABLE}, 0, '[', ']', '…', 12) "
            f"FROM {FTS_TABLE} JOIN scenes_meta m ON m.rowid = {FTS_TABLE}.rowid "
            f"WHERE {FTS_TABLE} MATCH ? ORDER BY bm25({FTS_TABLE}) LIMIT ?",
            (match, limit),
        ).fetchall()
    finally:
        conn.close()


def test_prefix_match_finds_scene_and_highlights(tmp_path):
    rows = scene_transcript_rows(7, "L21_V001", _keyframes([0, 250]), _scenes())
    db = build_transcript_db(rows, str(tmp_path / "t.sqlite"))

    hits = _query(db, "khí*")
    # Cả hai scene đều nhắc "khí hậu".
    assert {h[1] for h in hits} == {0, 2}
    # snippet bọc keyword trong [ ] để FE highlight.
    assert any("[" in h[2] and "]" in h[2] for h in hits)


def test_diacritics_preserved_khi_khac_khi(tmp_path):
    # remove_diacritics 0: "khi" (không dấu) KHÔNG khớp "khí" (có dấu).
    rows = scene_transcript_rows(1, "V", _keyframes([0]), _scenes()[:1])
    db = build_transcript_db(rows, str(tmp_path / "t.sqlite"))
    assert _query(db, "khí*")      # có dấu -> khớp
    assert not _query(db, "xyz*")  # không có -> rỗng


# ============ build từ snapshot payload + ASR transcript (không cần Postgres) ============
def _payload_scenes():
    # 2 scene liền nhau, mỗi scene 2 keyframe (test dedup theo (video_name, scene_idx)).
    return [
        {"scene_idx": 0, "start_sec": 0.0, "end_sec": 5.0},
        {"scene_idx": 1, "start_sec": 5.0, "end_sec": 10.0},
    ]


def test_assign_segments_midpoint_rule():
    scenes = _payload_scenes()
    segments = [
        {"start": 1.0, "end": 3.0, "text": "câu một"},   # mid 2.0 -> scene 0
        {"start": 4.0, "end": 8.0, "text": "câu hai"},   # mid 6.0 -> scene 1 (dù overlap cả 2)
        {"start": 9.0, "end": 9.4, "text": "câu ba"},    # mid 9.2 -> scene 1
    ]
    got = assign_segments_to_scenes(scenes, segments)
    assert got == {0: "câu một", 1: "câu hai câu ba"}


def _write_payload(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["video_name", "scene_idx", "start_sec", "end_sec", "clip_key", "keyframe_key"]
    pq.write_table(pa.table({c: [r[c] for r in rows] for c in cols}), path)


def test_rows_from_payload_and_transcripts(tmp_path):
    import json

    # payload: video A có 2 scene (mỗi scene lặp 2 keyframe -> test dedup); video B không có transcript.
    payload = str(tmp_path / "payload.parquet")
    url = "https://cdn/x/scene_{}.mp4"
    p_rows = []
    for si in (0, 1):
        for _ in range(2):  # 2 keyframe/scene
            p_rows.append({"video_name": "A", "scene_idx": si, "start_sec": si * 5.0,
                           "end_sec": si * 5.0 + 5.0, "clip_key": url.format(si),
                           "keyframe_key": f"https://cdn/kf/{si}.webp"})
    p_rows.append({"video_name": "B", "scene_idx": 0, "start_sec": 0.0, "end_sec": 5.0,
                   "clip_key": "https://cdn/b0.mp4", "keyframe_key": ""})
    _write_payload(payload, p_rows)

    tx_dir = tmp_path / "transcripts"
    tx_dir.mkdir()
    # A: scene 0 có thoại, scene 1 KHÔNG (không segment nào rơi vào) -> chỉ 1 row.
    (tx_dir / "A.json").write_text(json.dumps({
        "video_id": "A", "segments": [{"start": 1.0, "end": 2.0, "text": "biến đổi khí hậu"}],
    }), encoding="utf-8")
    # B: không có file transcript -> bị bỏ.

    rows = rows_from_payload_and_transcripts(payload, str(tx_dir))
    assert len(rows) == 1
    r = rows[0]
    assert r["video_name"] == "A" and r["scene_idx"] == 0
    assert r["clip_key"] == "https://cdn/x/scene_0.mp4"   # URL tuyệt đối giữ nguyên
    assert r["script"] == "biến đổi khí hậu"

    # build + query để chắc pipeline hoàn chỉnh chạy.
    db = build_transcript_db(rows, str(tmp_path / "t.sqlite"))
    hits = _query(db, "khí*")
    assert [h[1] for h in hits] == [0]
