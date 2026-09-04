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


def _load_asr_module():
    """Nạp `ingest/stages/asr.py` như module rời (xem test parity ở cuối file)."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "src", "ingest", "stages", "asr.py")
    spec = importlib.util.spec_from_file_location("_asr_standalone", os.path.abspath(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    # Assertion CHỐT của tokenizer: bỏ `remove_diacritics 0` khỏi create_schema thì FTS5 lột
    # dấu và "khi*" khớp "khí hậu" -> dòng này đỏ. Hai assert trên KHÔNG canh được điều đó
    # (chúng xanh ở cả hai tokenizer), nên nếu chỉ có chúng thì cấu hình tiếng Việt không
    # được test nào bảo vệ.
    assert not _query(db, "khi*")


def test_build_transcript_db_refuses_empty_rows(tmp_path):
    # DB đúng schema + 0 row là ca lỗi im lặng nhất: BE mở được, endpoint trả 200 items=[]
    # cho mọi keyword. Phải raise và KHÔNG để lại file.
    import pytest

    out = tmp_path / "empty.sqlite"
    with pytest.raises(ValueError, match="0 row"):
        build_transcript_db([], str(out))
    assert not out.exists()


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


def test_transcript_json_sai_schema_khong_bi_coi_la_khong_co_thoai(tmp_path, capsys):
    """JSON hỏng/thiếu `segments` phải được ĐẾM RIÊNG, không im lặng thành 0 row.

    `.get("segments", [])` biến "file tải dở" thành "video này không ai nói gì" — cùng kết
    quả với video thật sự im lặng, nên tải S3 lỗi một nửa vẫn build ra index trông bình thường.
    """
    import json

    payload = str(tmp_path / "payload.parquet")
    _write_payload(payload, [
        {"video_name": "OK", "scene_idx": 0, "start_sec": 0.0, "end_sec": 5.0,
         "clip_key": "https://cdn/ok.mp4", "keyframe_key": ""},
        {"video_name": "TRUNCATED", "scene_idx": 0, "start_sec": 0.0, "end_sec": 5.0,
         "clip_key": "https://cdn/t.mp4", "keyframe_key": ""},
        {"video_name": "NOSEGMENTS", "scene_idx": 0, "start_sec": 0.0, "end_sec": 5.0,
         "clip_key": "https://cdn/n.mp4", "keyframe_key": ""},
    ])
    tx = tmp_path / "transcripts"
    tx.mkdir()
    (tx / "OK.json").write_text(json.dumps({
        "segments": [{"start": 1.0, "end": 2.0, "text": "có thoại"}]}), encoding="utf-8")
    (tx / "TRUNCATED.json").write_text('{"segments": [{"start": 1.0,', encoding="utf-8")
    (tx / "NOSEGMENTS.json").write_text('{"video_id": "NOSEGMENTS"}', encoding="utf-8")

    rows = rows_from_payload_and_transcripts(payload, str(tx))
    assert [r["video_name"] for r in rows] == ["OK"]
    out = capsys.readouterr().out
    assert "json lỗi/sai schema: 2" in out
    assert "TRUNCATED.json" in out and "NOSEGMENTS.json" in out


def test_cli_khong_xoa_index_cu_khi_build_that_bai(tmp_path, monkeypatch):
    """Build lỗi phải để NGUYÊN index đang dùng.

    Trước đây CLI `os.remove(db_path)` NGAY TỪ ĐẦU rồi mới đi tải S3 + ghép, nên chạy lại
    lệnh với `SNAPSHOT_S3`/prefix sai là xoá mất index đang phục vụ, đổi lấy một file rỗng.
    Giờ ghi vào `<db>.tmp` rồi `os.replace`, nên mọi lỗi đều không đụng tới bản cũ.
    """
    import pytest

    from ingest.build_transcript_index import main

    # payload có 1 video nhưng KHÔNG có file transcript nào -> ghép ra 0 row.
    payload = str(tmp_path / "payload.parquet")
    _write_payload(payload, [{"video_name": "A", "scene_idx": 0, "start_sec": 0.0,
                              "end_sec": 5.0, "clip_key": "https://cdn/a.mp4",
                              "keyframe_key": ""}])
    tx = tmp_path / "transcripts"
    tx.mkdir()

    db = tmp_path / "transcript.sqlite"
    db.write_bytes(b"INDEX CU DANG DUNG")

    monkeypatch.setattr("sys.argv", [
        "build_transcript_index", "--payload", payload, "--transcripts", str(tx),
        "--db", str(db)])
    with pytest.raises(ValueError, match="0 row"):
        main()

    assert db.read_bytes() == b"INDEX CU DANG DUNG"
    assert not (tmp_path / "transcript.sqlite.tmp").exists()


def test_parity_voi_stages_asr_assign_script_to_scenes():
    """`assign_segments_to_scenes` phải cho ĐÚNG kết quả `stages.asr.assign_script_to_scenes`.

    Hai hàm là bản sao của nhau (một dùng start_sec/end_sec từ payload, một dùng
    start_time/end_time từ scene JSON) và cùng quyết định scene nào có thoại. Lệch nhau =
    transcript index không khớp cờ `has_speech` của snapshot, mà không gì báo.

    Nạp `asr.py` THEO ĐƯỜNG DẪN, không `from ingest.stages.asr import ...`: package
    `ingest.stages.__init__` import sẵn `media` -> cv2, và cv2 không nằm trong deps của job
    test ingest ở CI. Bản thân `asr.py` chỉ định nghĩa hằng ở mức module (chunkformer/torch
    import trong hàm) nên nạp rời được. Đây cũng là lý do `build_transcript_index.py` chép
    hàm này thay vì import.
    """
    assign_script_to_scenes = _load_asr_module().assign_script_to_scenes

    bounds = [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    segments = [
        {"start": 1.0, "end": 3.0, "text": "câu một"},
        {"start": 4.0, "end": 8.0, "text": "câu hai"},     # mid 6.0 -> scene 1
        {"start": 9.0, "end": 11.0, "text": "câu ba"},     # mid 10.0 -> ĐÚNG BIÊN 1|2
        {"start": 11.5, "end": 13.0, "text": "câu bốn"},   # mid 12.25 -> ngoài mọi scene
    ]

    payload_scenes = [
        {"scene_idx": i, "start_sec": lo, "end_sec": hi} for i, (lo, hi) in enumerate(bounds)
    ]
    asr_scenes = [
        {"scene_id": i, "start_time": lo, "end_time": hi} for i, (lo, hi) in enumerate(bounds)
    ]

    got = assign_segments_to_scenes(payload_scenes, segments)
    assign_script_to_scenes(asr_scenes, segments)
    want = {sc["scene_id"]: sc["script"] for sc in asr_scenes if sc["script"]}

    assert got == want
    # Midpoint đúng biên chung: cả hai dùng `lo <= mid <= hi` nên segment đó vào CẢ HAI scene.
    assert "câu ba" in got[1] and "câu ba" in got[2]
