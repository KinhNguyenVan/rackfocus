"""Contract transcript keyword search: sanitize MATCH, khớp FTS5, resolve URL, và endpoint.

Dựng SQLite FTS5 tạm (schema khớp ingest/build_transcript_index.py) — không phụ thuộc gói
ingest/S3/AWS: inject url_resolver thuần.
"""
import sqlite3

from app.services.transcript import TranscriptIndex, build_match


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE scenes_meta (video_name TEXT, scene_idx INTEGER, start_sec REAL,"
        " end_sec REAL, clip_key TEXT, keyframe_key TEXT, script TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE transcript_fts USING fts5(script, content='scenes_meta',"
        " content_rowid='rowid', tokenize='unicode61 remove_diacritics 0')")
    conn.executemany(
        "INSERT INTO scenes_meta VALUES (?,?,?,?,?,?,?)",
        [
            ("L21_V001", 0, 0.0, 4.0, "7/scenes/scene_000.mp4",
             "7/keyframes/000000.webp", "biến đổi khí hậu ảnh hưởng nông nghiệp"),
            ("L21_V001", 2, 8.0, 12.0, None, "7/keyframes/000250.webp",
             "phỏng vấn trong studio về khí hậu"),
            ("L30_V009", 5, 3.0, 6.0, "https://cdn.example/clip.mp4", "",
             "chương trình nấu ăn món cá"),
        ],
    )
    conn.execute("INSERT INTO transcript_fts(transcript_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    return path


def _index(tmp_path):
    db = _make_db(str(tmp_path / "t.sqlite"))
    # resolver thuần: key tương đối -> "URL:<key>", URL tuyệt đối giữ nguyên, rỗng -> "".
    def resolve(key):
        if not key:
            return ""
        return key if key.startswith("http") else f"URL:{key}"

    return TranscriptIndex(db, url_resolver=resolve)


# --------------------------- build_match ---------------------------
def test_build_match_adds_prefix_star_per_token():
    assert build_match("biến đổi") == "biến* đổi*"


def test_build_match_strips_fts_operators():
    # Ký tự toán tử (" *) bị loại, và "OR" hạ về "or*" (prefix vô hại, không phải toán tử)
    # -> query người dùng không thể phá cú pháp MATCH.
    assert build_match('khí" OR *') == "khí* or*"


def test_build_match_empty_when_no_tokens():
    assert build_match("   ") == ""
    assert build_match('"*(')== ""


# --------------------------- suggest ---------------------------
def test_suggest_prefix_matches_and_resolves_urls(tmp_path):
    idx = _index(tmp_path)
    items = idx.suggest("khí", 10)

    assert {(i["video_name"], i["scene_idx"]) for i in items} == {
        ("L21_V001", 0), ("L21_V001", 2)}
    first = next(i for i in items if i["scene_idx"] == 0)
    assert first["clip_url"] == "URL:7/scenes/scene_000.mp4"
    assert first["keyframe_url"] == "URL:7/keyframes/000000.webp"
    assert "[" in first["snippet"] and "]" in first["snippet"]


def test_suggest_passes_through_absolute_url_and_empty_key(tmp_path):
    idx = _index(tmp_path)
    items = idx.suggest("nấu", 10)
    assert len(items) == 1
    assert items[0]["clip_url"] == "https://cdn.example/clip.mp4"
    assert items[0]["keyframe_url"] == ""  # key rỗng -> URL rỗng


def test_suggest_no_match_returns_empty(tmp_path):
    assert _index(tmp_path).suggest("zzzkhông", 10) == []


def test_suggest_empty_query_returns_empty_without_touching_db(tmp_path):
    assert _index(tmp_path).suggest("", 10) == []


# --------------------------- endpoint (app standalone, không cần gRPC core) ---------------------------
# KHÔNG dùng fixture `client` (nó dựng core gRPC qua unix socket — không bind được trên
# Windows). Chỉ mount transcript router để test wiring HTTP + mã lỗi.
def _api_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api import transcript as transcript_api

    app = FastAPI()
    app.include_router(transcript_api.router, prefix="/api")
    return TestClient(app)


def test_endpoint_short_query_returns_empty():
    r = _api_client().get("/api/transcript/suggest?q=a")
    assert r.status_code == 200
    assert r.json() == {"query": "a", "items": []}


def test_endpoint_503_when_index_not_loaded():
    # index chưa nạp (module-level None) -> 503 (không phải 500).
    from app.services import transcript as svc

    assert svc.get() is None
    r = _api_client().get("/api/transcript/suggest?q=khí")
    assert r.status_code == 503
