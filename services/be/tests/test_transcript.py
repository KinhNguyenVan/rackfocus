"""Contract transcript keyword search: sanitize MATCH, khớp FTS5, resolve URL, và endpoint.

Dựng SQLite FTS5 tạm (schema khớp ingest/build_transcript_index.py) — không phụ thuộc gói
ingest/S3/AWS: inject url_resolver thuần.
"""
import sqlite3

import pytest

from app.services.transcript import TranscriptIndex, TranscriptIndexError, build_match


@pytest.fixture(autouse=True)
def _reset_module_index():
    """Đưa state module-level về "chưa nạp" TRƯỚC và SAU mỗi test.

    `app.services.transcript._index` là biến module -> test nào gọi `load()` sẽ để lại
    index cho test sau. Trước đây `test_endpoint_503_when_index_not_loaded` chỉ *assert*
    `get() is None` chứ không *thiết lập* nó, nên nó đỏ hay xanh tuỳ THỨ TỰ COLLECT — kiểu
    đỏ khó lần nhất khi thêm test mới.
    """
    from app.services import transcript as svc

    svc._index, svc._status = None, "disabled"
    yield
    svc._index, svc._status = None, "disabled"


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


# --------------------------- fail-closed lúc khởi tạo ---------------------------
# Cả ba ca dưới đây TRƯỚC ĐÂY lọt qua `__init__` (chỉ `connect().close()`, mà sqlite mở
# lazy nên không đọc gì) và chỉ vỡ ở tầng HTTP thành 500, hoặc tệ hơn: trả 200 items=[]
# cho mọi keyword, y như "không có câu thoại nào khớp".
def test_init_rejects_missing_file(tmp_path):
    with pytest.raises(TranscriptIndexError):
        TranscriptIndex(str(tmp_path / "khong-ton-tai.sqlite"), url_resolver=str)


def test_init_rejects_garbage_file(tmp_path):
    p = tmp_path / "rac.sqlite"
    p.write_bytes(b"day khong phai sqlite")
    with pytest.raises(TranscriptIndexError):
        TranscriptIndex(str(p), url_resolver=str)


def test_init_rejects_sqlite_without_expected_table(tmp_path):
    p = tmp_path / "khac.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE bang_khac (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(TranscriptIndexError):
        TranscriptIndex(str(p), url_resolver=str)


def test_init_rejects_empty_index(tmp_path):
    """Schema đúng, 0 row — dấu vết của build bị kill giữa CREATE TABLE và commit INSERT."""
    p = tmp_path / "rong.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE scenes_meta (video_name TEXT, scene_idx INTEGER, start_sec REAL,"
        " end_sec REAL, clip_key TEXT, keyframe_key TEXT, script TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(TranscriptIndexError, match="RỖNG"):
        TranscriptIndex(str(p), url_resolver=str)


def test_init_exposes_row_count(tmp_path):
    assert _index(tmp_path).row_count == 3


def test_suggest_wraps_sqlite_error(tmp_path):
    """Index mở xong rồi file bị hỏng -> TranscriptIndexError, không phải sqlite3.Error thô.

    Không bọc thì `sqlite3.DatabaseError` chạy thẳng lên FastAPI = 500 cho MỌI keystroke.
    """
    db = tmp_path / "t.sqlite"
    idx = _index(tmp_path)
    db.write_bytes(b"bi ghi de sau khi mo")
    with pytest.raises(TranscriptIndexError):
        idx.suggest("khí", 10)


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
    # index chưa nạp (fixture `_reset_module_index` đã set None) -> 503, không phải 500.
    from app.services import transcript as svc

    assert svc.get() is None
    r = _api_client().get("/api/transcript/suggest?q=khí")
    assert r.status_code == 503


def _load_module_index(tmp_path):
    """Nạp index thật vào state module-level (đường thành công của endpoint)."""
    from app.services import transcript as svc

    svc._index, svc._status = _index(tmp_path), "loaded"
    return svc


def test_endpoint_returns_items_serialized_by_response_model(tmp_path):
    """Đường THÀNH CÔNG của endpoint: `TranscriptSuggestItem(**row)` qua `response_model`.

    Các test suggest() ở trên chỉ chạm tầng service (dict thuần) — chưa test nào chứng minh
    dict đó điền được vào schema, tức là thiếu/thừa field là 500 lúc chạy thật.
    """
    _load_module_index(tmp_path)
    r = _api_client().get("/api/transcript/suggest?q=khí&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "khí"
    assert len(body["items"]) == 2
    item = next(i for i in body["items"] if i["scene_idx"] == 0)
    assert item == {
        "video_name": "L21_V001",
        "scene_idx": 0,
        "start_sec": 0.0,
        "end_sec": 4.0,
        "clip_url": "URL:7/scenes/scene_000.mp4",
        "keyframe_url": "URL:7/keyframes/000000.webp",
        "snippet": item["snippet"],  # nội dung snippet do FTS5 sinh, chỉ chốt có mặt
    }
    assert "[" in item["snippet"]


def test_endpoint_uses_config_limit_when_client_omits_it(tmp_path, monkeypatch):
    """`limit` không truyền -> lấy `TRANSCRIPT_SUGGEST_LIMIT`. Chưa test nào chạm nhánh này:
    mọi test cũ đều truyền limit=10 trên fixture 3 row nên `LIMIT ?` chưa từng chặn thật."""
    from app.config import get_settings

    _load_module_index(tmp_path)
    monkeypatch.setenv("TRANSCRIPT_SUGGEST_LIMIT", "1")
    get_settings.cache_clear()
    try:
        r = _api_client().get("/api/transcript/suggest?q=khí")  # không có &limit
        assert r.status_code == 200
        # "khí" khớp 2 scene; config cắt còn 1 -> chứng minh giá trị config được dùng thật.
        assert len(r.json()["items"]) == 1
    finally:
        get_settings.cache_clear()


def test_endpoint_503_when_index_breaks_after_load(tmp_path):
    """Index hỏng SAU khi khởi động -> 503, không phải 500 lặp mãi."""
    _load_module_index(tmp_path)
    (tmp_path / "t.sqlite").write_bytes(b"bi ghi de")
    r = _api_client().get("/api/transcript/suggest?q=khí")
    assert r.status_code == 503


def test_endpoint_rejects_over_long_query(tmp_path):
    """Trần độ dài: endpoint chạy mỗi keystroke, không để paste 100KB vào hot path."""
    _load_module_index(tmp_path)
    r = _api_client().get("/api/transcript/suggest?q=" + "a" * 201)
    assert r.status_code == 422
    assert _api_client().get("/api/transcript/suggest?q=" + "a" * 200).status_code == 200


# Ghim hành vi sanitize: mọi ký tự ngoài `\w` bị loại, nên không input nào dựng được toán tử
# FTS5 (NEAR/phrase/cột/ngoặc). Đổi `_TOKEN_RE` mà quên hệ quả này thì các dòng dưới đỏ.
@pytest.mark.parametrize("bad", [
    'khí" NEAR/2 "hậu',
    "script:khí",
    "khí AND (hậu OR x)",
    "khí^2",
    "^khí",
    "khí -hậu",
    'khí""',
    "*",
    "khí*)*(",
    "NEAR",
])
def test_build_match_never_emits_fts_operator(bad):
    match = build_match(bad)
    assert not set(match) & set('"():^-/*') - {"*"}
    for tok in match.split():
        assert tok.endswith("*") and tok[:-1].isalnum()
    # Và biểu thức sinh ra phải THỰC SỰ chạy được trên FTS5 (không chỉ trông an toàn).
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(script,"
                 " tokenize='unicode61 remove_diacritics 0')")
    conn.execute("INSERT INTO t VALUES ('biến đổi khí hậu')")
    if match:
        conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (match,)).fetchall()
    conn.close()
