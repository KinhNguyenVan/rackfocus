"""Keyword search transcript: mở SQLite FTS5 (ship kèm snapshot) và gợi ý scene.

Nguồn dữ liệu build ở `services/ingest/src/ingest/build_transcript_index.py`. Online KHÔNG
có Postgres nên transcript nằm ở file bất biến này; BE mở read-only lúc khởi động
(`main.py` lifespan) và giữ 1 instance module-level.

Khớp keyword = FTS5 prefix MATCH + xếp hạng bm25() + snippet() highlight. KHÔNG gọi LLM:
dropdown as-you-type phải rẻ và nhanh, không kẹt round-trip mỗi keystroke.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable

log = logging.getLogger("app.services.transcript")

# Token = cụm chữ/số liên tiếp (giữ Unicode để không cắt vụn tiếng Việt có dấu). Mọi ký tự
# khác (kể cả toán tử FTS5 " * : ( )) bị loại -> query người dùng không thể phá cú pháp MATCH.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_QUERY_SQL = (
    "SELECT m.video_name, m.scene_idx, m.start_sec, m.end_sec, m.clip_key, m.keyframe_key,"
    "       snippet(transcript_fts, 0, '[', ']', '…', 12) AS snippet "
    "FROM transcript_fts JOIN scenes_meta m ON m.rowid = transcript_fts.rowid "
    "WHERE transcript_fts MATCH ? ORDER BY bm25(transcript_fts) LIMIT ?"
)


def build_match(q: str) -> str:
    """Query người dùng -> biểu thức FTS5 MATCH prefix an toàn.

    Mỗi token thành `token*` (prefix, hợp autocomplete), nối bằng khoảng trắng = AND ngầm.
    Không token hợp lệ -> "" (caller phải coi là "không tìm").

    Lowercase token: FTS5 nhận diện AND/OR/NOT là toán tử KHI viết hoa; hạ về thường khiến
    "OR" người dùng gõ thành prefix "or*" vô hại (matching vốn không phân biệt hoa/thường)."""
    tokens = _TOKEN_RE.findall((q or "").lower())
    return " ".join(f"{t}*" for t in tokens)


def _default_resolver() -> Callable[[str], str]:
    """Dựng resolver key->URL công khai bằng S3 helper (giống browse.py). Tách ra để test
    inject được resolver thuần, không cần AWS."""
    from ..clients.s3 import AWSStorageHelper

    helper = AWSStorageHelper()

    def resolve(key: str) -> str:
        if not key:
            return ""
        if key.startswith("http://") or key.startswith("https://"):
            return key  # ingest đôi khi đã ghi URL tuyệt đối
        return helper.get_s3_public_url(key)

    return resolve


class TranscriptIndex:
    """SQLite FTS5 read-only. File bất biến; mở connection MỚI mỗi truy vấn.

    FastAPI chạy endpoint sync trong threadpool -> nhiều thread có thể suggest() cùng lúc.
    Dùng chung 1 connection sqlite qua các thread là nguồn lỗi khó lần; mở read-only mỗi
    lần (file đã ở page cache) rẻ hơn nhiều so với chi phí đó.
    """

    def __init__(self, db_path: str, url_resolver: Callable[[str], str] | None = None):
        self._uri = f"file:{db_path}?mode=ro"
        # Mở-đóng 1 lần lúc khởi tạo để FAIL-CLOSED nếu thiếu file (không im lặng trả rỗng).
        sqlite3.connect(self._uri, uri=True).close()
        self._resolve = url_resolver or _default_resolver()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._uri, uri=True)

    def suggest(self, q: str, limit: int) -> list[dict]:
        match = build_match(q)
        if not match:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(_QUERY_SQL, (match, limit)).fetchall()
        finally:
            conn.close()
        out = []
        for video_name, scene_idx, start_sec, end_sec, clip_key, keyframe_key, snippet in rows:
            out.append({
                "video_name": video_name,
                "scene_idx": scene_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "clip_url": self._resolve(clip_key or ""),
                "keyframe_url": self._resolve(keyframe_key or ""),
                "snippet": snippet,
            })
        return out

    def close(self) -> None:
        # Không giữ connection sống (mở theo từng truy vấn) -> không có gì để đóng.
        pass


# ── instance module-level (nạp ở lifespan) ──────────────────────────
_index: TranscriptIndex | None = None


def load(db_path: str) -> None:
    """Mở index. Lỗi (thiếu file) ném ra để lifespan log — endpoint sẽ trả 503."""
    global _index
    _index = TranscriptIndex(db_path)
    log.info("transcript index mở từ %s", db_path)


def get() -> TranscriptIndex | None:
    return _index
