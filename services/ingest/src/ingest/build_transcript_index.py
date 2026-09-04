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

CHẠY BẰNG CLI ở cuối file — không có caller nào khác. `build_index.py` KHÔNG gọi module này
(nó đang là stub 1 dòng), nên `transcript.sqlite` nằm NGOÀI hợp đồng checksum của snapshot
(`manifest.json`): ghép một DB dựng theo `snapshots/v1` với `snapshots/v2` là không phát hiện
được ở tầng nào. Cho tới khi `build_index.py` được viết, người vận hành phải tự đảm bảo
`SNAPSHOT_S3` lúc dựng DB trùng bản snapshot mà core đang chạy — xem docs/transcript-search.md.
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

    `rows` rỗng -> RAISE, không ghi file. Một DB đúng schema với 0 row là ca lỗi TỆ NHẤT
    của tính năng này: BE mở được, endpoint trả 200 {"items": []} cho mọi keyword, và
    "không có kết quả" trông y như "không ai nói câu đó". Rỗng ở đây gần như luôn nghĩa là
    prefix S3 sai / thiếu credential / payload lệch transcript, tức là lỗi cấu hình cần
    thấy NGAY lúc build, không phải lúc thi.
    """
    if not rows:
        raise ValueError(
            f"0 row — KHÔNG ghi index rỗng ({out_path}). Kiểm tra: payload.parquet có đúng "
            "video không, thư mục transcript có file <video_name>.json không, và tên video "
            "hai bên có trùng không."
        )
    conn = sqlite3.connect(out_path)
    try:
        create_schema(conn)
        add_rows(conn, rows)
        finalize(conn)
    finally:
        conn.close()
    return out_path


# =====================================================================
# Build từ snapshot payload.parquet + transcript ASR (KHÔNG cần Postgres)
# =====================================================================
# Vì sao cần path này: text transcript (`scene.script`) không có trên S3 dạng scene JSON,
# và Postgres ingest thường không còn/không nối được lúc online. Nhưng snapshot đang deploy
# đã có sẵn ranh giới scene + clip_key/keyframe_key ĐÚNG (payload.parquet), còn ASR gốc nằm
# ở `Transcripts_*/transcripts/<video_name>.json`. Ghép hai cái = index transcript đầy đủ,
# key trùng khít cái vector search trả về (click mở đúng scene clip).

def scenes_from_payload(payload_path: str) -> dict[str, list[dict]]:
    """`payload.parquet` -> {video_name: [scene dict sắp theo scene_idx]}.

    Mỗi scene có thể có nhiều keyframe (nhiều row) nhưng cùng start/end/clip_key -> dedup
    theo (video_name, scene_idx). clip_key/keyframe_key ở payload là URL tuyệt đối, giữ nguyên.
    """
    import pyarrow.parquet as pq

    cols = ["video_name", "scene_idx", "start_sec", "end_sec", "clip_key", "keyframe_key"]
    d = pq.read_table(payload_path, columns=cols).to_pydict()
    by_video: dict[str, list[dict]] = {}
    seen: set[tuple[str, int]] = set()
    for i in range(len(d["video_name"])):
        vn = d["video_name"][i]
        si = int(d["scene_idx"][i])
        if (vn, si) in seen:
            continue
        seen.add((vn, si))
        by_video.setdefault(vn, []).append({
            "scene_idx": si,
            "start_sec": float(d["start_sec"][i]),
            "end_sec": float(d["end_sec"][i]),
            "clip_key": d["clip_key"][i],
            "keyframe_key": d["keyframe_key"][i] or "",
        })
    for scenes in by_video.values():
        scenes.sort(key=lambda s: s["scene_idx"])
    return by_video


def assign_segments_to_scenes(scenes: list[dict], segments: list[dict]) -> dict[int, str]:
    """Gán script cho từng scene — sao ĐÚNG `stages.asr.assign_script_to_scenes`.

    Script scene = nối text mọi segment có MIDPOINT ((start+end)/2) nằm trong [start_sec,
    end_sec] của scene. Cùng công thức lúc ingest nên scene có script khớp cờ `has_speech`.
    Trả {scene_idx: script} chỉ cho scene có text (bỏ scene rỗng).
    """
    out: dict[int, str] = {}
    for sc in scenes:
        lo, hi = sc["start_sec"], sc["end_sec"]
        texts = [
            seg["text"]
            for seg in segments
            if lo <= (float(seg["start"]) + float(seg["end"])) / 2.0 <= hi
        ]
        script = " ".join(texts).strip()
        if script:
            out[sc["scene_idx"]] = script
    return out


def rows_from_payload_and_transcripts(payload_path: str, transcripts_dir: str) -> list[dict]:
    """Ghép payload (ranh giới scene + key) với ASR (`<transcripts_dir>/<video_name>.json`).

    Chỉ video có trong payload (đã embed = phát được clip) mới ra row. Trả rows cho
    `build_transcript_db`.
    """
    import glob
    import json
    import os

    by_video = scenes_from_payload(payload_path)
    rows: list[dict] = []
    missing = 0
    bad = 0
    for vn, scenes in by_video.items():
        tpath = os.path.join(transcripts_dir, f"{vn}.json")
        if not os.path.exists(tpath):
            missing += 1
            continue
        # File CÓ nhưng không đọc/parse được, hoặc thiếu field `segments`: trước đây
        # `.get("segments", [])` biến ca này thành "video không có thoại" và nó chỉ hiện ra
        # dưới dạng số row thấp hơn kỳ vọng — lẫn hẳn vào `missing` (vốn chỉ đếm
        # file-not-exists). Đếm riêng + in từng file để tải S3 dở dang không im lặng trôi qua.
        try:
            with open(tpath, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as ex:
            bad += 1
            print(f"  [BAD] {tpath}: {type(ex).__name__}: {ex}")
            continue
        segments = doc.get("segments") if isinstance(doc, dict) else None
        if not isinstance(segments, list):
            bad += 1
            print(f"  [BAD] {tpath}: thiếu field 'segments' dạng list")
            continue
        scripts = assign_segments_to_scenes(scenes, segments)
        for sc in scenes:
            script = scripts.get(sc["scene_idx"])
            if not script:
                continue
            rows.append({
                "video_name": vn,
                "scene_idx": sc["scene_idx"],
                "start_sec": sc["start_sec"],
                "end_sec": sc["end_sec"],
                "clip_key": sc["clip_key"],
                "keyframe_key": sc["keyframe_key"],
                "script": script,
            })
    n_tx = len(glob.glob(os.path.join(transcripts_dir, "*.json")))
    print(f"payload: {len(by_video)} video; transcript json: {n_tx}; "
          f"video không có transcript: {missing}; json lỗi/sai schema: {bad}; "
          f"rows(scene có thoại): {len(rows)}")
    return rows


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


def _repo_root() -> str:
    """.../services/ingest/src/ingest/build_transcript_index.py -> repo root (4 cấp lên)."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _load_env(root: str) -> None:
    """Nạp `.env` vào os.environ (AWS_* cho boto3) nếu biến chưa có sẵn."""
    import os

    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def download_s3_inputs(dest_root: str, snapshot_s3: str | None = None,
                       transcript_prefix: str = "Transcripts_") -> tuple[str, str]:
    """Tải `payload.parquet` (từ SNAPSHOT_S3) + mọi `Transcripts_*/transcripts/*.json` về máy.

    Trả `(payload_path, transcripts_dir)`. File đã có thì bỏ qua (rẻ khi chạy lại).
    """
    import os

    from .storage import _bucket, get_client, parse_s3_uri

    client = get_client()
    bucket = _bucket()
    os.makedirs(dest_root, exist_ok=True)
    tx_dir = os.path.join(dest_root, "transcripts")
    os.makedirs(tx_dir, exist_ok=True)

    snap = snapshot_s3 or os.environ.get("SNAPSHOT_S3", "")
    prefix = parse_s3_uri(snap)[1] if snap.startswith("s3://") else ""
    payload_key = f"{prefix}/payload.parquet" if prefix else "payload.parquet"
    payload_path = os.path.join(dest_root, "payload.parquet")
    client.download_file(bucket, payload_key, payload_path)
    print(f"payload.parquet ({payload_key}) -> {payload_path}")

    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=transcript_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/transcripts/" in key and key.endswith(".json"):
                dest = os.path.join(tx_dir, os.path.basename(key))
                if not os.path.exists(dest):
                    client.download_file(bucket, key, dest)
                n += 1
    print(f"transcript json: {n} file -> {tx_dir}")
    # 0 file = prefix sai, bucket sai, hoặc credential không thấy object nào. Trước đây chỉ
    # in "transcript json: 0 file" rồi đi tiếp -> ghép ra 0 row -> ghi ra một DB rỗng nhưng
    # hợp lệ. Chặn ngay ở đây thì lỗi hiện ra ở chỗ gây ra nó, kèm prefix để sửa.
    if n == 0:
        raise RuntimeError(
            f"không thấy file transcript nào dưới s3://{bucket}/{transcript_prefix}* "
            "(cần key dạng '<prefix>/transcripts/<video_name>.json'). Kiểm tra "
            "transcript_prefix, AWS_BUCKET_NAME và credential."
        )
    return payload_path, tx_dir


def _fresh_tmp_path(db_path: str) -> str:
    """Đường dẫn tạm cạnh `db_path`, đã dọn tàn dư của lần chạy trước.

    Cạnh (không phải /tmp) để `os.replace` là rename trong CÙNG filesystem = atomic.
    Phải dọn trước: `build_transcript_db` mở-append nên tmp sót lại từ lần bị kill sẽ
    làm row của hai lần build cộng dồn.
    """
    import os

    tmp = db_path + ".tmp"
    for suffix in ("", "-journal", "-wal", "-shm"):
        if os.path.exists(tmp + suffix):
            os.remove(tmp + suffix)
    return tmp


def _build_from_ingest_outputs(out_root: str, db_path: str) -> None:
    """[legacy] Dựng từ output ingest (`<name>/scene_*.json`) + Postgres cấp video_id."""
    import os

    from .db import get_conn, get_or_create_video_id

    tmp = _fresh_tmp_path(db_path)
    conn = sqlite3.connect(tmp)
    pg = get_conn()
    total = 0
    try:
        create_schema(conn)
        for name, keyframes, scenes in _iter_video_outputs(out_root):
            video_id = get_or_create_video_id(pg, name)
            rows = scene_transcript_rows(video_id, name, keyframes, scenes)
            total += add_rows(conn, rows)
            print(f"{name}: +{len(rows)} scene có transcript")
        if not total:
            raise ValueError(f"0 scene có transcript dưới {out_root} — không ghi index rỗng")
        finalize(conn)
    finally:
        conn.close()
        pg.close()
    os.replace(tmp, db_path)
    print(f"Xong: {total} scene -> {db_path}")


def main() -> None:
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Dựng transcript.sqlite (FTS5) cho keyword search lời thoại scene")
    # Mode khuyến nghị: ghép snapshot payload.parquet + ASR transcript (không cần Postgres).
    ap.add_argument("--from-s3", action="store_true",
                    help="Tải payload.parquet (SNAPSHOT_S3) + mọi Transcripts_* từ S3 rồi dựng")
    ap.add_argument("--payload", help="payload.parquet cục bộ (dùng kèm --transcripts)")
    ap.add_argument("--transcripts", help="Thư mục chứa <video_name>.json (ASR segments)")
    ap.add_argument("--work-dir", default=None,
                    help="Nơi tải S3 về khi --from-s3 (mặc định <repo>/.tmp/transcript_build)")
    # Mode cũ: output ingest cục bộ + Postgres.
    ap.add_argument("--out-root", help="[legacy] output ingest (<name>/scene_*.json), cần Postgres")
    ap.add_argument("--db", default=None, help=f"File sqlite ra (mặc định cạnh payload / {DB_FILENAME})")
    args = ap.parse_args()

    if args.from_s3 or args.payload:
        root = _repo_root()
        if args.from_s3:
            _load_env(root)
            work = args.work_dir or os.path.join(root, ".tmp", "transcript_build")
            payload, tx_dir = download_s3_inputs(work)
        else:
            payload, tx_dir = args.payload, args.transcripts
            if not tx_dir:
                ap.error("--payload phải đi kèm --transcripts")
        db_path = args.db or os.path.join(os.path.dirname(os.path.abspath(payload)), DB_FILENAME)
        # GHI ATOMIC: dựng vào <db>.tmp rồi `os.replace` sang chỗ thật. Trước đây xoá
        # `db_path` NGAY TỪ ĐẦU rồi mới đi tải S3 + ghép — nên chạy lại lệnh với
        # SNAPSHOT_S3/prefix sai là XOÁ MẤT index đang dùng, đổi lấy một file rỗng. Với
        # thứ tự này, mọi lỗi (0 file S3, 0 row, Ctrl-C giữa build) đều để nguyên bản cũ.
        tmp = _fresh_tmp_path(db_path)
        rows = rows_from_payload_and_transcripts(payload, tx_dir)
        build_transcript_db(rows, tmp)
        os.replace(tmp, db_path)
        print(f"Xong: {len(rows)} scene có thoại -> {db_path}")
        return

    if not args.out_root:
        ap.error("cần một trong: --from-s3 | --payload+--transcripts | --out-root")
    db_path = args.db or os.path.join(args.out_root, DB_FILENAME)
    _build_from_ingest_outputs(args.out_root, db_path)


if __name__ == "__main__":
    main()
