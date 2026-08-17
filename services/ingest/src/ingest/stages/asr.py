"""ASR tiếng Việt: chunkformer (khanhld/chunkformer-ctc-large-vie).

`endless_decode(return_timestamps=True)` cho phép transcribe long-form (cả video)
kèm mốc thời gian từng đoạn. Sau đó gán script cho từng scene theo overlap thời gian.
"""

from __future__ import annotations

DEFAULT_MODEL = "khanhld/chunkformer-ctc-large-vie"


def load_asr_model(name: str = DEFAULT_MODEL):
    """Nạp ChunkFormerModel (nạp 1 lần, tái dùng cho mọi video)."""
    from chunkformer import ChunkFormerModel

    return ChunkFormerModel.from_pretrained(name)


def _to_seconds(value) -> float:
    """Chuẩn hoá timestamp về giây.

    chunkformer trả timestamp dạng chuỗi "HH:MM:SS:ms" (đôi khi "HH:MM:SS.ms")
    hoặc số giây. Hàm này chịu được cả hai — chỉnh lại nếu format thực khác.
    """
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # "HH:MM:SS:ms" -> tách phần ms ở cụm cuối; hoặc "HH:MM:SS.ms".
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 4:  # HH:MM:SS:ms
            h, m, sec, ms = parts
            return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
        if len(parts) == 3:  # HH:MM:SS(.ms)
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
    return float(s)


def transcribe(video_path: str, model, **decode_kwargs) -> list[dict]:
    """Transcribe cả video -> list[{"start", "end", "text"}] (start/end tính bằng giây).

    decode_kwargs override tham số endless_decode nếu cần (chunk_size, context...).
    """
    params = dict(
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=14400,  # giây; đủ cho video dài
        return_timestamps=True,
    )
    params.update(decode_kwargs)

    result = model.endless_decode(audio_path=video_path, **params)
    return _normalize_segments(result)


def _normalize_segments(result) -> list[dict]:
    """Ép output endless_decode về list[{"start","end","text"}] chuẩn (giây).

    Chấp nhận: list các dict segment, hoặc chuỗi text thuần (không timestamp).
    """
    if isinstance(result, str):
        return [{"start": 0.0, "end": 0.0, "text": result.strip()}]

    segments = []
    for seg in result:
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or seg.get("decode") or "").strip()
        if not text:
            continue
        segments.append({
            "start": _to_seconds(seg.get("start", 0)),
            "end": _to_seconds(seg.get("end", seg.get("start", 0))),
            "text": text,
        })
    return segments


def assign_script_to_scenes(scenes: list[dict], segments: list[dict]) -> None:
    """Gán `script` cho mỗi scene = nối text các segment overlap thời gian scene.

    Overlap xác định theo midpoint segment nằm trong [start_time, end_time] của
    scene — đơn giản, tránh 1 segment bị đếm cho 2 scene. Ghi trực tiếp vào scene dict.
    """
    for sc in scenes:
        lo, hi = float(sc["start_time"]), float(sc["end_time"])
        texts = [
            seg["text"]
            for seg in segments
            if lo <= (seg["start"] + seg["end"]) / 2.0 <= hi
        ]
        sc["script"] = " ".join(texts).strip()
