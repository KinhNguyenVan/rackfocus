-- videos, scenes, shots, ocr_text, asr_segments, scene_objects.
-- Mới tạo videos + scenes (đủ để cấp video_id/scene_idx cho payload embedding).
-- Các bảng còn lại tạo khi có stage tương ứng (shots, ocr, asr, objects).

CREATE TABLE videos (
    id   BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE scenes (
    id        BIGSERIAL PRIMARY KEY,
    video_id  BIGINT NOT NULL REFERENCES videos(id),
    scene_idx INT NOT NULL,
    start_sec DOUBLE PRECISION NOT NULL,
    end_sec   DOUBLE PRECISION NOT NULL,
    script    TEXT NOT NULL DEFAULT '',
    clip_key  TEXT,
    UNIQUE (video_id, scene_idx)
);
