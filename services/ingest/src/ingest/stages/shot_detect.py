"""PySceneDetect (coarse) -> TransNetV2 (refine). CPU-bound, chạy nhiều worker.

Hiện dùng thẳng TransNetV2 (đủ nhanh trên GPU, ranh giới sát). `predict_video`
giữ toàn bộ frame 48x27 trong RAM nên phải giải phóng ngay sau mỗi video.
"""

from __future__ import annotations


def load_transnet():
    """Khởi tạo model TransNetV2 (nạp 1 lần, tái dùng cho mọi video).

    `transnetv2` là module trong thư mục `inference/` của repo TransNetV2-master —
    thêm thư mục đó vào PYTHONPATH trước khi import (xem README ingest).
    """
    from transnetv2 import TransNetV2

    return TransNetV2()


def detect_shots(video_path: str, model, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Trả danh sách shot [(start_frame, end_frame), ...] theo frame index.

    threshold: ngưỡng boundary của TransNetV2; hạ 0.3-0.4 nếu sót gradual cut.
    """
    video_frames = single_pred = all_pred = None
    try:
        video_frames, single_pred, all_pred = model.predict_video(video_path)
        scenes = model.predictions_to_scenes(single_pred, threshold=threshold)
        # predictions_to_scenes trả về np.ndarray (N, 2); ép về tuple int cho gọn.
        return [(int(s), int(e)) for s, e in scenes]
    finally:
        # Bắt buộc: video_frames giữ toàn bộ frame 48x27, không xoá thì RAM leo dần.
        del video_frames, single_pred, all_pred
