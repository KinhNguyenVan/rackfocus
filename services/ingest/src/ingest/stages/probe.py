"""ffprobe: duration, fps, resolution, codec."""

from __future__ import annotations

import subprocess


def probe(path: str) -> tuple[int, int, float, int]:
    """Trả (width, height, fps, n_frames_ước_lượng).

    n_frames = -1 khi container không ghi sẵn `nb_frames` (thường gặp với mp4
    stream copy). Khi đó dùng số frame decode thực tế ở stage keyframe làm chuẩn.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.split()

    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den)
    try:
        n = int(out[3])
    except (IndexError, ValueError):
        n = -1
    return w, h, fps, n
