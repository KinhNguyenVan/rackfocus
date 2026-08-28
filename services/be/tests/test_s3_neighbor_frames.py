"""Test `get_neighbor_frames` với S3 giả -- chiến lược LIST, không phải logic cắt.

Tách khỏi test_s3_neighbors.py (test `neighbor_slice_bounds`, hàm thuần): ở đây kiểm
đúng phần dễ sai nhất và trước giờ CHƯA có test nào — cách gọi `list_objects_v2`.

Vì sao đáng test: bản cũ `paginator.paginate` cả thư mục video cho MỖI click. Trên
corpus hiện tại hai cách nhanh bằng nhau (đo thật: 860ms vs 896ms — xem docstring
`get_neighbor_frames`), nhưng cách cũ tăng tuyến tính theo số object của video còn cách
này thì không. Không có test thì lần refactor sau rất dễ vô tình quay lại kiểu cũ — và
nó vẫn trả kết quả ĐÚNG, chỉ chậm dần theo kích thước video, nên không có gì báo.
"""
from __future__ import annotations

import pytest

from app.clients.s3 import AWSStorageHelper

PREFIX = "Keyframes_L21_a/keyframes/L21_V001/"


class FakeS3:
    """Bắt chước đúng ba tham số mình dùng: Prefix, StartAfter, MaxKeys.

    `calls` ghi lại từng request để test khẳng định được về SỐ LƯỢNG và KÍCH THƯỚC
    request, không chỉ về kết quả.
    """

    def __init__(self, frames: list[int], prefix: str = PREFIX) -> None:
        self.keys = sorted(f"{prefix}{f:06d}.webp" for f in frames)
        self.calls: list[dict] = []

    def list_objects_v2(self, *, Bucket, Prefix, StartAfter="", MaxKeys=1000):
        self.calls.append({"StartAfter": StartAfter, "MaxKeys": MaxKeys})
        hit = [k for k in self.keys if k.startswith(Prefix) and k > StartAfter]
        return {"Contents": [{"Key": k} for k in hit[:MaxKeys]]}


def helper_for(frames: list[int], prefix: str = PREFIX) -> AWSStorageHelper:
    # Bỏ qua __init__ để không cần boto3/credential.
    h = AWSStorageHelper.__new__(AWSStorageHelper)
    h.bucket = "aic-bucket-2026"
    h.s3_client = FakeS3(frames, prefix)
    return h


def key(frame: int, prefix: str = PREFIX) -> str:
    return f"{prefix}{frame:06d}.webp"


def frames_of(keys: list[str]) -> list[int]:
    return [int(k.rsplit("/", 1)[1].removesuffix(".webp")) for k in keys]


# ── một mỏ neo ───────────────────────────────────────────────────────
def test_mot_moc_tra_ca_frame_hien_tai():
    """Phải CÓ frame hiện tại: browse.py đánh dấu `is_current` dựa vào nó, và UI đặt
    nó ở giữa dải."""
    h = helper_for(list(range(0, 200, 10)))
    got = frames_of(h.get_neighbor_frames(key(100), before=2, after=2))
    assert got == [80, 90, 100, 110, 120]


def test_so_request_khong_tang_theo_do_dai_video():
    """Điểm mấu chốt: số request phải KHÔNG phụ thuộc độ dài video.

    Không assert "ít request" chung chung, vì vòng `before` vẫn xin MaxKeys=1000 hai
    lần — trên video 2-3k object thì bằng luôn cách paginate cũ (đo thật: 860 vs
    896ms). Thứ duy nhất thật sự khác là ĐỘ CO GIÃN, nên test đúng thứ đó: video dài
    gấp 25 lần mà số request không được tăng.
    """
    small = helper_for(list(range(0, 2_000, 5)))        # 400 frame
    big = helper_for(list(range(0, 50_000, 5)))         # 10.000 frame
    small.get_neighbor_frames(key(1_000), before=25, after=25)
    big.get_neighbor_frames(key(25_000), before=25, after=25)
    assert len(big.s3_client.calls) == len(small.s3_client.calls), (
        f"số request tăng theo độ dài video: {len(small.s3_client.calls)} -> "
        f"{len(big.s3_client.calls)} — đã quay về kiểu paginate cả thư mục?")
    # Không request nào quét từ đầu thư mục.
    assert all(c["StartAfter"] != "" for c in big.s3_client.calls), big.s3_client.calls


def test_moc_dau_video():
    h = helper_for(list(range(0, 100, 10)))
    got = frames_of(h.get_neighbor_frames(key(0), before=5, after=2))
    assert got == [0, 10, 20]


def test_before_0_va_after_0():
    h = helper_for(list(range(0, 100, 10)))
    assert frames_of(h.get_neighbor_frames(key(50), before=0, after=0)) == [50]


def test_frame_thua_van_lay_du_before():
    """Keyframe KHÔNG cách đều nhau (7/shot, shot dài ngắn khác nhau) nên cửa sổ số
    frame ban đầu có thể không đủ -- phải nới ra chứ không trả thiếu."""
    h = helper_for([0, 5000, 10_000, 15_000, 20_000, 25_000])
    got = frames_of(h.get_neighbor_frames(key(20_000), before=3, after=1))
    assert got == [5000, 10_000, 15_000, 20_000, 25_000]


# ── hai mỏ neo (temporal search) ─────────────────────────────────────
def test_hai_moc_tra_tat_ca_frame_o_giua():
    h = helper_for(list(range(0, 300, 10)))
    got = frames_of(h.get_neighbor_frames(key(100), before=2, after=2, to_key=key(150)))
    assert got == [80, 90, 100, 110, 120, 130, 140, 150, 160, 170]


def test_hai_moc_dao_thu_tu_van_dung():
    """Chain temporal luôn đúng thứ tự, nhưng đảo vào không được ra kết quả khác."""
    h = helper_for(list(range(0, 300, 10)))
    a = h.get_neighbor_frames(key(100), before=1, after=1, to_key=key(150))
    b = helper_for(list(range(0, 300, 10))).get_neighbor_frames(
        key(150), before=1, after=1, to_key=key(100))
    assert frames_of(a) == frames_of(b)


def test_hai_moc_khoang_cach_lon_van_lay_du_after():
    """Khoảng giữa dài hơn một trang 1000 object thì phải đi tiếp trang, không cắt sớm."""
    h = helper_for(list(range(0, 30_000, 10)))     # 3000 frame
    got = frames_of(h.get_neighbor_frames(
        key(1_000), before=0, after=3, to_key=key(20_000)))
    assert got[0] == 1_000
    assert 20_000 in got
    assert got[-3:] == [20_010, 20_020, 20_030]


def test_hai_moc_khac_video_thi_raise():
    """Cả 2 hit của TemporalChain luôn cùng video, nên khác video là lỗi của caller --
    raise chứ không im lặng trả rác."""
    h = helper_for(list(range(0, 100, 10)))
    with pytest.raises(ValueError, match="cùng video"):
        h.get_neighbor_frames(key(50), to_key=key(60, "Keyframes_L21_a/keyframes/L21_V002/"))


def test_to_key_none_giong_het_1_moc():
    h1 = helper_for(list(range(0, 200, 10)))
    h2 = helper_for(list(range(0, 200, 10)))
    assert (h1.get_neighbor_frames(key(100), before=2, after=2)
            == h2.get_neighbor_frames(key(100), before=2, after=2, to_key=None))


# ── lỗi ──────────────────────────────────────────────────────────────
def test_key_sai_dinh_dang_thi_raise():
    h = helper_for([10, 20])
    with pytest.raises(ValueError, match="không hợp lệ"):
        h.get_neighbor_frames("Keyframes_L21_a/keyframes/L21_V001/abc.webp")


def test_moc_khong_ton_tai_tren_s3_thi_raise():
    """Thà báo lỗi rõ còn hơn trả một dải frame không chứa mỏ neo -- UI sẽ hiển thị
    dải đó như thể đúng."""
    h = helper_for([0, 10, 20, 30])
    with pytest.raises(ValueError, match="Không tìm thấy frame"):
        h.get_neighbor_frames(key(15), before=1, after=1)
