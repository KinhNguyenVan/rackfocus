"""Chọn execution provider + canonicalize text.

`choose_providers` tách thành hàm thuần để test được mà KHÔNG phải nạp bundle 1.8GB —
đây là logic dễ sai âm thầm: chọn sai EP thì hoặc chạy chậm 20x mà không báo gì, hoặc
session chết vì không có CPU chốt hạ.
"""
from __future__ import annotations

from searchcore.encoder.text import CPU_EP, canonicalize, choose_providers

CUDA = "CUDAExecutionProvider"
ROCM = "ROCMExecutionProvider"
COREML = "CoreMLExecutionProvider"


# ── auto ─────────────────────────────────────────────────────────────
def test_auto_khong_co_gpu_thi_dung_cpu_va_khong_canh_bao():
    """Đây là đường mặc định trên gói `onnxruntime` thường. Không được ồn ào."""
    eps, warn = choose_providers([CPU_EP], "auto")
    assert eps == [CPU_EP]
    assert warn == ""


def test_auto_co_cuda_thi_dung_cuda_va_giu_cpu_lam_luoi():
    eps, warn = choose_providers([CUDA, CPU_EP], "auto")
    assert eps == [CUDA, CPU_EP]
    assert warn == ""


def test_auto_nhan_ca_rocm():
    eps, _ = choose_providers([ROCM, CPU_EP], "auto")
    assert eps == [ROCM, CPU_EP]


def test_auto_khong_tu_bat_tensorrt_hay_coreml():
    """TensorRT build engine vài phút ở lần chạy đầu -> không được bật tự động.
    CoreML thì đổi numeric so với CPU nên chỉ bật khi khai rõ."""
    eps, _ = choose_providers(["TensorrtExecutionProvider", COREML, CPU_EP], "auto")
    assert eps == [CPU_EP]


def test_chuoi_rong_hay_none_coi_nhu_auto():
    for want in ("", "   ", None):
        eps, warn = choose_providers([CUDA, CPU_EP], want)
        assert eps == [CUDA, CPU_EP], want
        assert warn == ""


# ── ép tay ───────────────────────────────────────────────────────────
def test_ep_cpu_thi_khong_dung_gpu_du_co_gpu():
    eps, warn = choose_providers([CUDA, CPU_EP], "cpu")
    assert eps == [CPU_EP]
    assert warn == ""


def test_viet_tat_cuda_duoc_map_sang_ten_day_du():
    eps, warn = choose_providers([CUDA, CPU_EP], "cuda")
    assert eps == [CUDA, CPU_EP]
    assert warn == ""


def test_yeu_cau_cuda_ma_khong_co_thi_canh_bao_ro_roi_ve_cpu():
    """KHÔNG raise: lúc thi thà chậm còn hơn pod không lên. Nhưng phải cảnh báo, vì chạy
    CPU im lặng (chậm ~20x) là kết cục tệ nhất."""
    eps, warn = choose_providers([CPU_EP], "cuda")
    assert eps == [CPU_EP]
    assert CUDA in warn
    assert "onnxruntime-gpu" in warn      # chỉ đúng nguyên nhân hay gặp nhất


def test_luon_them_cpu_vao_cuoi():
    """ORT chỉ chuyển EP theo từng node. Thiếu CPU chốt hạ thì một op lạ giết cả session."""
    eps, _ = choose_providers([CUDA, CPU_EP], "cuda")
    assert eps[-1] == CPU_EP
    eps, _ = choose_providers([COREML, CPU_EP], "coreml")
    assert eps == [COREML, CPU_EP]


def test_danh_sach_nhieu_ep_giu_dung_thu_tu_va_bo_cai_thieu():
    eps, warn = choose_providers([ROCM, CPU_EP], "cuda,rocm")
    assert eps == [ROCM, CPU_EP]
    # Chỉ CUDA bị báo thiếu. ROCM vẫn xuất hiện trong warning ở phần liệt kê "đang có",
    # nên phải soi đúng đoạn "yêu cầu [...]" chứ không tìm cả chuỗi.
    assert f"yêu cầu ['{CUDA}']" in warn


def test_khong_nhan_dien_duoc_ten_thi_van_khong_chet():
    eps, warn = choose_providers([CPU_EP], "SomeRandomEP")
    assert eps == [CPU_EP]
    assert "SomeRandomEP" in warn


# ── canonicalize (bẫy số 3 trong docstring của module) ───────────────
def test_canonicalize_bo_dau_cau_lowercase_gop_khoang_trang():
    assert canonicalize("Chùa  Một_Cột, buổi sáng!") == "chùa một cột buổi sáng"


def test_canonicalize_idempotent():
    once = canonicalize("Football Player — celebrating!")
    assert canonicalize(once) == once
