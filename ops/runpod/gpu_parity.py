"""So vector GPU vs CPU trên cùng bundle, cùng query.

Vì sao phải so: chuyển encoder sang GPU KHÔNG đổi file model.onnx (fp32 chạy được cả
hai), nhưng đổi thứ tự cộng dồn và có thể đổi cả độ chính xác (TF32 trên Ampere+ dùng
mantissa 10 bit thay vì 23). Qua 27 layer, sai số tích luỹ có thể đảo thứ tự các hit gần
bằng điểm nhau — đúng loại lỗi mà docstring của encoder/text.py cảnh báo: vector lệch mà
KHÔNG có lỗi nào báo.

Ngưỡng đọc kết quả (cosine giữa vector CPU và GPU của CÙNG một câu):
  >= 0.99999  coi như y hệt, yên tâm
  >= 0.9999   lệch ở mức fp32 bình thường, thứ hạng top-300 gần như không đổi
  <  0.999    có gì sai (nhầm bundle, TF32 bật, hoặc EP đặt node lên thiết bị khác)

Dùng:
    python ops/runpod/gpu_parity.py            # đọc ENCODER_S3/ENCODER_PATH từ env
    SC_CUDA_TF32=1 python ops/runpod/gpu_parity.py   # xem TF32 lệch bao nhiêu
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Chạy từ repo thì cần thêm đường dẫn; chạy trong ảnh pod thì PYTHONPATH đã có
# /app/core/src nên nhánh này không tồn tại và bị bỏ qua.
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "services", "core", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from searchcore.encoder import text as textenc   # noqa: E402

QUERIES = [
    "football player celebrating a goal",
    "students sitting in a classroom",
    "a woman cooking in the kitchen",
    "heavy traffic on a city street",
    "farmer working in a rice field",
    "news anchor in a television studio",
    "aerial view of a bridge over a river",
    "market vendor selling fruit",
]

SRC = os.environ.get("ENCODER_PATH") or os.environ.get("ENCODER_S3") or ""
if not SRC:
    sys.exit("Cần ENCODER_PATH hoặc ENCODER_S3")


def load(provider: str):
    os.environ["SC_ENCODER_PROVIDERS"] = provider
    t0 = time.perf_counter()
    enc = textenc.load(SRC, name=os.environ.get("ENCODER_NAME", ""))
    return enc, time.perf_counter() - t0


gpu, t_gpu = load("auto")
if not gpu.on_gpu:
    sys.exit(f"Không có GPU (EP={gpu.providers}). Chạy trên pod GPU với ảnh build "
             f"--build-arg GPU=1, hoặc bỏ qua kiểm tra này.")

cpu, t_cpu = load("cpu")

# Hai session cùng lúc = 2 x model trong RAM (~3.6GB). Chấp nhận vì đây là script chạy
# tay một lần, không phải đường phục vụ request.
print(f"GPU EP={gpu.providers[0]} (nạp {t_gpu:.0f}s) | CPU (nạp {t_cpu:.0f}s) | "
      f"TF32={os.environ.get('SC_CUDA_TF32', '0')}")

vg = gpu.encode(QUERIES)
vc = cpu.encode(QUERIES)

cos = np.sum(vg * vc, axis=1)          # cả hai đã L2-normalize
worst = int(np.argmin(cos))
print(f"cosine: min={cos.min():.7f} mean={cos.mean():.7f}")
print(f"  câu lệch nhất: {QUERIES[worst]!r} -> {cos[worst]:.7f}")
print(f"  max |Δ| trên một chiều: {np.abs(vg - vc).max():.2e}")

# Đo tốc độ luôn, vì đó là lý do chuyển sang GPU.
for label, enc in (("GPU", gpu), ("CPU", cpu)):
    enc.encode(QUERIES[:2])                       # warm
    t0 = time.perf_counter()
    for q in QUERIES:
        enc.encode_one(q)
    ms = (time.perf_counter() - t0) / len(QUERIES) * 1000
    print(f"  {label}: {ms:7.1f} ms/query")

if cos.min() < 0.999:
    sys.exit(f"LỆCH QUÁ LỚN (min cosine {cos.min():.6f}) — xem ngưỡng ở đầu file")
print("OK")
