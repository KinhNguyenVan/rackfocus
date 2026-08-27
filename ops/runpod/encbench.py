"""Đo chi phí encode text để chỉnh OMP_NUM_THREADS trên máy thật.

Vì sao cần đo lại trên từng máy: encode chiếm ~93% CPU của một request (53 GFLOP/query),
và số thread tối ưu KHÁC nhau theo kiến trúc CPU. Đo trên M2 (4 P-core + 4 E-core):

    threads=1  wall p50= 938ms  CPU-s/encode=0.94
    threads=2  wall p50= 517ms  CPU-s/encode=1.04
    threads=4  wall p50= 521ms  CPU-s/encode=1.76   <- thêm 70% CPU, latency y nguyên
    threads=8  wall p50=1006ms  CPU-s/encode=4.98   <- tệ hơn cả 1 thread

Đọc kết quả:
  * `wall p50` là latency người dùng thấy -> chọn số thread nhỏ nhất còn ở đáy đường cong.
  * `CPU-s/encode` là chi phí -> nhân với số request/giây để biết cần bao nhiêu core.

Dùng trong pod:
    OMP_NUM_THREADS=<n> python /app/ops/encbench.py <n> [số_query]
"""
import os
import resource
import statistics as st
import sys
import time

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# Phải đặt TRƯỚC khi import onnxruntime, không thì session đã chốt số thread.
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)

from searchcore.encoder import text as textenc   # noqa: E402

SRC = os.environ.get("ENCODER_PATH") or os.environ.get("ENCODER_S3") or ""
if not SRC:
    sys.exit("Cần ENCODER_PATH hoặc ENCODER_S3 (bundle text tower ONNX)")

enc = textenc.load(SRC, name=os.environ.get("ENCODER_NAME", ""))

QS = [f"football player celebrating a goal number {i}" for i in range(N)]
for q in QS[:3]:
    enc.encode_one(q)


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


lat = []
c0, t0 = cpu(), time.perf_counter()
for q in QS:
    t = time.perf_counter()
    enc.encode_one(q)
    lat.append((time.perf_counter() - t) * 1000)
wall = time.perf_counter() - t0
used = cpu() - c0

p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
print(f"threads={THREADS} n={N}  wall p50={st.median(lat):7.1f}ms p95={p95:7.1f}ms  "
      f"CPU-s/encode={used / N:.4f}  (CPU/wall={used / wall:.2f})")
print(f"  -> 1 core làm được {N / used:.2f} encode/s; "
      f"cần {used / N * 100:.0f} core cho 100 rps ở 100% util")
