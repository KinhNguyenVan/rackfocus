"""Đọc env: SNAPSHOT_DIR, OMP_NUM_THREADS, EF_SEARCH, ENCODER_PATH, SOCKET_PATH."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    socket_path: str = os.getenv("SC_SOCKET_PATH", "/var/run/searchcore/sc.sock")
    tcp_port: int = int(os.getenv("SC_TCP_PORT", "50051"))
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "/data/snapshots/current")
    encoder_path: str = os.getenv("ENCODER_PATH", "")
    ef_search: int = int(os.getenv("FAISS_EF_SEARCH", "64"))
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "800"))
    omp_threads: int = int(os.getenv("OMP_NUM_THREADS", "4"))
    # Bật khi chưa có snapshot thật: trả kết quả giả để test đường ống.
    stub_mode: bool = os.getenv("SC_STUB_MODE", "1") == "1"


cfg = Config()
