"""Đọc env. Xem docs/search-design.md §2 (kiểm tra snapshot) và §7 (đồng thời)."""
import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


@dataclass(frozen=True)
class Config:
    socket_path: str = os.getenv("SC_SOCKET_PATH", "/var/run/searchcore/sc.sock")
    tcp_port: int = _int("SC_TCP_PORT", 50051)

    # ── Snapshot ─────────────────────────────────────────────────────
    # SNAPSHOT_DIR (bind mount) ưu tiên hơn SNAPSHOT_S3. Có cả hai thì dùng bind mount
    # và log cảnh báo — hai nguồn cùng lúc là cách nhanh nhất để serve sai bản.
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "")
    snapshot_s3: str = os.getenv("SNAPSHOT_S3", "")

    # ── Encoder ──────────────────────────────────────────────────────
    # Bundle ONNX *text tower*. KHÁC bundle vision mà ingest dùng: export_siglip_onnx.py
    # chỉ export get_image_features. Xem docs/search-design.md §5.
    encoder_s3: str = os.getenv("ENCODER_S3", "")
    encoder_path: str = os.getenv("ENCODER_PATH", "")

    # ── Hợp đồng offline↔online (docs/search-design.md §2) ───────────
    # Hai field này TRƯỚC ĐÂY KHÔNG TỒN TẠI, nên check manifest của snapshot là no-op.
    # dim=0 nghĩa là "lấy theo manifest, không kiểm" — chỉ dùng khi chưa chốt được dim
    # thật (Handoff ghi 3072, SigLIP so400m thực tế 1152).
    dim: int = _int("VECTOR_DIM", 0)
    encoder_name: str = os.getenv("ENCODER_NAME", "")

    # ── Tham số search ───────────────────────────────────────────────
    # ef_search KHÔNG nhận per-request: chỉ áp được bằng cách gán
    # index.hnsw.efSearch, tức mutable state trên object dùng chung mọi thread.
    # Xem docs/search-design.md §7.
    ef_search: int = _int("FAISS_EF_SEARCH", 64)
    rerank_candidates: int = _int("RERANK_CANDIDATES", 800)

    # Ngưỡng DUY NHẤT chọn EXACT_SUBSET vs 2-tier. Đo được ở dim=1152: 8.5k candidate
    # = 7.4ms, 20k = 12.6ms, 50k = 63ms (bound bởi băng thông, không phải FLOPs).
    # Phải đo lại trên máy thật rồi chốt.
    exact_subset_max: int = _int("EXACT_SUBSET_MAX", 20_000)

    # ── Đồng thời (docs/search-design.md §7) ─────────────────────────
    # Encoder tốn 53.3 GFLOP/query. 8 worker x 4 BLAS thread trên 4 vCPU = 8x
    # oversubscription -> p99 sụp. Giới hạn số encode đồng thời.
    max_workers: int = _int("SC_MAX_WORKERS", 8)
    max_concurrent_encodes: int = _int("SC_MAX_CONCURRENT_ENCODES", 2)
    omp_threads: int = _int("OMP_NUM_THREADS", 4)

    # ── Vận hành ─────────────────────────────────────────────────────
    # Checksum là phòng vệ DUY NHẤT cho visual.f16 (file raw không header: size check
    # pass với bất kỳ file cùng độ dài). Đo: 1.9GB = 1-4s. Chỉ tắt khi debug.
    verify_checksums: bool = _bool("SC_VERIFY_CHECKSUMS", True)
    warmup_queries: int = _int("SC_WARMUP_QUERIES", 50)
    stub_mode: bool = _bool("SC_STUB_MODE", True)

    tags_sentinel: int = field(default=65535, init=False)  # uint16, KHÔNG phải 255

    def resolved_snapshot(self) -> tuple[str, str]:
        """(nguồn, giá trị) — bind mount thắng S3. Rỗng cả hai -> ('', '')."""
        if self.snapshot_dir:
            return ("dir", self.snapshot_dir)
        if self.snapshot_s3:
            return ("s3", self.snapshot_s3)
        return ("", "")

    def resolved_encoder(self) -> tuple[str, str]:
        if self.encoder_path:
            return ("dir", self.encoder_path)
        if self.encoder_s3:
            return ("s3", self.encoder_s3)
        return ("", "")


cfg = Config()
