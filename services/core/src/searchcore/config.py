"""Đọc env. Xem docs/search-design.md §2 (kiểm tra snapshot) và §7 (đồng thời)."""
import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


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
    # 64 là con số cho search KHÔNG lọc. Với filtered search (HNSW + IDSelector) nó làm
    # LỌC TAG TẮT HẲN mà không báo lỗi: HNSW chỉ thăm ~ef_search node rồi mới lọc, nên
    # với tỉ lệ nhận ~6-10% thì trả về ~4-6 kết quả. search_with_fallback thấy
    # rows < top_k liền rơi về full corpus và xoá tags_used -> mọi tag đi đường `pre`
    # (bucket > EXACT_SUBSET_MAX) đều vô hiệu, chỉ để lại warning "tag_fallback".
    # Đo với recall@300 so GROUND TRUTH THẬT (chế độ exact brute-force), corpus 613k/
    # 13 tag, top_k=300, rerank_candidates=1000, EXACT_SUBSET_MAX=100k:
    #            untagged            tag 361k (59% corpus, tag duy nhất còn qua HNSW)
    #   ef=2000  99,3% /  20ms       71,4% / 177ms
    #   ef=4000  99,7% /  29ms       92,4% / 238ms
    #   ef=10000 99,8% /  95ms         --
    #   ef=20000 99,9% / 276ms         --
    # Chọn 4000: recall untagged gần như PHẲNG theo ef (99,3 -> 99,9% khi ef tăng 10x)
    # nên ef cao là mua rất ít recall bằng rất nhiều latency. Sau khi nâng
    # EXACT_SUBSET_MAX lên 100k thì 12/13 tag đi đường exact (recall 100%), HNSW chỉ
    # còn phục vụ untagged + tag 361k — mà cả hai đều có tỉ lệ nhận cao nên ef thấp đủ.
    #
    # rerank_candidates KHÔNG phải nút thắt: rc=1000 và rc=4000 cho recall y hệt
    # (99,7%/79,0%), vì số candidate tới được rerank bị chặn bởi ef x tỉ lệ nhận, chứ
    # không phải bởi rc.
    ef_search: int = _int("FAISS_EF_SEARCH", 4_000)
    rerank_candidates: int = _int("RERANK_CANDIDATES", 800)

    # Ngưỡng DUY NHẤT chọn EXACT_SUBSET vs 2-tier. Nâng 20k -> 100k sau khi đo trên
    # corpus thật 613k/13 tag, top_k=300, visual.f16 đã pin trong page cache:
    #    9.098 điểm ->   9ms |  37.674 ->  47ms |  59.750 ->  36ms |  61.146 ->  39ms
    #  361.319 điểm -> 630ms | 612.975 -> 12.163ms
    # Tuyến tính tới ~60-100k rồi vỡ hẳn (361k = 16x thời gian của 60k dù chỉ 6x điểm):
    # working set vượt RAM còn trống nên thrash. Chọn 100k vì 12/13 tag hiện tại đều
    # <= 61.146 điểm, tức đi đường exact và đạt recall 100% ở 25-47ms — vừa CHÍNH XÁC
    # HƠN vừa NHANH HƠN nhánh HNSW+IDSelector (88,9% ở 95ms với ef=10000).
    #
    # BẮT BUỘC pin visual.f16 vào RAM (vmtouch, đã có trong Dockerfile): cùng query
    # tag 59.750 điểm đo được 477ms lúc page fault so với 24ms khi warm — chênh 20x.
    # Ngưỡng này phụ thuộc RAM còn trống của máy, đổi máy phải đo lại điểm vỡ.
    exact_subset_max: int = _int("EXACT_SUBSET_MAX", 100_000)

    # ── TRAKE (docs/superpowers/specs/2026-08-24-temporal-search-design.md) ──────
    # Top-K candidate rows per event BEFORE joining by video — bounds join cost,
    # not the whole corpus (a full brute-force rerank per event was already measured
    # as too slow for a single query; doing it twice would be worse).
    trake_candidates_per_event: int = _int("TRAKE_CANDIDATES_PER_EVENT", 500)
    trake_max_pairs_per_video: int = _int("TRAKE_MAX_PAIRS_PER_VIDEO", 1)
    # Hard floor on (t2 - t1): pairs closer together (including negative = order
    # violated) are excluded entirely, never scored. Also doubles as the decay's
    # zero-penalty reference point.
    trake_min_gap_sec: float = _float("TRAKE_MIN_GAP_SEC", 5.0)
    # Hard ceiling on (t2 - t1): pairs further apart are excluded entirely.
    trake_max_gap_sec: float = _float("TRAKE_MAX_GAP_SEC", 120.0)
    trake_lambda: float = _float("TRAKE_LAMBDA", 0.00557)
    trake_sim_weight: float = _float("TRAKE_SIM_WEIGHT", 0.8)
    trake_time_weight: float = _float("TRAKE_TIME_WEIGHT", 0.2)
    trake_top_k_chains: int = _int("TRAKE_TOP_K_CHAINS", 20)

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
