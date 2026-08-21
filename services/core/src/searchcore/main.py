"""Entrypoint: config -> tải snapshot + encoder (S3 hoặc bind mount) -> warmup -> serve.

Thứ tự CÓ Ý NGHĨA: nạp encoder TRƯỚC snapshot. Encoder là thứ hay sai nhất (bundle vision
bị dùng lẫn cho text, thiếu tokenizer, sai output name) và rẻ nhất để phát hiện. Fail ở đó
trước khi bỏ vài phút tải + validate 2GB snapshot.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from concurrent import futures

import grpc

from . import snapshot as snap_mod
from . import warmup as warmup_mod
from .config import cfg
from .encoder import text as text_encoder
from .holder import IndexHolder
from .pb.searchcore.v1 import admin_pb2_grpc as apb_grpc
from .pb.searchcore.v1 import search_pb2_grpc as pb_grpc
from .server import AdminServiceServicer, SearchCoreServiceServicer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("searchcore")


def _tune_threads() -> None:
    """Giới hạn thread TRƯỚC khi import numpy/faiss — sau đó set là vô tác dụng.

    OMP_NUM_THREADS KHÔNG điều khiển OpenBLAS (bánh xe pthreads bỏ qua nó), mà matmul
    rerank là numpy->OpenBLAS. Phải set cả hai, nếu không 8 gRPC worker x 4 thread trên
    4 vCPU = 8x oversubscription và p99 sụp. Xem docs/search-design.md §7.
    """
    n = str(cfg.omp_threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, n)


def load_encoder():
    kind, source = cfg.resolved_encoder()
    if not source:
        log.warning("chưa cấu hình ENCODER_S3/ENCODER_PATH — core không encode được text, "
                    "BE phải tự gửi `vector` trong QueryPart")
        return None
    t0 = time.perf_counter()
    enc = text_encoder.load(source, name=cfg.encoder_name,
                            max_concurrent=cfg.max_concurrent_encodes)
    log.info("encoder (%s) %s: dim=%d, nạp %.0fs", kind, source, enc.dim,
             time.perf_counter() - t0)
    return enc


def load_snapshot(path: str | None = None):
    if path is None:
        _kind, source = cfg.resolved_snapshot()
        if not source:
            raise RuntimeError("chưa cấu hình SNAPSHOT_DIR hoặc SNAPSHOT_S3")
        if cfg.snapshot_dir and cfg.snapshot_s3:
            log.warning("có CẢ SNAPSHOT_DIR và SNAPSHOT_S3 — dùng bind mount (%s), "
                        "bỏ qua S3. Hai nguồn cùng lúc là cách nhanh nhất để serve sai bản",
                        cfg.snapshot_dir)
        path = source

    if path.startswith("s3://"):
        from .encoder.base import resolve_bundle

        path = resolve_bundle(path)

    t0 = time.perf_counter()
    snap = snap_mod.load(path, cfg)
    log.info("snapshot %s nạp %.1fs", path, time.perf_counter() - t0)
    return snap


def serve() -> None:
    _tune_threads()

    holder = IndexHolder()
    encoder = None

    if cfg.stub_mode:
        log.warning("SC_STUB_MODE=1 — không nạp snapshot/encoder. Search sẽ trả UNAVAILABLE. "
                    "Đặt SC_STUB_MODE=0 để chạy thật.")
    else:
        encoder = load_encoder()
        snap = load_snapshot()

        if encoder is not None and encoder.dim != snap.dim:
            # Bắt sớm: encoder và snapshot khác không gian embedding thì mọi điểm là rác
            # nhưng không có exception nào.
            raise RuntimeError(
                f"encoder dim {encoder.dim} != snapshot dim {snap.dim} — bundle encoder "
                "không cùng model với lúc embed corpus")

        holder.swap(snap)
        warmup_mod.warmup(snap, encoder, queries=cfg.warmup_queries,
                          exact_max=cfg.exact_subset_max,
                          rerank_candidates=cfg.rerank_candidates)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.max_workers),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    pb_grpc.add_SearchCoreServiceServicer_to_server(
        SearchCoreServiceServicer(holder, encoder, cfg), server)
    apb_grpc.add_AdminServiceServicer_to_server(
        AdminServiceServicer(holder, encoder, cfg, loader=load_snapshot), server)

    os.makedirs(os.path.dirname(cfg.socket_path), exist_ok=True)
    if os.path.exists(cfg.socket_path):
        os.remove(cfg.socket_path)
    server.add_insecure_port(f"unix://{cfg.socket_path}")
    server.add_insecure_port(f"[::]:{cfg.tcp_port}")

    server.start()
    os.chmod(cfg.socket_path, 0o666)
    log.info("sẵn sàng — unix:%s và tcp:%d", cfg.socket_path, cfg.tcp_port)

    def _stop(signum, _frame):
        log.info("nhận signal %s, dừng...", signum)
        server.stop(grace=5).wait()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
