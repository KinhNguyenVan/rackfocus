"""Entrypoint: đọc config, load snapshot, warmup, chạy gRPC server."""
import logging
import os
from concurrent import futures

import grpc

from .config import cfg
from .pb.searchcore.v1 import search_pb2_grpc as pb_grpc
from .server import SearchCoreServiceServicer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("searchcore")


def serve() -> None:
    holder = None
    if cfg.stub_mode:
        log.warning("STUB MODE — trả kết quả giả, chưa load index thật")
    else:
        raise NotImplementedError("TODO: load snapshot + warmup")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    pb_grpc.add_SearchCoreServiceServicer_to_server(SearchCoreServiceServicer(holder), server)

    os.makedirs(os.path.dirname(cfg.socket_path), exist_ok=True)
    if os.path.exists(cfg.socket_path):
        os.remove(cfg.socket_path)
    server.add_insecure_port(f"unix://{cfg.socket_path}")
    server.add_insecure_port(f"[::]:{cfg.tcp_port}")

    server.start()
    os.chmod(cfg.socket_path, 0o666)
    log.info("sẵn sàng — unix:%s và tcp:%d", cfg.socket_path, cfg.tcp_port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
