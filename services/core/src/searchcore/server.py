"""Implement SearchCoreServiceServicer: map request proto -> hàm search, đo Timings."""
import time

from .config import cfg
from .pb.searchcore.v1 import search_pb2 as pb
from .pb.searchcore.v1 import search_pb2_grpc as pb_grpc


class SearchCoreServiceServicer(pb_grpc.SearchCoreServiceServicer):
    def __init__(self, holder=None):
        self.holder = holder  # IndexHolder — None khi stub_mode

    def Health(self, request, context):
        return pb.HealthResponse(
            ready=True,
            snapshot_ver="stub" if cfg.stub_mode else self.holder.snap.version,
            vector_count=0 if cfg.stub_mode else self.holder.snap.count,
            dim=3072,
        )

    def Search(self, request, context):
        t0 = time.perf_counter()

        if cfg.stub_mode:
            # Kết quả giả — chỉ để xác nhận đường ống BE <-> core thông.
            hits = [
                pb.SearchHit(
                    scene_id=1000 + i,
                    faiss_row=i,
                    score_coarse=0.9 - i * 0.01,
                    score_exact=0.91 - i * 0.01,
                    rank=i,
                )
                for i in range(min(request.top_k or 10, 10))
            ]
        else:
            raise NotImplementedError("TODO: gọi search.two_tier(...)")

        total = (time.perf_counter() - t0) * 1000
        return pb.SearchResponse(
            hits=hits,
            timings=pb.Timings(total_ms=total),
            snapshot_ver="stub" if cfg.stub_mode else self.holder.snap.version,
        )

    # TODO: SearchStream, SearchTemporal, Encode, LoadSnapshot
