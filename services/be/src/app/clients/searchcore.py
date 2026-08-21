"""Wrap gRPC stub: Encode, Search, Health, GetTagVocab. Channel dùng chung, lazy init.

Bản trước viết cho proto CŨ (`SearchRequest(text=...)`, `Health` trên SearchCoreService)
nên không còn chạy được: `Search` giờ nhận `repeated QueryPart`, và `Health`/`GetTagVocab`
đã chuyển sang AdminService.
"""
from __future__ import annotations

import logging

import grpc

from ..pb.searchcore.v1 import admin_pb2 as apb
from ..pb.searchcore.v1 import admin_pb2_grpc as apb_grpc
from ..pb.searchcore.v1 import common_pb2 as cpb
from ..pb.searchcore.v1 import search_pb2 as pb
from ..pb.searchcore.v1 import search_pb2_grpc as pb_grpc

log = logging.getLogger("app.searchcore")

_channel: grpc.aio.Channel | None = None
_search: pb_grpc.SearchCoreServiceStub | None = None
_admin: apb_grpc.AdminServiceStub | None = None


async def connect(target: str) -> None:
    global _channel, _search, _admin
    if _channel is None:
        _channel = grpc.aio.insecure_channel(
            target,
            options=[("grpc.max_send_message_length", 32 * 1024 * 1024),
                     ("grpc.max_receive_message_length", 32 * 1024 * 1024)],
        )
        _search = pb_grpc.SearchCoreServiceStub(_channel)
        _admin = apb_grpc.AdminServiceStub(_channel)
        log.info("kết nối searchcore: %s", target)


async def close() -> None:
    global _channel, _search, _admin
    if _channel is not None:
        await _channel.close()
        _channel, _search, _admin = None, None, None


def _ctx(request_id: str = "") -> cpb.RequestContext:
    return cpb.RequestContext(request_id=request_id)


async def health(timeout: float = 2.0):
    return await _admin.Health(apb.HealthRequest(), timeout=timeout)


async def tag_vocab(timeout: float = 5.0) -> tuple[dict[int, str], str]:
    """({tag_id: description}, snapshot_ver). Rỗng nếu core chưa có tag."""
    resp = await _admin.GetTagVocab(apb.GetTagVocabRequest(ctx=_ctx()), timeout=timeout)
    return {t.id: t.description for t in resp.tags}, resp.snapshot_ver


async def encode(text: str, *, request_id: str = "", timeout: float = 3.0) -> list[float]:
    """Encode text -> vector. Gọi SONG SONG với LLM, đừng gọi tuần tự sau nó."""
    resp = await _search.Encode(
        pb.EncodeRequest(ctx=_ctx(request_id), text=text, normalize=True), timeout=timeout)
    return list(resp.vector.values)


async def search(vector, *, top_k: int, tags=None, request_id: str = "",
                 min_score: float | None = None, with_payload: bool = True,
                 max_per_video: int = 0, min_time_gap_sec: float = 0.0,
                 dedup_threshold: float = 0.0, timeout: float = 2.0):
    req = pb.SearchRequest(
        ctx=_ctx(request_id),
        query=[pb.QueryPart(vector=cpb.Vector(values=list(vector)))],
        top_k=top_k,
        filter=cpb.Filter(tags=list(tags or [])),
        params=pb.SearchParams(min_score=min_score or 0.0),
        diversity=cpb.Diversity(max_per_video=max_per_video,
                                min_time_gap_sec=min_time_gap_sec,
                                dedup_threshold=dedup_threshold),
        with_payload=with_payload,
    )
    return await _search.Search(req, timeout=timeout)
