"""Wrap gRPC stub: retry, deadline, đổi Unix socket <-> TCP theo env."""
import os

import grpc

from ..pb.searchcore.v1 import search_pb2 as pb
from ..pb.searchcore.v1 import search_pb2_grpc as pb_grpc

TARGET = os.getenv("SEARCHCORE_TARGET", "unix:///var/run/searchcore/sc.sock")

_channel: grpc.aio.Channel | None = None
_stub: pb_grpc.SearchCoreStub | None = None


async def get_stub() -> pb_grpc.SearchCoreStub:
    global _channel, _stub
    if _stub is None:
        _channel = grpc.aio.insecure_channel(TARGET)
        _stub = pb_grpc.SearchCoreStub(_channel)
    return _stub


async def close() -> None:
    global _channel, _stub
    if _channel is not None:
        await _channel.close()
        _channel, _stub = None, None


async def search(text: str, top_k: int = 10, timeout: float = 2.0):
    stub = await get_stub()
    return await stub.Search(pb.SearchRequest(text=text, top_k=top_k), timeout=timeout)


async def health(timeout: float = 2.0):
    stub = await get_stub()
    return await stub.Health(pb.Empty(), timeout=timeout)
