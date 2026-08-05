"""Tạo FastAPI app, mount router, lifespan: mở gRPC channel + pool Postgres/Redis."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .clients import searchcore


@asynccontextmanager
async def lifespan(app: FastAPI):
    await searchcore.get_stub()
    yield
    await searchcore.close()


app = FastAPI(title="rackfocus BE", lifespan=lifespan)


class SearchReq(BaseModel):
    text: str
    top_k: int = 10


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    h = await searchcore.health()
    return {"ready": h.ready, "snapshot": h.snapshot_ver, "dim": h.dim}


@app.post("/api/search")
async def search(req: SearchReq):
    r = await searchcore.search(req.text, req.top_k)
    return {
        "hits": [
            {"scene_id": h.scene_id, "score": h.score_exact, "rank": h.rank}
            for h in r.hits
        ],
        "timings": {"total_ms": r.timings.total_ms},
        "snapshot": r.snapshot_ver,
    }
