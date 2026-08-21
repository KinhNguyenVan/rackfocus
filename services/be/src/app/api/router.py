"""Gom sub-router. /healthz + /readyz ở root (Caddyfile route riêng), còn lại /api."""
from fastapi import APIRouter

from . import browse, health, search

root_router = APIRouter()
root_router.include_router(health.router, tags=["health"])

api_router = APIRouter(prefix="/api")
api_router.include_router(search.router, tags=["search"])
api_router.include_router(browse.router, tags=["browse"])
