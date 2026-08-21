"""Cache tag vocab lấy từ core, key theo snapshot_ver.

Vocab nằm trong snapshot của core nên tag id CHỈ có nghĩa trong đúng bản đó. Cache mà
không pin theo version thì sau khi core swap snapshot, BE sẽ đưa mô tả của bản cũ cho LLM
và gửi lại tag id đã đổi nghĩa — sai âm thầm, không có lỗi nào báo.

Vocab ~500 tag là 15-25k prompt token mỗi query, nên fetch lại mỗi request là vừa chậm vừa
tốn; cache là bắt buộc, nhưng phải kiểm version.
"""
from __future__ import annotations

import asyncio
import logging

from ..clients import searchcore

log = logging.getLogger("app.tagvocab")

_lock = asyncio.Lock()
_vocab: dict[int, str] = {}
_version: str = ""
_checked_ver: str = ""


async def get(settings, *, force: bool = False) -> tuple[dict[int, str], str]:
    """({tag_id: description}, snapshot_ver). Core chưa có tag -> ({}, ver)."""
    global _vocab, _version, _checked_ver

    if not force and _version:
        # Health rẻ (không truyền vocab) -> dùng nó để phát hiện core đã swap snapshot.
        try:
            h = await searchcore.health(timeout=1.0)
            if h.snapshot_ver == _checked_ver:
                return _vocab, _version
            log.info("core đã swap snapshot %s -> %s, nạp lại vocab",
                     _checked_ver, h.snapshot_ver)
        except Exception as ex:  # noqa: BLE001 — health lỗi thì dùng cache cũ
            log.debug("health lỗi (%s), dùng vocab đã cache", type(ex).__name__)
            return _vocab, _version

    async with _lock:
        try:
            vocab, ver = await searchcore.tag_vocab(timeout=5.0)
        except Exception as ex:  # noqa: BLE001
            log.warning("không lấy được tag vocab (%s: %s) -> search sẽ không lọc tag",
                        type(ex).__name__, ex)
            return _vocab, _version
        _vocab, _version = vocab, ver
        _checked_ver = ver
        log.info("tag vocab: %d tag (snapshot %s)", len(vocab), ver or "?")
        return _vocab, _version


def cached() -> tuple[dict[int, str], str]:
    return _vocab, _version
