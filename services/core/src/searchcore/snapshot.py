"""Snapshot bất biến: đọc manifest, validate, load faiss + mmap refine + idmap + tags.

Xem docs/search-design.md §2. Mọi kiểm tra ở đây là FAIL-CLOSED: sai thì từ chối load,
không degrade. Lý do: các chế độ hỏng của snapshot đều **âm thầm** — trộn `visual.faiss`
của build #2 với `visual.f16` của build #1 cho mọi điểm là dot product với vector của
frame khác, mà mọi check kích thước/độ dài vẫn pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

import numpy as np

log = logging.getLogger("searchcore.snapshot")

VISUAL_FAISS = "visual.faiss"
VISUAL_REFINE = "visual.f16"
IDMAP = "idmap.npy"
PAYLOAD = "payload.parquet"
TOMBSTONE = "tombstone.bin"
MANIFEST = "manifest.json"
TAGS = "tags.npy"
TAG_VOCAB = "tag_vocab.json"

TAGS_SENTINEL = 65535  # uint16. KHÔNG phải 255 — với vocab 100-500 thì 255 là tag hợp lệ.

# Chỉ kiểm norm trên mẫu này khi file quá lớn; mặc định kiểm hết (xem _check_normalized).
NORM_TOL = 1e-3


class SnapshotError(RuntimeError):
    """Snapshot không dùng được. Thông báo phải nói rõ file nào, lệch bao nhiêu."""


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _read_vocab(path: str) -> dict[int, dict]:
    """Đọc tag_vocab.json, chấp nhận hai dạng và chuẩn hoá về {id: {name, description}}.

    Dạng chính (đơn giản, dùng để nhồi vào prompt LLM):
        {"0": "cảnh bóng đá trên sân", "1": "phỏng vấn trong studio"}
    Dạng đầy đủ, nếu sau này cần tách name khỏi description:
        {"0": {"name": "sport", "description": "cảnh bóng đá trên sân"}}
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            raise SnapshotError(
                f"{TAG_VOCAB}: key {k!r} không phải số. Định dạng là "
                '{"tag_id": "description"}') from None
        if isinstance(v, str):
            out[tid] = {"name": "", "description": v}
        elif isinstance(v, dict):
            out[tid] = {"name": v.get("name", ""), "description": v.get("description", "")}
        else:
            raise SnapshotError(
                f"{TAG_VOCAB}: giá trị của tag {tid} phải là string hoặc object, "
                f"nhận {type(v).__name__}")
    return out


class Snapshot:
    """Bất biến. Đổi bản = tạo object mới + swap pointer, không mutate."""

    def __init__(self, path: str, *, dim: int = 0, encoder_name: str = "",
                 verify_checksums: bool = True, ef_search: int = 64):
        self.path = path
        man_path = os.path.join(path, MANIFEST)
        if not os.path.exists(man_path):
            raise SnapshotError(f"Thiếu {MANIFEST} trong {path}")
        with open(man_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.version = str(self.manifest.get("version", ""))
        self.count = int(self.manifest["count"])
        self.dim = int(self.manifest["dim"])
        self.metric = self.manifest.get("metric", "")
        self.encoder_name = self.manifest.get("encoder_name", "")
        self.groups = self.manifest.get("groups")

        self._check_contract(dim, encoder_name)
        if verify_checksums:
            self._check_checksums()
        else:
            log.warning("BỎ QUA checksum — visual.f16 không có header nên đây là phòng "
                        "vệ duy nhất chống trộn file giữa các lần build")

        self._load_files()
        self._check_alignment()
        self._check_normalized()
        self._build_tag_index()

        self.index.hnsw.efSearch = ef_search   # đặt 1 lần, KHÔNG per-request (§7)
        log.info("snapshot v%s: %d point, dim=%d, %d tag, %d chưa gán",
                 self.version, self.count, self.dim, len(self.tag_vocab),
                 int(self.unassigned_count))

    # ── kiểm tra ─────────────────────────────────────────────────────
    def _check_contract(self, dim: int, encoder_name: str) -> None:
        """Manifest là hợp đồng offline↔online: lệch dim/encoder là trộn snapshot của
        model khác, và sai kiểu đó không bao giờ tự lộ ra ở kết quả."""
        if dim and self.dim != dim:
            raise SnapshotError(
                f"dim lệch: manifest={self.dim}, config VECTOR_DIM={dim}. "
                "Handoff ghi 3072 nhưng SigLIP so400m thực tế 1152 — chốt lại rồi sửa env.")
        if encoder_name and self.encoder_name != encoder_name:
            raise SnapshotError(
                f"encoder lệch: manifest={self.encoder_name!r}, config={encoder_name!r}")
        if self.metric and self.metric != "cosine":
            raise SnapshotError(f"metric không hỗ trợ: {self.metric!r} (chỉ 'cosine')")

    def _check_checksums(self) -> None:
        sums = self.manifest.get("checksums") or {}
        if not sums:
            raise SnapshotError(f"{MANIFEST} không có 'checksums'")
        # File tag do một pass khác sinh ra SAU khi manifest được seal -> có thể chưa
        # nằm trong checksums. Cảnh báo, không chặn; nhưng nếu CÓ thì phải khớp.
        for name in (TAGS, TAG_VOCAB):
            if os.path.exists(os.path.join(self.path, name)) and name not in sums:
                log.warning("%s không nằm trong manifest.checksums — không có gì bảo đảm "
                            "nó được sinh cho đúng bản snapshot này", name)
        for name, want in sums.items():
            fp = os.path.join(self.path, name)
            if not os.path.exists(fp):
                raise SnapshotError(f"manifest khai {name} nhưng file không tồn tại")
            got = _sha256(fp)
            if got != want:
                raise SnapshotError(
                    f"checksum {name} lệch: manifest={want[:12]}… thực tế={got[:12]}… "
                    "(trộn file giữa hai lần build?)")

    def _load_files(self) -> None:
        import faiss
        import pyarrow.parquet as pq

        p = self.path
        refine_path = os.path.join(p, VISUAL_REFINE)
        want_bytes = self.count * self.dim * 2
        actual = os.path.getsize(refine_path)
        if actual != want_bytes:
            # File raw không header -> đọc lệch là ra rác, không phải lỗi.
            raise SnapshotError(
                f"{VISUAL_REFINE} sai kích thước: {actual} byte, cần "
                f"count*dim*2 = {self.count}*{self.dim}*2 = {want_bytes}")

        self.index = faiss.read_index(os.path.join(p, VISUAL_FAISS))
        self.refine = np.memmap(refine_path, dtype=np.float16,
                                mode="r", shape=(self.count, self.dim))
        self.idmap = np.load(os.path.join(p, IDMAP))
        self.payload = pq.read_table(os.path.join(p, PAYLOAD))
        self.tombstone = np.fromfile(os.path.join(p, TOMBSTONE), dtype=np.uint8)

        tags_path = os.path.join(p, TAGS)
        if os.path.exists(tags_path):
            self.tags = np.load(tags_path)
            if self.tags.dtype != np.uint16:
                raise SnapshotError(f"{TAGS} phải là uint16, nhận {self.tags.dtype}")
        else:
            # Chưa có dữ liệu tag -> mọi row coi như chưa gán. Đường không-tag vẫn chạy.
            log.warning("không có %s — chỉ chạy được đường search toàn bộ", TAGS)
            self.tags = np.full(self.count, TAGS_SENTINEL, dtype=np.uint16)

        self.tag_vocab = _read_vocab(os.path.join(p, TAG_VOCAB))

    def _check_alignment(self) -> None:
        """5 file phải row-aligned. Lệch là mọi kết quả sai nhưng không có exception."""
        for name, got in ((IDMAP, len(self.idmap)),
                          (PAYLOAD, self.payload.num_rows),
                          (TAGS, len(self.tags)),
                          ("faiss.ntotal", self.index.ntotal)):
            if got != self.count:
                raise SnapshotError(
                    f"{name} có {got} row nhưng manifest.count={self.count}")

        want_tomb = (self.count + 7) // 8
        if len(self.tombstone) != want_tomb:
            raise SnapshotError(
                f"{TOMBSTONE} {len(self.tombstone)} byte, cần ceil({self.count}/8)="
                f"{want_tomb} — bitset bị cắt sẽ gây IndexError chỉ ở một số query")

        if self.index.d != self.dim:
            raise SnapshotError(f"faiss.d={self.index.d} != manifest.dim={self.dim}")

        assigned = self.tags[self.tags != TAGS_SENTINEL]
        if self.tag_vocab and len(assigned):
            unknown = np.setdiff1d(np.unique(assigned),
                                   np.fromiter(self.tag_vocab, dtype=np.int64))
            if unknown.size:
                raise SnapshotError(
                    f"{TAGS} có tag không nằm trong {TAG_VOCAB}: {unknown[:10].tolist()}")

    def _check_normalized(self) -> None:
        """manifest ghi metric='cosine' nhưng index dùng METRIC_INNER_PRODUCT — hai cái
        CHỈ tương đương khi vector đã L2-normalize.

        Kiểm toàn bộ, không lấy mẫu: nếu một group được embed bằng bundle ONNX chưa bake
        L2 thì norm ~15 và những row đó thống trị MỌI query bất kể nội dung. Vài nghìn
        row hỏng trên 500k thì lấy mẫu sẽ bỏ sót.
        """
        worst = 0.0
        bad = 0
        step = 1 << 16
        for i in range(0, self.count, step):
            block = np.asarray(self.refine[i : i + step], dtype=np.float32)
            dev = np.abs(np.linalg.norm(block, axis=1) - 1.0)
            bad += int((dev > NORM_TOL).sum())
            worst = max(worst, float(dev.max(initial=0.0)))
        if bad:
            raise SnapshotError(
                f"{bad}/{self.count} vector không L2-normalized (lệch tối đa {worst:.4f}). "
                "Group nào embed bằng bundle ONNX chưa bake L2 phải embed lại.")

    # ── tag index ────────────────────────────────────────────────────
    def _build_tag_index(self) -> None:
        """CSR tag -> row. Dựng lúc load (~30-50ms ở 500k) thay vì ship thành file thứ ba
        phải pin integrity riêng.

        Mỗi frame đúng 1 tag nên đây là PHÂN HOẠCH: bucket không giao nhau, candidate của
        một tập tag = nối các bucket, không cần dedup.
        """
        order = np.argsort(self.tags, kind="stable").astype(np.int64)
        sorted_tags = self.tags[order]
        # Bucket chỉ cho tag thật; sentinel dồn về cuối và bị loại.
        n_assigned = int(np.searchsorted(sorted_tags, TAGS_SENTINEL, side="left"))
        self.unassigned_count = self.count - n_assigned
        self._csr_rows = order[:n_assigned]
        n_tags = (int(sorted_tags[:n_assigned].max()) + 1) if n_assigned else 0
        counts = np.bincount(sorted_tags[:n_assigned].astype(np.int64), minlength=n_tags)
        self._csr_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.tag_counts = counts

    def rows_for_tags(self, tags) -> np.ndarray:
        """Row của các tag đã cho. Tag ngoài vocab -> ValueError (không phải IndexError)."""
        n_tags = len(self._csr_indptr) - 1
        out = []
        for t in tags:
            t = int(t)
            if t < 0 or t >= n_tags:
                raise ValueError(
                    f"tag {t} ngoài phạm vi [0,{n_tags}) — LLM trả tag không tồn tại?")
            out.append(self._csr_rows[self._csr_indptr[t] : self._csr_indptr[t + 1]])
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(out) if len(out) > 1 else out[0]

    # ── tombstone ────────────────────────────────────────────────────
    def alive_mask(self, rows: np.ndarray) -> np.ndarray:
        """True = còn sống. Bit order LSB-first (bit i của byte i//8 là row i)."""
        if not self.tombstone.size:
            return np.ones(len(rows), dtype=bool)
        byte = self.tombstone[rows >> 3]
        return ((byte >> (rows & 7).astype(np.uint8)) & 1) == 0


def load(path: str, cfg) -> Snapshot:
    return Snapshot(path, dim=cfg.dim, encoder_name=cfg.encoder_name,
                    verify_checksums=cfg.verify_checksums, ef_search=cfg.ef_search)
