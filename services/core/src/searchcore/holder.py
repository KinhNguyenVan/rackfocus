"""IndexHolder: giữ snapshot hiện tại, atomic pointer swap khi load bản mới.

Xem docs/search-design.md §7. Hai hazard mà pointer swap thuần KHÔNG che được:

1. `Snapshot.refine` là np.memmap. Xoá file cũ thì inode còn sống tới khi mapping cuối
   đóng -> an toàn. Nhưng **thay file tại chỗ cùng đường dẫn** thì mapping bắt đầu trả
   byte của file mới ở offset cũ -> điểm số là rác, KHÔNG có exception. Nên snapshot mới
   phải nằm ở **thư mục mới**, không bao giờ ghi đè.
2. Hai lần load chồng nhau giữ 3x snapshot trong RAM -> OOM. Cần single-flight.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("searchcore.holder")


class IndexHolder:
    def __init__(self) -> None:
        self._snap = None
        self._swap_lock = threading.Lock()
        self._load_lock = threading.Lock()   # single-flight, KHÁC lock của swap
        self._loading = False

    @property
    def snap(self):
        """Snapshot đang serve. Request giữ ref này suốt vòng đời của nó, nên swap
        giữa đường không làm nó đọc lẫn dữ liệu."""
        return self._snap

    @property
    def ready(self) -> bool:
        return self._snap is not None

    def swap(self, new) -> None:
        with self._swap_lock:
            old, self._snap = self._snap, new
        if old is not None:
            log.info("swap snapshot v%s -> v%s", old.version, new.version)
        # Không del tường minh: request đang chạy vẫn giữ ref bản cũ, GC dọn khi xong.

    def load_and_swap(self, loader) -> object:
        """Chạy `loader()` rồi swap. Single-flight: lần gọi thứ hai khi đang load sẽ
        bị từ chối thay vì nhân đôi bộ nhớ."""
        with self._load_lock:
            if self._loading:
                raise RuntimeError("đang có một lần load snapshot khác chạy dở")
            self._loading = True
        try:
            new = loader()
            self.swap(new)
            return new
        finally:
            with self._load_lock:
                self._loading = False
