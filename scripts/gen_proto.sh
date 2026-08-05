#!/usr/bin/env bash
# Sinh stub gRPC cho cả core và be từ proto/.
# Chạy lại mỗi khi sửa proto/searchcore/v1/search.proto
set -euo pipefail
cd "$(dirname "$0")/.."

command -v python3 >/dev/null || { echo "Cần python3"; exit 1; }
python3 -c "import grpc_tools" 2>/dev/null || {
  echo "Thiếu grpcio-tools. Chạy: pip install grpcio-tools"; exit 1; }

for OUT in services/core/src/searchcore/pb services/be/src/app/pb; do
  mkdir -p "$OUT"
  python3 -m grpc_tools.protoc \
    -I proto \
    --python_out="$OUT" \
    --grpc_python_out="$OUT" \
    --pyi_out="$OUT" \
    proto/searchcore/v1/search.proto

  # protoc sinh import tuyệt đối (from searchcore.v1 import ...) -> sửa thành tương đối,
  # nếu không sẽ ModuleNotFoundError khi import từ package khác.
  find "$OUT" -name '*_pb2*.py' -print0 | while IFS= read -r -d '' f; do
    sed -i.bak -E 's/^from searchcore\.v1 import/from . import/' "$f" && rm -f "$f.bak"
  done

  touch "$OUT/__init__.py" "$OUT/searchcore/__init__.py" "$OUT/searchcore/v1/__init__.py" 2>/dev/null || true
done

echo "Đã sinh stub cho core và be."
