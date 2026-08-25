#!/usr/bin/env bash
# Sinh stub gRPC cho cả core và be từ proto/.
# Chạy lại mỗi khi sửa bất kỳ file .proto nào.
set -euo pipefail
cd "$(dirname "$0")/.."

# python3 trên Linux/CI; Windows (native, không docker) chỉ có `python`.
PYTHON="$(command -v python3 || command -v python || true)"
[ -n "$PYTHON" ] || { echo "Cần python3 hoặc python"; exit 1; }
"$PYTHON" -c "import grpc_tools" 2>/dev/null || {
  echo "Thiếu grpcio-tools. Chạy: pip install grpcio-tools"; exit 1; }

for OUT in services/core/src/searchcore/pb services/be/src/app/pb; do
  rm -rf "$OUT"; mkdir -p "$OUT"

  "$PYTHON" -m grpc_tools.protoc \
    -I proto \
    --python_out="$OUT" \
    --grpc_python_out="$OUT" \
    --pyi_out="$OUT" \
    proto/searchcore/v1/*.proto

  # protoc sinh import tuyệt đối (from searchcore.v1 import ...) -> sửa thành
  # tương đối, nếu không sẽ ModuleNotFoundError khi import từ package khác.
  find "$OUT" -name '*_pb2*.py' -print0 | while IFS= read -r -d '' f; do
    sed -i.bak -E 's/^from searchcore\.v1 import/from . import/' "$f" && rm -f "$f.bak"
  done

  # protoc không tạo __init__.py, phải tự thêm để import được
  find "$OUT" -type d -exec touch {}/__init__.py \;
done

echo "Đã sinh stub cho core và be."
