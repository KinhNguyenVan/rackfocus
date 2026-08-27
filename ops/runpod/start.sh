#!/usr/bin/env bash
# Boot 3 tiến trình trong 1 container: searchcore -> BE -> Caddy.
#
# Thứ tự KHÔNG cần chờ nhau:
#   - BE tự lành: tagvocab thấy cache rỗng thì thử nạp lại ở MỖI request, nên chạy
#     trước core cũng được, không phải restart (xem services/be/.../tagvocab.py).
#   - Trong lúc core chưa lên, /api/search trả 503 chứ không trả kết quả sai.
# Nhờ vậy Caddy phục vụ được FE ngay giây thứ 5, không phải đợi core tải 3.7GB.
set -uo pipefail

log() { echo "[start] $(date -u +%H:%M:%S) $*"; }

# ── Pod hay Serverless? ──────────────────────────────────────────────
# Ba thứ khác nhau giữa hai chế độ, sai bất kỳ cái nào là worker không lên:
#   1. Cổng: Serverless load-balancing bắt phục vụ trên $PORT (mặc định 80 của họ).
#   2. Network volume: Serverless mount ở /runpod-volume, Pod ở /workspace.
#   3. Auth: Serverless đã bắt Bearer token ở trước -> basic_auth chỉ gây rắc rối.
if [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
	MODE=serverless
	# GÁN THẲNG, không dùng `: "${POD_PORT:=...}"`: hợp đồng của RunPod là "container
	# phải nghe trên $PORT", và $PORT là nguồn sự thật duy nhất. Dùng `:=` thì POD_PORT
	# nướng sẵn trong image (hoặc trong .env) sẽ thắng, Caddy nghe cổng khác cổng RunPod
	# gọi tới -> health check không bao giờ 200, endpoint không bao giờ lên. Đúng lỗi này
	# đã lộ ra khi chạy thử container với PORT=8081.
	POD_PORT="${PORT:-80}"
	: "${MODEL_CACHE_DIR:=/runpod-volume/cache/searchcore}"
	VOLUME_ROOT=/runpod-volume
else
	MODE=pod
	: "${POD_PORT:=${PORT:-8080}}"
	: "${MODEL_CACHE_DIR:=/workspace/cache/searchcore}"
	VOLUME_ROOT=/workspace
fi
export POD_PORT MODEL_CACHE_DIR
log "chế độ $MODE — cổng $POD_PORT, cache $MODEL_CACHE_DIR"

: "${SC_SOCKET_PATH:=/var/run/searchcore/sc.sock}"

mkdir -p "$(dirname "$SC_SOCKET_PATH")" "$MODEL_CACHE_DIR"

# ── Cảnh báo sớm thay vì chết giữa đường ─────────────────────────────
# Không kiểm được giá trị đúng/sai, chỉ kiểm CÓ hay KHÔNG. Thiếu key thì core tải
# snapshot thất bại sau vài phút chờ, rất tốn thời gian để nhận ra.
for v in AWS_ACCESS_KEY AWS_SECRET_KEY AWS_REGION AWS_BUCKET_NAME SNAPSHOT_S3 ENCODER_S3; do
	[ -z "${!v:-}" ] && log "THIẾU $v — core sẽ không tải được snapshot/encoder"
done
[ -z "${LLM_API_KEY:-}${CEREBRAS_API_KEY:-}" ] && \
	log "THIẾU LLM key — search vẫn chạy nhưng enrich lỗi, không lọc tag"

case "$MODEL_CACHE_DIR" in
	"$VOLUME_ROOT"/*)
		if [ ! -d "$VOLUME_ROOT" ]; then
			log "CẢNH BÁO: $VOLUME_ROOT không tồn tại — chưa gắn network volume?"
			log "          -> tải lại ~3.7GB mỗi lần worker/pod khởi động"
		fi ;;
	*) log "CẢNH BÁO: MODEL_CACHE_DIR=$MODEL_CACHE_DIR KHÔNG nằm trong $VOLUME_ROOT"
	   log "          -> mất khi container biến mất, lần sau tải lại ~3.7GB" ;;
esac

# ── Basic auth: chỉ bật khi có đủ user + hash ────────────────────────
# URL proxy của RunPod (pod) là public, ai biết pod id là gọi được /api/search.
# Trên serverless thì bỏ qua: RunPod đã bắt Bearer token, và basic_auth ở đây chỉ làm
# health check /ping bị 401 -> LB coi worker là hỏng (Caddyfile đã cho /ping ra ngoài
# block auth, nhưng vẫn không có lý do bật thêm một lớp nữa).
if [ "$MODE" = "serverless" ] && [ -n "${POD_USER:-}" ]; then
	log "bỏ qua basic_auth: chế độ serverless đã có Bearer token của RunPod"
	POD_USER=""
fi

if [ -n "${POD_USER:-}" ] && [ -n "${POD_PASS_HASH:-}" ]; then
	cat > /etc/caddy/auth.conf <<EOF
basic_auth {
	$POD_USER $POD_PASS_HASH
}
EOF
	log "basic_auth: BẬT (user=$POD_USER)"
else
	: > /etc/caddy/auth.conf
	if [ "$MODE" = "pod" ]; then
		log "basic_auth: TẮT — URL pod ai có cũng gọi được. Đặt POD_USER + POD_PASS_HASH để bật."
	else
		log "basic_auth: TẮT (RunPod Serverless lo phần xác thực)"
	fi
fi

# ── searchcore ───────────────────────────────────────────────────────
rm -f "$SC_SOCKET_PATH"
log "core: khởi động (lần đầu tải ~3.7GB từ S3, im lặng vài phút là bình thường)"
python -m searchcore.main &
CORE_PID=$!

# ── BE ───────────────────────────────────────────────────────────────
log "be: khởi động trên :8000"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log &
BE_PID=$!

# ── vmtouch: kéo visual.f16 vào page cache sau khi core nạp xong ─────
# vmtouch đã có trong services/core/Dockerfile từ lâu nhưng CHƯA HỀ được gọi ở đâu.
# Ở đây nó thực sự cần: cache nằm trên network volume của RunPod, mmap từ network
# storage chậm hơn NVMe nhiều lần. Đo trên máy local (page cache nguội vs nóng):
# exact_subset 100k điểm = 477ms -> 24ms.
(
	for _ in $(seq 1 120); do
		sleep 10
		curl -sf --max-time 5 "http://127.0.0.1:8000/readyz" 2>/dev/null \
			| grep -q '"ready":true' || continue
		f16=$(find "$MODEL_CACHE_DIR" -name "visual.f16" -type f 2>/dev/null | head -1)
		if [ -n "$f16" ]; then
			log "vmtouch: đang ghim $(du -h "$f16" | cut -f1) vào page cache"
			vmtouch -t "$f16" >/dev/null 2>&1 && log "vmtouch: xong"
		else
			log "vmtouch: không tìm thấy visual.f16 trong $MODEL_CACHE_DIR"
		fi
		exit 0
	done
	log "vmtouch: core không ready sau 20 phút, bỏ qua"
) &

# ── Caddy (foreground) ───────────────────────────────────────────────
# Chạy foreground để container sống/chết theo Caddy, và log Caddy ra stdout của pod.
log "caddy: phục vụ FE + proxy /api trên :$POD_PORT"
trap 'kill $CORE_PID $BE_PID 2>/dev/null' TERM INT
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
