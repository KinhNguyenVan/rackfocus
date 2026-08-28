import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Mặc định localhost cho chạy local/venv trực tiếp; đặt VITE_BE_TARGET=http://be:8000
// khi chạy dưới docker-compose (tên service, không resolve được ngoài network đó).
const BE_TARGET = process.env.VITE_BE_TARGET || "http://localhost:8000";

// RunPod Serverless: mọi request tới endpoint PHẢI có `Authorization: Bearer <api key>`.
// Trình duyệt KHÔNG gắn được header vào một lần điều hướng thường (gõ URL vào thanh địa
// chỉ), nên không thể mở FE trực tiếp từ endpoint. Cách gọn nhất khi thi: chạy FE ở máy
// mình, để vite proxy chèn header — key nằm trên máy, không bao giờ vào bundle JS.
//
//   RUNPOD_ENDPOINT_ID=xxxx RUNPOD_API_KEY=yyyy npm run dev
//
// KHÔNG dùng tiền tố VITE_ cho key: biến VITE_* bị nhúng thẳng vào bundle và ai mở
// devtools cũng đọc được.
const RUNPOD_ID = process.env.RUNPOD_ENDPOINT_ID || "";
const RUNPOD_KEY = process.env.RUNPOD_API_KEY || "";
const useRunpod = Boolean(RUNPOD_ID && RUNPOD_KEY);

const target = useRunpod ? `https://${RUNPOD_ID}.api.runpod.ai` : BE_TARGET;
const headers = useRunpod ? { Authorization: `Bearer ${RUNPOD_KEY}` } : undefined;

// Transcript keyword search là artifact phụ, không nằm trong image RunPod đang deploy. Khi
// search vector vẫn muốn đi RunPod (target ở trên) mà transcript chạy trên BE local (nhánh
// feat/transcript-search-v2 + transcript.sqlite), set VITE_TRANSCRIPT_TARGET=http://localhost:8000:
// chỉ /api/transcript tách về BE local, còn lại giữ nguyên. Không set = mọi /api đi 1 target
// như cũ (không đổi hành vi mặc định / lúc deploy).
const TRANSCRIPT_TARGET = process.env.VITE_TRANSCRIPT_TARGET || "";
if (TRANSCRIPT_TARGET) {
  console.log(`[vite] proxy /api/transcript -> ${TRANSCRIPT_TARGET} (BE local, không kèm token)`);
}

if (RUNPOD_ID && !RUNPOD_KEY) {
  console.warn("[vite] có RUNPOD_ENDPOINT_ID nhưng thiếu RUNPOD_API_KEY -> vẫn dùng " + BE_TARGET);
}
console.log(`[vite] proxy /api -> ${target}${useRunpod ? " (kèm Bearer token)" : ""}`);

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      // /api/transcript phải đứng TRƯỚC /api: vite chọn rule đầu tiên mà path khớp tiền tố.
      // Chỉ thêm khi opt-in (VITE_TRANSCRIPT_TARGET) để mặc định không đổi.
      ...(TRANSCRIPT_TARGET
        ? { "/api/transcript": { target: TRANSCRIPT_TARGET, changeOrigin: true } }
        : {}),
      // changeOrigin bắt buộc khi target là RunPod: họ định tuyến theo Host header, gửi
      // "localhost:5173" thì không tới được endpoint.
      "/api": { target, changeOrigin: true, headers },
      "/ws": { target: target.replace(/^http/, "ws"), ws: true, headers },
    },
  },
});
