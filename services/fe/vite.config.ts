import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Mặc định localhost cho chạy local/venv trực tiếp; đặt VITE_BE_TARGET=http://be:8000
// khi chạy dưới docker-compose (tên service, không resolve được ngoài network đó).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const beTarget = env.VITE_BE_TARGET || "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/api": { target: beTarget, changeOrigin: true },
        "/ws": { target: beTarget.replace("http", "ws"), ws: true },
      },
    },
  };
});