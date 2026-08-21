import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Mặc định localhost cho chạy local/venv trực tiếp; đặt VITE_BE_TARGET=http://be:8000
// khi chạy dưới docker-compose (tên service, không resolve được ngoài network đó).
const BE_TARGET = process.env.VITE_BE_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": { target: BE_TARGET, changeOrigin: true },
      "/ws":  { target: BE_TARGET.replace("http", "ws"), ws: true },
    },
  },
});