import { defineConfig, type ProxyOptions, type HttpProxy } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import type http from "node:http";

// Dev-server proxy target — the vo-rag FastAPI surface. Override with
// VORAG_API_DEV_PROXY_TARGET; VORAG_PORTAL_DEV_API_KEY injects the API key
// server-side so EventSource (which cannot set headers) works in dev.
const apiTarget = process.env.VORAG_API_DEV_PROXY_TARGET || "http://localhost:8000";
const devApiKey = process.env.VORAG_PORTAL_DEV_API_KEY || "";

const configure: ProxyOptions["configure"] = (proxy) => {
  if (devApiKey) {
    proxy.on("proxyReq", (proxyReq: http.ClientRequest) => {
      proxyReq.setHeader("X-API-Key", devApiKey);
    });
  }
};

const proxyEntry: ProxyOptions = {
  target: apiTarget,
  changeOrigin: true,
  configure,
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    proxy: {
      "/health": proxyEntry,
      "/status": proxyEntry,
      "/api": proxyEntry,
    },
  },
});
