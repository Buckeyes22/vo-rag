import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
// Dev-server proxy target — the vo-rag FastAPI surface. Override with
// VORAG_API_DEV_PROXY_TARGET; VORAG_PORTAL_DEV_API_KEY injects the API key
// server-side so EventSource (which cannot set headers) works in dev.
var apiTarget = process.env.VORAG_API_DEV_PROXY_TARGET || "http://localhost:8000";
var devApiKey = process.env.VORAG_PORTAL_DEV_API_KEY || "";
var configure = function (proxy) {
    if (devApiKey) {
        proxy.on("proxyReq", function (proxyReq) {
            proxyReq.setHeader("X-API-Key", devApiKey);
        });
    }
};
var proxyEntry = {
    target: apiTarget,
    changeOrigin: true,
    configure: configure,
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
