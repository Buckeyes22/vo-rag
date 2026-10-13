import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect } from "vitest";

const conf = readFileSync(resolve(__dirname, "..", "nginx.conf.template"), "utf8");

function locationBlock(selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = conf.match(new RegExp(`location\\s+${escapedSelector}\\s*\\{[\\s\\S]*?\\n\\s*\\}`, "m"));

  if (!match) {
    throw new Error(`missing nginx location block: ${selector}`);
  }

  return match[0];
}

describe("nginx.conf.template", () => {
  it("SPA fallback: location / uses try_files ... /index.html", () => {
    expect(conf).toMatch(/try_files \$uri \$uri\/ \/index\.html/);
  });

  it.each(["= /health", "= /status", "/api/"])(
    "forwards Cloudflare Access JWT headers and the envsubst upstream in location %s",
    (selector) => {
      const block = locationBlock(selector);

      expect(block).toContain("proxy_pass ${VORAG_API_UPSTREAM};");
      expect(block).toContain(
        "proxy_set_header Cf-Access-Jwt-Assertion $http_cf_access_jwt_assertion;",
      );
    },
  );

  it("keeps the /api/ proxy SSE-safe", () => {
    const block = locationBlock("/api/");

    expect(block).toContain("proxy_read_timeout 1800s;");
    expect(block).toContain("proxy_buffering off;");
    expect(block).toContain("proxy_cache off;");
    expect(block).toContain('proxy_set_header Connection "";');
    expect(block).toContain("proxy_http_version 1.1;");
  });

  it("does not pin any literal IP addresses", () => {
    expect(conf).not.toMatch(/\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b|(?<![a-z0-9])(?:[0-9a-f]{1,4}:){2,}[0-9a-f:]*(?![a-z0-9])/i);
  });

  it("defines only the documented location blocks", () => {
    const selectors = [...conf.matchAll(/location\s+((?:=\s*)?\S+)\s*\{/g)].map((m) => (m[1] ?? "").replace(/\s+/g, " "));

    expect(selectors).toEqual(["/assets/", "/", "= /health", "= /status", "/api/"]);
  });
});
