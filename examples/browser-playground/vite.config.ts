import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root,
  resolve: {
    alias: {
      "@symtorch/agent": fileURLToPath(new URL("../../packages/agent/src/index.ts", import.meta.url)),
      "@symtorch/core": fileURLToPath(new URL("../../packages/core/src/index.ts", import.meta.url)),
      "@symtorch/logic": fileURLToPath(new URL("../../packages/logic/src/index.ts", import.meta.url)),
      "@symtorch/nn": fileURLToPath(new URL("../../packages/nn/src/index.ts", import.meta.url))
    }
  },
  server: {
    host: "127.0.0.1",
    port: 5173
  }
});
