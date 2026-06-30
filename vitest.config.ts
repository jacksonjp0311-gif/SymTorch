import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    alias: {
      "@symtorch/core": fileURLToPath(new URL("./packages/core/src/index.ts", import.meta.url)),
      "@symtorch/nn": fileURLToPath(new URL("./packages/nn/src/index.ts", import.meta.url)),
      "@symtorch/logic": fileURLToPath(new URL("./packages/logic/src/index.ts", import.meta.url)),
      "@symtorch/agent/node": fileURLToPath(new URL("./packages/agent/src/node.ts", import.meta.url)),
      "@symtorch/agent": fileURLToPath(new URL("./packages/agent/src/index.ts", import.meta.url)),
      "@symtorch/webgpu": fileURLToPath(new URL("./packages/webgpu/src/index.ts", import.meta.url))
    }
  },
  test: {
    include: ["tests/**/*.test.ts"],
    pool: "forks"
  }
});
