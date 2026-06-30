import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { AGENT_DECISION_SCHEMA_VERSION, DECISION_LEDGER_SCHEMA_VERSION } from "@symtorch/agent";
import { EXPLANATION_SCHEMA_VERSION } from "@symtorch/logic";
import {
  PLAYGROUND_STATE_VERSION,
  SCENARIO_SCHEMA_VERSION,
  TRAINING_RUN_SCHEMA_VERSION
} from "../examples/browser-playground/src/app-model";

type ReleaseManifest = {
  version: string;
  status: string;
  schemaVersions: {
    explanation: string;
    agentDecision: string;
    decisionLedger: string;
    playgroundState: string;
    scenario: string;
    trainingRun: string;
  };
  validationGates: string[];
  nonClaims: string[];
};

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")) as T;
}

describe("production readiness manifest", () => {
  it("matches package and runtime schema versions", () => {
    const manifest = readJson<ReleaseManifest>("../docs/release-manifest.json");
    const rootPackage = readJson<{ version: string }>("../package.json");

    expect(manifest.version).toBe(rootPackage.version);
    expect(manifest.status).toBe("webgpu-stable-logsumexp-kernel");
    expect(manifest.schemaVersions).toEqual({
      explanation: EXPLANATION_SCHEMA_VERSION,
      agentDecision: AGENT_DECISION_SCHEMA_VERSION,
      decisionLedger: DECISION_LEDGER_SCHEMA_VERSION,
      playgroundState: PLAYGROUND_STATE_VERSION,
      scenario: SCENARIO_SCHEMA_VERSION,
      trainingRun: TRAINING_RUN_SCHEMA_VERSION
    });
  });

  it("documents the required validation gate and non-claims", () => {
    const manifest = readJson<ReleaseManifest>("../docs/release-manifest.json");

    expect(manifest.validationGates).toEqual([
      "pnpm install --frozen-lockfile",
      "pnpm typecheck",
      "pnpm test",
      "pnpm playground:test",
      "pnpm build",
      "pnpm playground:build",
      "pnpm exec tsx scripts/smoke-browser-playground.ts",
      "pnpm playground:e2e",
      "pnpm demo:all"
    ]);
    expect(manifest.nonClaims).toContain("not a production authorization system");
    expect(manifest.nonClaims).toContain("not an npm stability guarantee");
  });
});
