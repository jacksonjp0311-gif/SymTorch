import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { RuleAgent, serializeDecisionLedger } from "@symtorch/agent";
import { FileDecisionLedgerSink } from "@symtorch/agent/node";
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

const execFileAsync = promisify(execFile);

describe("policy replay CLI", () => {
  it("passes for matching policy replay and fails on decision drift", async () => {
    const dir = await mkdtemp(join(tmpdir(), "symtorch-replay-cli-"));
    try {
      const ledgerPath = join(dir, "ledger.json");
      const programText = "escalate(X) :- high_risk(X), not approved(X).";
      const program = new RuleProgram(programText);
      const registry = new PredicateRegistry()
        .register(new FactPredicate("high_risk"))
        .register(new FactPredicate("approved"));
      const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
      agent.memory.observeEntity("case-hot", { high_risk: 0.9, approved: 0.1 });
      agent.recordEntityDecision("case-hot", new Date("2026-06-30T00:00:00.000Z"));
      const sink = new FileDecisionLedgerSink(ledgerPath);
      const snapshot = serializeDecisionLedger(agent.ledger);
      await sink.write(snapshot);

      const pass = await runReplay(ledgerPath, programText, "0.5");
      const json = await runReplay(ledgerPath, programText, "0.5", "--json");
      const fail = await runReplay(ledgerPath, programText, "0.95").catch((error: unknown) => error as {
        code: number;
        stdout: string;
        stderr: string;
      });
      const driftedPath = join(dir, "drifted-ledger.json");
      const drifted = JSON.parse(JSON.stringify(snapshot));
      drifted.entries[0].decision.score += 0.001;
      await new FileDecisionLedgerSink(driftedPath).write(drifted);
      const tolerant = await runReplay(driftedPath, programText, "0.5", "--atol", "0.01");

      expect(pass.stdout).toContain("PASS (1 decisions checked)");
      expect(JSON.parse(json.stdout)).toMatchObject({ ok: true, checked: 1, mismatches: [] });
      expect(fail.code).toBe(1);
      expect(fail.stderr).toContain("FAIL (1/1 mismatched)");
      expect(fail.stderr).toContain("decision-1");
      expect(tolerant.stdout).toContain("PASS (1 decisions checked)");
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

async function runReplay(
  ledger: string,
  program: string,
  threshold: string,
  ...extraArgs: string[]
): Promise<{ stdout: string; stderr: string }> {
  const repoRoot = fileURLToPath(new URL("..", import.meta.url));
  return execFileAsync(process.execPath, [
    join(repoRoot, "node_modules", "tsx", "dist", "cli.mjs"),
    "scripts/replay-ledger.ts",
    "--ledger",
    ledger,
    "--program",
    program,
    "--predicates",
    "high_risk,approved",
    "--threshold",
    threshold,
    ...extraArgs
  ], {
    cwd: repoRoot,
    encoding: "utf8"
  });
}
