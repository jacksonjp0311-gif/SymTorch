import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createDecisionTraceSnapshot, DecisionLedger, isDecisionTraceSnapshot, RuleAgent } from "@symtorch/agent";
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")) as T;
}

describe("golden trace corpus", () => {
  it("matches the checked-in escalation decision trace", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
    const ledger = new DecisionLedger();

    agent.memory.observeEntity("case-hot", { high_risk: 0.9, approved: 0.1 });
    const decision = agent.decideEntityTrace("case-hot");
    ledger.append({
      kind: "entity",
      context: agent.memory.entitySnapshot("case-hot"),
      decision
    }, new Date("2026-06-30T12:00:00.000Z"));
    const snapshot = createDecisionTraceSnapshot(decision, {
      ledger,
      createdAt: new Date("2026-06-30T12:00:00.000Z")
    });
    const expected = readJson<unknown>("../examples/traces/golden-escalation.trace.json");

    expect(isDecisionTraceSnapshot(expected)).toBe(true);
    expect(snapshot).toEqual(expected);
  });
});
