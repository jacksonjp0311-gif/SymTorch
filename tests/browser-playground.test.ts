import { describe, expect, it } from "vitest";
import { RuleProgram } from "@symtorch/logic";
import {
  buildAgent,
  createFactRegistry,
  createPlaygroundState,
  defaultCases,
  defaultRule,
  exportPlaygroundState,
  parsePlaygroundState,
  trainHighRiskRule,
  validateRuleSource
} from "../examples/browser-playground/src/app-model";

describe("browser playground model", () => {
  it("validates the default editable policy", () => {
    const validation = validateRuleSource(defaultRule, createFactRegistry());

    expect(validation.ok).toBe(true);
    expect(validation.diagnostics).toEqual([]);
  });

  it("ranks entity decisions and records accepted ledger entries", () => {
    const agent = buildAgent(new RuleProgram(defaultRule), defaultCases());
    const ranked = agent.decideEntitiesTrace();
    const accepted = agent.recordEntityDecisions(
      { acceptedOnly: true, topK: 2 },
      new Date("2026-06-29T00:00:00.000Z")
    );

    expect(ranked).toHaveLength(4);
    expect(ranked[0]?.entityId).toBe("case-approved");
    expect(ranked[0]?.action).toBe("defer(X)");
    expect(ranked.some((decision) => decision.entityId === "case-hot" && decision.action === "escalate(X)")).toBe(true);
    expect(accepted).toHaveLength(2);
    expect(agent.ledger.all()).toEqual(accepted);
    expect(JSON.parse(JSON.stringify(agent.ledger.all()))).toEqual(agent.ledger.all());
  });

  it("trains the high-risk threshold and preserves explanation output", () => {
    const result = trainHighRiskRule(defaultRule, 0.9);

    expect(result.afterThreshold).toBeLessThan(result.beforeThreshold);
    expect(result.afterScore).toBeGreaterThan(result.beforeScore);
    expect(result.finalLoss).toBeLessThan(0.1);
    expect(result.historyLength).toBe(100);
    expect(result.explanationPredicateCount).toBe(2);
    expect(JSON.stringify(result.explanationJson)).toContain("high_risk");
  });

  it("round-trips versioned playground state and rejects invalid state", () => {
    const state = createPlaygroundState(defaultRule, defaultCases(), 0.42);
    const roundTrip = parsePlaygroundState(JSON.stringify(state));

    expect(roundTrip).toEqual(state);
    expect(parsePlaygroundState(null)).toBeNull();
    expect(parsePlaygroundState("{")).toBeNull();
    expect(parsePlaygroundState(JSON.stringify({ ...state, schemaVersion: "old" }))).toBeNull();
    expect(parsePlaygroundState(JSON.stringify({ ...state, cases: [{ entityId: "bad", high_risk: 2, approved: -1 }] })))
      .toMatchObject({
        cases: [{ entityId: "bad", high_risk: 1, approved: 0 }]
      });
  });

  it("exports readable versioned playground state", () => {
    const exported = exportPlaygroundState(defaultRule, defaultCases(), 0.5);
    const parsed = parsePlaygroundState(exported);

    expect(exported).toContain("\n");
    expect(exported).toContain("symtorch.playground.v1");
    expect(parsed?.ruleSource).toBe(defaultRule);
    expect(parsed?.trainedThreshold).toBe(0.5);
  });
});
