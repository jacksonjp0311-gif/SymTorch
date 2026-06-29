import { describe, expect, it } from "vitest";
import { RuleProgram } from "@symtorch/logic";
import {
  buildAgent,
  createFactRegistry,
  defaultCases,
  defaultRule,
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
});
