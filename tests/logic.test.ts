import { describe, expect, it } from "vitest";
import { tensor } from "@symtorch/core";
import { FuzzyRuleEngine, RuleProgram } from "@symtorch/logic";

describe("@symtorch/logic", () => {
  it("evaluates differentiable fuzzy rules with explanations", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const engine = new FuzzyRuleEngine((call) => tensor(call.name === "high_risk" ? 0.9 : 0.2));
    const result = engine.evaluate(program.rules[0]!);
    expect(result.score.item()).toBeCloseTo(0.72, 5);
    expect(result.explanation.predicates).toHaveLength(2);
    expect(result.explanation.head).toBe("escalate(X)");
  });
});

