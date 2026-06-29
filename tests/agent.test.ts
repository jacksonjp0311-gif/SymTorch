import { describe, expect, it } from "vitest";
import { RuleAgent } from "@symtorch/agent";
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

describe("@symtorch/agent", () => {
  it("uses fact-store working memory for rule decisions", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.9, approved: 0.2 });

    const decision = agent.decide();
    expect(decision.action).toBe("escalate(X)");
    expect(decision.results[0]?.explanation.rules[0]?.predicates[0]?.detail?.key).toBe("high_risk");
  });

  it("selects actions after aggregating same-head rules", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X).
      escalate(X) :- customer_vip(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("customer_vip"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.6);

    agent.observe({ high_risk: 0.4, customer_vip: 0.5 });

    const decision = agent.decide();
    expect(decision.action).toBe("escalate(X)");
    expect(decision.results[0]?.score.item()).toBeCloseTo(0.7, 5);
    expect(decision.results[0]?.explanation.ruleCount).toBe(2);
  });

  it("supports entity-scoped working memory snapshots", () => {
    const agent = new RuleAgent(new RuleProgram("noop(X)."), new FuzzyRuleEngine(() => { throw new Error("unused"); }));
    agent.memory.observeEntity("case-1", { risk: 0.8, approved: 0.1 });
    agent.memory.observeEntity("case-2", { risk: 0.2, approved: 0.9 });

    expect(agent.memory.entitySnapshot("case-1")).toMatchObject({ entity: "case-1", risk: 0.8, approved: 0.1 });
    expect(agent.memory.entitySnapshot("case-2")).toMatchObject({ entity: "case-2", risk: 0.2, approved: 0.9 });
  });
});
