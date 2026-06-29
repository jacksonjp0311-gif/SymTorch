import { describe, expect, it } from "vitest";
import { RuleAgent } from "@symtorch/agent";
import { EXPLANATION_SCHEMA_VERSION, FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

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

  it("returns a stable serialized decision contract", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      defer(X) :- approved(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.9, approved: 0.2 });

    const decision = agent.decideTrace();
    const roundTrip = JSON.parse(JSON.stringify(decision));

    expect(decision).toMatchObject({
      action: "escalate(X)",
      selectedHead: "escalate(X)",
      threshold: 0.5,
      accepted: true
    });
    expect(decision.score).toBeCloseTo(0.72, 5);
    expect(decision.trace?.schemaVersion).toBe(EXPLANATION_SCHEMA_VERSION);
    expect(decision.trace?.type).toBe("aggregate");
    expect(decision.trace?.rules[0]?.predicates[0]?.detail).toEqual({ key: "high_risk" });
    expect(decision.results.map((result) => result.head)).toEqual(["escalate(X)", "defer(X)"]);
    expect(roundTrip).toEqual(decision);
  });

  it("keeps selected trace while returning no_action below threshold", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.8);

    agent.observe({ high_risk: 0.4 });

    const decision = agent.decideTrace();

    expect(decision.action).toBe("no_action");
    expect(decision.selectedHead).toBe("escalate(X)");
    expect(decision.accepted).toBe(false);
    expect(decision.score).toBeCloseTo(0.4, 5);
    expect(decision.trace?.head).toBe("escalate(X)");
  });

  it("supports entity-scoped working memory snapshots", () => {
    const agent = new RuleAgent(new RuleProgram("noop(X)."), new FuzzyRuleEngine(() => { throw new Error("unused"); }));
    agent.memory.observeEntity("case-1", { risk: 0.8, approved: 0.1 });
    agent.memory.observeEntity("case-2", { risk: 0.2, approved: 0.9 });

    expect(agent.memory.entitySnapshot("case-1")).toMatchObject({ entity: "case-1", risk: 0.8, approved: 0.1 });
    expect(agent.memory.entitySnapshot("case-2")).toMatchObject({ entity: "case-2", risk: 0.2, approved: 0.9 });
  });
});
