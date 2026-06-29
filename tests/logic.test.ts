import { describe, expect, it } from "vitest";
import { tensor } from "@symtorch/core";
import { mseLoss, SGD } from "@symtorch/nn";
import { FactPredicate, FactStore, FuzzyRuleEngine, LinearPredicate, PredicateRegistry, RuleProgram, RuleTrainer, ThresholdPredicate } from "@symtorch/logic";

describe("@symtorch/logic", () => {
  it("evaluates differentiable fuzzy rules with explanations", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const engine = new FuzzyRuleEngine((call) => tensor(call.name === "high_risk" ? 0.9 : 0.2));
    const result = engine.evaluate(program.rules[0]!);
    expect(result.score.item()).toBeCloseTo(0.72, 5);
    expect(result.explanation.predicates).toHaveLength(2);
    expect(result.explanation.head).toBe("escalate(X)");
  });

  it("evaluates registered learnable predicates and exposes trace metadata", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const registry = new PredicateRegistry()
      .register(new ThresholdPredicate("high_risk", "risk", 0.5))
      .fixed("approved", () => tensor(0.1));

    const result = new FuzzyRuleEngine(registry).evaluate(program.rules[0]!, { risk: 0.9 });

    expect(result.score.item()).toBeGreaterThan(0.5);
    expect(result.explanation.predicates[0]?.kind).toBe("learnable");
    expect(result.explanation.predicates[0]?.detail?.threshold).toBeTypeOf("number");
    expect(registry.parameters()).toHaveLength(1);
  });

  it("builds rule contexts from fact stores and fact predicates", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const facts = new FactStore()
      .setEntity("case-1", { high_risk: 0.9, approved: 0.2 })
      .setEntity("case-2", { high_risk: 0.3, approved: 0.1 });
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const engine = new FuzzyRuleEngine(registry);

    expect(engine.evaluate(program.rules[0]!, facts.entityContext("case-1")).score.item()).toBeCloseTo(0.72, 5);
    expect(engine.evaluate(program.rules[0]!, facts.entityContext("case-2")).score.item()).toBeCloseTo(0.27, 5);
  });

  it("trains a threshold predicate through a fuzzy rule", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const predicate = new ThresholdPredicate("high_risk", "risk", 0.9, 10);
    const registry = new PredicateRegistry().register(predicate);
    const engine = new FuzzyRuleEngine(registry);
    const optim = new SGD(registry.parameters(), 0.2);
    const data = [
      { risk: 0.2, label: 0 },
      { risk: 0.35, label: 0 },
      { risk: 0.75, label: 1 },
      { risk: 0.9, label: 1 }
    ];

    const before = predicate.threshold.item();
    for (let epoch = 0; epoch < 80; epoch++) {
      for (const row of data) {
        optim.zeroGrad();
        const result = engine.evaluate(program.rules[0]!, row);
        const loss = mseLoss(result.score, tensor(row.label));
        loss.backward();
        optim.step();
      }
    }

    expect(predicate.threshold.item()).toBeLessThan(before);
    expect(engine.evaluate(program.rules[0]!, { risk: 0.8 }).score.item()).toBeGreaterThan(0.7);
    expect(engine.evaluate(program.rules[0]!, { risk: 0.25 }).score.item()).toBeLessThan(0.3);
  });

  it("trains a linear predicate over feature vectors", () => {
    const program = new RuleProgram("escalate(X) :- suspicious(X).");
    const predicate = new LinearPredicate("suspicious", "features", 2);
    const registry = new PredicateRegistry().register(predicate);
    const engine = new FuzzyRuleEngine(registry);
    const optim = new SGD(registry.parameters(), 0.4);
    const data = [
      { features: [0.1, 0.1], label: 0 },
      { features: [0.2, 0.0], label: 0 },
      { features: [0.8, 0.7], label: 1 },
      { features: [0.9, 0.6], label: 1 }
    ];

    for (let epoch = 0; epoch < 120; epoch++) {
      for (const row of data) {
        optim.zeroGrad();
        const result = engine.evaluate(program.rules[0]!, row);
        mseLoss(result.score, tensor(row.label)).backward();
        optim.step();
      }
    }

    expect(engine.evaluate(program.rules[0]!, { features: [0.95, 0.8] }).score.item()).toBeGreaterThan(0.7);
    expect(engine.evaluate(program.rules[0]!, { features: [0.05, 0.1] }).score.item()).toBeLessThan(0.35);
  });

  it("trains rules through RuleTrainer", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const highRisk = new ThresholdPredicate("high_risk", "risk", 0.85, 10);
    const registry = new PredicateRegistry()
      .register(highRisk)
      .fixed("approved", (_call, context) => tensor(typeof context.approved === "number" ? context.approved : 0));
    const engine = new FuzzyRuleEngine(registry);
    const trainer = new RuleTrainer(engine, program.rules[0]!, registry, { learningRate: 0.2 });

    const result = trainer.fit([
      { risk: 0.2, approved: 0.05, label: 0 },
      { risk: 0.4, approved: 0.1, label: 0 },
      { risk: 0.75, approved: 0.05, label: 1 },
      { risk: 0.92, approved: 0.1, label: 1 },
      { risk: 0.9, approved: 0.95, label: 0 }
    ], { epochs: 80 });

    expect(result.history).toHaveLength(80);
    expect(result.finalLoss).toBeLessThan(result.history[0]!.loss);
    expect(highRisk.threshold.item()).toBeLessThan(0.85);
    expect(trainer.predict({ risk: 0.82, approved: 0.05 }).score.item()).toBeGreaterThan(0.75);
  });
});
