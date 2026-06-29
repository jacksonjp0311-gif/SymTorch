import { describe, expect, it } from "vitest";
import { tensor } from "@symtorch/core";
import { mseLoss, SGD } from "@symtorch/nn";
import { decisionCard, decisionTrace, EXPLANATION_SCHEMA_VERSION, FactPredicate, FactStore, FuzzyRuleEngine, LinearPredicate, parseProgram, PredicateRegistry, renderAggregatedExplanation, renderRuleExplanation, RuleParseError, RuleProgram, RuleTrainer, serializeExplanation, ThresholdPredicate } from "@symtorch/logic";

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
    expect(facts.entityIds()).toEqual(["case-1", "case-2"]);
  });

  it("aggregates multiple rules with the same head using probabilistic OR", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X).
      escalate(X) :- customer_vip(X).
      defer(X) :- approved(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("customer_vip"))
      .register(new FactPredicate("approved"));
    const results = new FuzzyRuleEngine(registry).evaluateProgramGrouped(program, {
      high_risk: 0.4,
      customer_vip: 0.5,
      approved: 0.2
    });

    const escalate = results.find((result) => result.head === "escalate(X)");
    const defer = results.find((result) => result.head === "defer(X)");
    expect(escalate?.score.item()).toBeCloseTo(0.7, 5);
    expect(escalate?.explanation.ruleCount).toBe(2);
    expect(escalate?.explanation.rules.map((rule) => rule.rule)).toEqual([
      "escalate(X) :- high_risk(X).",
      "escalate(X) :- customer_vip(X)."
    ]);
    expect(defer?.score.item()).toBeCloseTo(0.2, 5);
  });

  it("evaluates and ranks entity batches by rule head", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      escalate(X) :- customer_vip(X).
      defer(X) :- approved(X).
    `);
    const facts = new FactStore()
      .setEntity("case-low", { high_risk: 0.2, approved: 0.1, customer_vip: 0.1 })
      .setEntity("case-approved", { high_risk: 0.9, approved: 0.95, customer_vip: 0.1 })
      .setEntity("case-hot", { high_risk: 0.9, approved: 0.1, customer_vip: 0.2 })
      .setEntity("case-vip", { high_risk: 0.3, approved: 0.1, customer_vip: 0.8 });
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"))
      .register(new FactPredicate("customer_vip"));
    const engine = new FuzzyRuleEngine(registry);

    const evaluated = engine.evaluateEntities(program, facts);
    const ranked = engine.rankEntitiesByHead(program, facts, "escalate(X)");

    expect(evaluated).toHaveLength(4);
    expect(evaluated[0]?.results.some((result) => result.head === "escalate(X)")).toBe(true);
    expect(ranked.map((item) => item.entityId)).toEqual(["case-vip", "case-hot", "case-low", "case-approved"]);
    expect(ranked[0]?.result.score.item()).toBeGreaterThan(0.85);
  });

  it("renders rule and aggregate explanations as decision cards", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X).
      escalate(X) :- customer_vip(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("customer_vip"));
    const engine = new FuzzyRuleEngine(registry);
    const single = engine.evaluate(program.rules[0]!, { high_risk: 0.4 });
    const aggregate = engine.evaluateProgramGrouped(program, { high_risk: 0.4, customer_vip: 0.5 })[0]!;

    expect(renderRuleExplanation(single.explanation)).toContain("rule: escalate(X) :- high_risk(X).");
    expect(renderAggregatedExplanation(aggregate.explanation)).toContain("from 2 rules");
    expect(decisionCard(aggregate)).toContain("customer_vip(X)");
  });

  it("serializes explanations into a versioned agent-safe schema", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      escalate(X) :- customer_vip(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"))
      .register(new FactPredicate("customer_vip"));
    const aggregate = new FuzzyRuleEngine(registry).evaluateProgramGrouped(program, {
      high_risk: 0.8,
      approved: 0.25,
      customer_vip: 0.1
    })[0]!;

    const serialized = serializeExplanation(aggregate.explanation);
    const traced = decisionTrace(aggregate);
    const roundTrip = JSON.parse(JSON.stringify(serialized));

    expect(serialized.schemaVersion).toBe(EXPLANATION_SCHEMA_VERSION);
    expect(serialized.type).toBe("aggregate");
    expect(serialized.head).toBe("escalate(X)");
    expect(serialized.ruleCount).toBe(2);
    expect(serialized.rules[0]?.schemaVersion).toBe(EXPLANATION_SCHEMA_VERSION);
    expect(serialized.rules[0]?.type).toBe("rule");
    expect(serialized.rules[0]?.predicates[0]).toMatchObject({
      name: "high_risk(X)",
      negated: false,
      kind: "fixed",
      detail: { key: "high_risk" }
    });
    expect(serialized.rules[0]?.predicates[1]).toMatchObject({
      name: "not approved(X)",
      negated: true,
      kind: "fixed",
      detail: { key: "approved" }
    });
    expect(traced).toEqual(serialized);
    expect(roundTrip).toEqual(serialized);
  });

  it("reports rule parser diagnostics with line, column, and snippets", () => {
    const source = `
      ok(X) :- known(X).
      escalate(X) :- high-risk(X).
    `;

    expect(() => parseProgram(source)).toThrow(RuleParseError);
    try {
      parseProgram(source);
      throw new Error("Expected parseProgram to fail.");
    } catch (error) {
      expect(error).toBeInstanceOf(RuleParseError);
      const diagnostic = error as RuleParseError;
      expect(diagnostic.line).toBe(3);
      expect(diagnostic.column).toBe(22);
      expect(diagnostic.snippet).toContain("escalate(X) :- high-risk(X).");
      expect(diagnostic.message).toContain("Invalid predicate call");
      expect(diagnostic.message).toContain("^");
    }
  });

  it("points parser diagnostics at invalid terms and unbalanced bodies", () => {
    try {
      parseProgram("escalate(X) :- high_risk(123bad).");
      throw new Error("Expected invalid term to fail.");
    } catch (error) {
      expect(error).toBeInstanceOf(RuleParseError);
      const diagnostic = error as RuleParseError;
      expect(diagnostic.line).toBe(1);
      expect(diagnostic.column).toBe(26);
      expect(diagnostic.message).toContain("Invalid term");
    }

    try {
      parseProgram("escalate(X) :- high_risk(X, approved(X).");
      throw new Error("Expected unbalanced body to fail.");
    } catch (error) {
      expect(error).toBeInstanceOf(RuleParseError);
      expect((error as RuleParseError).message).toContain("Unclosed parenthesis");
    }
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

  it("preserves explanations while training a threshold rule", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const highRisk = new ThresholdPredicate("high_risk", "risk", 0.9, 10);
    const registry = new PredicateRegistry()
      .register(highRisk)
      .fixed("approved", (_call, context) => tensor(typeof context.approved === "number" ? context.approved : 0));
    const engine = new FuzzyRuleEngine(registry);
    const trainer = new RuleTrainer(engine, program.rules[0]!, registry, { learningRate: 0.2 });
    const examples = [
      { risk: 0.2, approved: 0.05, label: 0 },
      { risk: 0.35, approved: 0.15, label: 0 },
      { risk: 0.75, approved: 0.05, label: 1 },
      { risk: 0.9, approved: 0.1, label: 1 },
      { risk: 0.88, approved: 0.95, label: 0 }
    ];

    const thresholdBefore = highRisk.threshold.item();
    const predictionBefore = trainer.predict({ risk: 0.82, approved: 0.08 }).score.item();
    const result = trainer.fit(examples, { epochs: 100 });
    const predictionAfter = trainer.predict({ risk: 0.82, approved: 0.08 });

    expect(result.finalLoss).toBeLessThan(result.history[0]!.loss);
    expect(highRisk.threshold.item()).toBeLessThan(thresholdBefore);
    expect(predictionAfter.score.item()).toBeGreaterThan(predictionBefore);
    expect(predictionAfter.explanation.predicates).toHaveLength(2);
    expect(predictionAfter.explanation.predicates[0]?.name).toBe("high_risk(X)");
    expect(predictionAfter.explanation.predicates[0]?.kind).toBe("learnable");
  });
});
