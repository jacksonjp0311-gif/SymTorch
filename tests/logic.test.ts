import { describe, expect, it } from "vitest";
import { ResourceLimitError, tensor } from "@symtorch/core";
import { mseLoss, SGD } from "@symtorch/nn";
import { assessPolicyBundleSecurity, createDomainContract, createPolicyBundle, decisionCard, decisionTrace, DOMAIN_CONTRACT_SCHEMA_VERSION, EXPLANATION_SCHEMA_VERSION, FactPredicate, FactStore, FuzzyRuleEngine, getProductionReadinessReport, isSerializedPolicyBundle, LinearPredicate, loadPolicyBundle, POLICY_BUNDLE_SCHEMA_VERSION, POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION, productionRuntimeLimits, PRODUCTION_READINESS_SCHEMA_VERSION, parseProgram, PredicateEvaluationError, PredicateRegistry, renderAggregatedExplanation, renderRuleExplanation, RuleParseError, RuleProgram, RuleTrainer, serializeExplanation, signPolicyBundle, ThresholdPredicate, validateDomainContext, validateProgram, validatePrograms, verifyPolicyBundleHash, verifySignedPolicyBundle, verifySignedPolicyBundleDetailed } from "@symtorch/logic";

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

  it("emits rule and program evaluation observer events", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      defer(X) :- approved(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const events: unknown[] = [];
    const engine = new FuzzyRuleEngine(registry, {
      observer: {
        onRuleEvaluate: (event) => events.push(event),
        onProgramEvaluate: (event) => events.push(event)
      }
    });

    engine.evaluateProgramGrouped(program, { high_risk: 0.9, approved: 0.2 });

    expect(events).toHaveLength(3);
    expect(events[0]).toMatchObject({
      kind: "rule.evaluate",
      rule: "escalate(X) :- high_risk(X), not approved(X).",
      head: "escalate(X)",
      predicateCount: 2,
      contextKeys: ["approved", "high_risk"]
    });
    expect(events[1]).toMatchObject({
      kind: "rule.evaluate",
      rule: "defer(X) :- approved(X).",
      head: "defer(X)",
      predicateCount: 1,
      contextKeys: ["approved", "high_risk"]
    });
    expect(events[2]).toMatchObject({
      kind: "program.evaluate",
      ruleCount: 2,
      groupCount: 2,
      contextKeys: ["approved", "high_risk"]
    });
    expect((events[0] as { durationMs: number }).durationMs).toBeGreaterThanOrEqual(0);
    expect((events[2] as { durationMs: number }).durationMs).toBeGreaterThanOrEqual(0);
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

  it("validates rule programs without throwing for authoring loops", () => {
    const valid = validateProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      defer(X) :- approved(X).
    `);
    const invalid = validateProgram(`
      escalate(X) :- high-risk(X).
    `);

    expect(valid.ok).toBe(true);
    if (valid.ok) {
      expect(valid.rules).toHaveLength(2);
      expect(valid.rules[0]?.source).toBe("escalate(X) :- high_risk(X), not approved(X).");
    }

    expect(invalid.ok).toBe(false);
    if (!invalid.ok) {
      expect(invalid.error).toBeInstanceOf(RuleParseError);
      expect(invalid.error.line).toBe(2);
      expect(invalid.error.message).toContain("Invalid predicate call");
    }
  });

  it("validates predicate bindings against a registry before runtime", () => {
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const missing = validateProgram("escalate(X) :- high_risk(X), not approved(X).", { registry });
    const bound = validateProgram("escalate(X) :- high_risk(X).", { registry });

    expect(bound.ok).toBe(true);
    expect(missing.ok).toBe(false);
    if (!missing.ok) {
      expect(missing.rules).toHaveLength(1);
      expect(missing.diagnostics).toEqual([{
        code: "missing_predicate",
        severity: "error",
        predicate: "approved",
        message: "Predicate \"approved\" is not registered."
      }]);
      expect(missing.error).toBe(missing.diagnostics[0]);
    }
  });

  it("validates many rule drafts in one authoring run", () => {
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const batch = validatePrograms([
      { id: "good", source: "escalate(X) :- high_risk(X), not approved(X)." },
      { id: "missing", source: "escalate(X) :- customer_vip(X)." },
      { id: "syntax", source: "escalate(X) :- high-risk(X)." }
    ], { registry });

    expect(batch.map((item) => item.id)).toEqual(["good", "missing", "syntax"]);
    expect(batch.map((item) => item.result.ok)).toEqual([true, false, false]);
    expect(batch[0]?.result.diagnostics).toEqual([]);
    expect(batch[1]?.result.diagnostics[0]?.code).toBe("missing_predicate");
    expect(batch[2]?.result.diagnostics[0]?.code).toBe("parse_error");
  });

  it("enforces rule evaluation limits and typed predicate errors", () => {
    const source = "escalate(X) :- high_risk(X), not approved(X).";

    expect(() => parseProgram(source, { limits: { maxRuleSourceLength: 10 } })).toThrow(ResourceLimitError);
    expect(() => new RuleProgram(source, { limits: { maxPredicatesPerRule: 1 } })).toThrow(ResourceLimitError);

    const program = new RuleProgram(source);
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const engine = new FuzzyRuleEngine(registry);

    expect(() => engine.evaluate(program.rules[0]!, { high_risk: 0.8 })).toThrow(PredicateEvaluationError);
  });

  it("creates and verifies versioned policy bundles", () => {
    const bundle = createPolicyBundle({
      name: "Escalation Policy",
      version: "2026.06.30",
      rules: "escalate(X) :- high_risk(X), not approved(X).",
      predicates: [
        { kind: "fact", name: "approved" },
        { kind: "threshold", name: "high_risk", valueKey: "risk", threshold: 0.7, slope: 10 }
      ],
      metadata: {
        owner: "risk",
        trainable: true
      }
    });

    expect(bundle.schemaVersion).toBe(POLICY_BUNDLE_SCHEMA_VERSION);
    expect(bundle.hash).toMatch(/^fnv1a32:/);
    expect(isSerializedPolicyBundle(bundle)).toBe(true);
    expect(verifyPolicyBundleHash(bundle)).toBe(true);
    expect(isSerializedPolicyBundle({ ...bundle, rules: "tampered(X)." })).toBe(false);
  });

  it("loads policy bundles into executable programs and predicates", () => {
    const bundle = createPolicyBundle({
      name: "Escalation Policy",
      version: "2026.06.30",
      rules: `
        escalate(X) :- high_risk(X), not approved(X).
        suspicious(X) :- suspicious_features(X).
      `,
      predicates: [
        { kind: "fact", name: "approved" },
        { kind: "threshold", name: "high_risk", valueKey: "risk", threshold: 0.7, slope: 12 },
        { kind: "linear", name: "suspicious_features", featureKey: "features", featureCount: 2, weights: [2, 1], bias: -1 }
      ],
      metadata: { owner: "risk" }
    });

    const loaded = loadPolicyBundle(bundle);
    const escalation = loaded.engine.evaluateProgramGrouped(loaded.program, {
      risk: 0.9,
      approved: 0.1,
      features: [0.9, 0.8]
    });
    const highRisk = loaded.registry.resolve(loaded.program.rules[0]!.body[0]!, { risk: 0.7 });
    const linear = loaded.registry.resolve(loaded.program.rules[1]!.body[0]!, { features: [1, 0] });

    expect(escalation.find((result) => result.head === "escalate(X)")?.score.item()).toBeGreaterThan(0.7);
    expect(highRisk.score.item()).toBeCloseTo(0.5, 5);
    expect(highRisk.detail?.threshold).toBeCloseTo(0.7, 5);
    expect(highRisk.detail).toMatchObject({ slope: 12, valueKey: "risk" });
    expect(linear.score.item()).toBeCloseTo(1 / (1 + Math.exp(-1)), 5);
  });

  it("rejects tampered policy bundles before loading", () => {
    const bundle = createPolicyBundle({
      name: "Escalation Policy",
      version: "2026.06.30",
      rules: "escalate(X) :- high_risk(X).",
      predicates: [{ kind: "fact", name: "high_risk" }],
      metadata: {}
    });

    expect(() => loadPolicyBundle({ ...bundle, rules: "escalate(X) :- low_risk(X)." })).toThrow("valid hash");
  });

  it("validates typed domain contexts", () => {
    const contract = createDomainContract({
      case: {
        fields: {
          high_risk: { type: "number", min: 0, max: 1 },
          approved: { type: "number", min: 0, max: 1 },
          label: { type: "string", required: false }
        }
      }
    });

    expect(contract.schemaVersion).toBe(DOMAIN_CONTRACT_SCHEMA_VERSION);
    expect(validateDomainContext(contract, "case", { high_risk: 0.8, approved: 0.1 }).ok).toBe(true);
    const invalid = validateDomainContext(contract, "case", { high_risk: 2, approved: "no" });
    expect(invalid.ok).toBe(false);
    expect(invalid.diagnostics.map((item) => item.path)).toEqual(["$.case.high_risk", "$.case.approved"]);
  });

  it("signs policy bundles and rejects signature drift", () => {
    const bundle = createPolicyBundle({
      name: "Signed Policy",
      version: "test",
      rules: "escalate(X) :- high_risk(X).",
      predicates: [{ kind: "fact", name: "high_risk" }],
      metadata: { purpose: "test" }
    });
    const signed = signPolicyBundle(bundle, "local-dev", "secret");
    const tampered = { ...signed, hash: "fnv1a32:00000000" };

    expect(signed.signature.schemaVersion).toBe(POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION);
    expect(verifySignedPolicyBundle(signed, { "local-dev": "secret" })).toBe(true);
    expect(verifySignedPolicyBundle(signed, { "local-dev": "wrong" })).toBe(false);
    expect(verifySignedPolicyBundle(tampered, { "local-dev": "secret" })).toBe(false);
    expect(verifySignedPolicyBundleDetailed(signed, { "local-dev": "secret" })).toMatchObject({ ok: true, keyId: "local-dev" });
    expect(verifySignedPolicyBundleDetailed(signed, { "other": "secret" })).toEqual({ ok: false, reason: "unknown_key" });
  });

  it("assesses policy bundle security and exposes production readiness tracks", () => {
    const bundle = createPolicyBundle({
      name: "Assessed Policy",
      version: "test",
      rules: "escalate(X) :- high_risk(X).",
      predicates: [{ kind: "fact", name: "high_risk" }],
      metadata: { owner: "test" }
    });
    const signed = signPolicyBundle(bundle, "local-dev", "secret");
    const report = getProductionReadinessReport("0.29.0");
    const limits = productionRuntimeLimits({ maxRules: 8 });

    expect(limits.maxRules).toBe(8);
    expect(report.schemaVersion).toBe(PRODUCTION_READINESS_SCHEMA_VERSION);
    expect(report.productionReady).toBe(false);
    expect(report.tracks.map((track) => track.id)).toEqual([
      "typed_domains",
      "bundle_signing",
      "durable_persistence",
      "trace_snapshots",
      "runtime_limits",
      "error_taxonomy",
      "cpu_gpu_parity",
      "api_stability",
      "security_model",
      "real_apps"
    ]);
    expect(assessPolicyBundleSecurity(signed, {
      secrets: { "local-dev": "secret" },
      trustedKeyIds: ["local-dev"]
    }).ok).toBe(true);
    expect(assessPolicyBundleSecurity(signed, {
      secrets: { "local-dev": "secret" },
      trustedKeyIds: ["other"]
    }).diagnostics[0]).toMatchObject({ code: "untrusted_key" });
    expect(assessPolicyBundleSecurity(bundle).diagnostics[0]).toMatchObject({ code: "security_boundary" });
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
