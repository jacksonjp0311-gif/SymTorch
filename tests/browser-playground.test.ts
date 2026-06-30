import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { FuzzyRuleEngine, loadPolicyBundle, RuleProgram } from "@symtorch/logic";
import { RuleAgent } from "@symtorch/agent";
import {
  buildAgent,
  buildPolicyBundleAgent,
  createFactRegistry,
  createPolicyLibrary,
  createPlaygroundPolicyBundle,
  createPolicyHealth,
  createPlaygroundState,
  createTrainingRun,
  DEFAULT_SCENARIO_ID,
  defaultCases,
  defaultRule,
  defaultScenario,
  defaultTrainingExamples,
  exportPolicyBundleLibrary,
  exportPlaygroundPolicyBundle,
  exportPlaygroundScenario,
  exportPlaygroundState,
  parsePolicyBundleLibrary,
  parsePlaygroundPolicyBundle,
  parsePlaygroundState,
  parsePlaygroundScenario,
  POLICY_LIBRARY_SCHEMA_VERSION,
  playgroundScenarios,
  scenarioIdFromPolicyBundle,
  SCENARIO_SCHEMA_VERSION,
  savePolicyBundleToLibrary,
  summarizeTrainingRun,
  thresholdFromPolicyBundle,
  trainHighRiskRule,
  TRAINING_RUN_SCHEMA_VERSION,
  validatePlaygroundScenario,
  validateRuleSource
} from "../examples/browser-playground/src/app-model";

describe("browser playground model", () => {
  it("validates the default editable policy", () => {
    const validation = validateRuleSource(defaultRule, createFactRegistry());

    expect(validation.ok).toBe(true);
    expect(validation.diagnostics).toEqual([]);
  });

  it("ships valid policy scenarios with decision data", () => {
    expect(playgroundScenarios.map((scenario) => scenario.id)).toEqual([
      "case-escalation",
      "fraud-review",
      "support-routing"
    ]);

    for (const scenario of playgroundScenarios) {
      const validation = validateRuleSource(scenario.ruleSource, createFactRegistry());
      const scenarioValidation = validatePlaygroundScenario(scenario);
      const agent = buildAgent(new RuleProgram(scenario.ruleSource), scenario.cases);
      const decisions = agent.decideEntitiesTrace();

      expect(validation.ok).toBe(true);
      expect(scenarioValidation.ok).toBe(true);
      expect(scenario.cases.length).toBeGreaterThanOrEqual(4);
      expect(scenario.trainingExamples.length).toBeGreaterThanOrEqual(5);
      expect(decisions[0]?.score).toBeGreaterThan(0);
    }
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
    const result = trainHighRiskRule(defaultRule, 0.9, defaultTrainingExamples());

    expect(result.afterThreshold).toBeLessThan(result.beforeThreshold);
    expect(result.afterScore).toBeGreaterThan(result.beforeScore);
    expect(result.finalLoss).toBeLessThan(0.1);
    expect(result.initialLoss).toBeGreaterThan(result.finalLoss);
    expect(result.historyLength).toBe(100);
    expect(result.history).toHaveLength(100);
    expect(result.explanationPredicateCount).toBe(2);
    expect(JSON.stringify(result.explanationJson)).toContain("high_risk");
  });

  it("creates and persists versioned training runs", () => {
    const result = trainHighRiskRule(defaultRule, 0.9, defaultTrainingExamples());
    const run = createTrainingRun(DEFAULT_SCENARIO_ID, result);
    const state = createPlaygroundState(DEFAULT_SCENARIO_ID, defaultRule, defaultCases(), run.finalThreshold, defaultTrainingExamples(), run);
    const roundTrip = parsePlaygroundState(JSON.stringify(state));

    expect(run.schemaVersion).toBe(TRAINING_RUN_SCHEMA_VERSION);
    expect(run.initialLoss).toBeGreaterThan(run.finalLoss);
    expect(summarizeTrainingRun(run)).toContain("epochs: 100");
    expect(roundTrip?.lastTrainingRun).toEqual(run);
  });

  it("uses caller-provided training examples", () => {
    const result = trainHighRiskRule(defaultRule, 0.9, [
      { risk: 0.1, approved: 0.05, label: 0 },
      { risk: 0.2, approved: 0.05, label: 0 },
      { risk: 0.95, approved: 0.05, label: 1 }
    ]);

    expect(result.historyLength).toBe(100);
    expect(result.afterThreshold).toBeLessThan(result.beforeThreshold);
  });

  it("round-trips versioned playground state and rejects invalid state", () => {
    const state = createPlaygroundState(DEFAULT_SCENARIO_ID, defaultRule, defaultCases(), 0.42, defaultTrainingExamples(), null);
    const roundTrip = parsePlaygroundState(JSON.stringify(state));

    expect(roundTrip).toEqual(state);
    expect(parsePlaygroundState(null)).toBeNull();
    expect(parsePlaygroundState("{")).toBeNull();
    expect(parsePlaygroundState(JSON.stringify({ ...state, schemaVersion: "old" }))).toBeNull();
    expect(parsePlaygroundState(JSON.stringify({ ...state, cases: [{ entityId: "bad", high_risk: 2, approved: -1 }] })))
      .toMatchObject({
        cases: [{ entityId: "bad", high_risk: 1, approved: 0 }]
      });
    expect(parsePlaygroundState(JSON.stringify({ ...state, trainingExamples: [{ risk: 2, approved: -1, label: 0.6 }] })))
      .toMatchObject({
        trainingExamples: [{ risk: 1, approved: 0, label: 1 }]
      });
    expect(parsePlaygroundState(JSON.stringify({ ...state, lastTrainingRun: { schemaVersion: "bad" } }))).toBeNull();
  });

  it("keeps old playground state compatible by defaulting an empty policy library", () => {
    const state = createPlaygroundState(DEFAULT_SCENARIO_ID, defaultRule, defaultCases(), 0.42, defaultTrainingExamples(), null);
    const legacyState = { ...state, policyLibrary: undefined };
    delete legacyState.policyLibrary;
    const roundTrip = parsePlaygroundState(JSON.stringify(legacyState));

    expect(roundTrip?.policyLibrary).toEqual(createPolicyLibrary());
  });

  it("exports readable versioned playground state", () => {
    const exported = exportPlaygroundState(defaultScenario().id, defaultRule, defaultCases(), 0.5, defaultTrainingExamples());
    const parsed = parsePlaygroundState(exported);

    expect(exported).toContain("\n");
    expect(exported).toContain("symtorch.playground.v1");
    expect(parsed?.scenarioId).toBe(DEFAULT_SCENARIO_ID);
    expect(parsed?.ruleSource).toBe(defaultRule);
    expect(parsed?.trainedThreshold).toBe(0.5);
    expect(parsed?.trainingExamples).toHaveLength(5);
    expect(parsed?.policyLibrary.schemaVersion).toBe(POLICY_LIBRARY_SCHEMA_VERSION);
  });

  it("exports and parses standalone scenario contracts", () => {
    const scenario = defaultScenario();
    const exported = exportPlaygroundScenario(scenario);
    const parsed = parsePlaygroundScenario(exported);

    expect(exported).toContain(SCENARIO_SCHEMA_VERSION);
    expect(parsed.ok).toBe(true);
    if (parsed.ok) {
      expect(parsed.scenario).toEqual(scenario);
    }
  });

  it("exports, parses, and health-checks policy bundles", () => {
    const exported = exportPlaygroundPolicyBundle(DEFAULT_SCENARIO_ID, defaultRule, 0.42);
    const parsed = parsePlaygroundPolicyBundle(exported);

    expect(exported).toContain("symtorch.policyBundle.v1");
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) throw new Error("expected policy bundle");

    const health = createPolicyHealth(parsed.bundle);
    expect(health.schemaVersion).toBe("symtorch.policyBundle.v1");
    expect(health.hashVerified).toBe(true);
    expect(health.ruleCount).toBe(2);
    expect(health.predicateCount).toBe(2);
    expect(thresholdFromPolicyBundle(parsed.bundle)).toBe(0.42);
    expect(scenarioIdFromPolicyBundle(parsed.bundle)).toBe(DEFAULT_SCENARIO_ID);
  });

  it("saves, dedupes, exports, and parses policy bundle libraries", () => {
    const bundle = createPlaygroundPolicyBundle(DEFAULT_SCENARIO_ID, defaultRule, 0.42);
    const library = savePolicyBundleToLibrary(savePolicyBundleToLibrary(createPolicyLibrary(), bundle, "2026-06-30T00:00:00.000Z"), bundle, "2026-06-30T01:00:00.000Z");
    const exported = exportPolicyBundleLibrary(library);
    const parsed = parsePolicyBundleLibrary(exported);

    expect(library.schemaVersion).toBe(POLICY_LIBRARY_SCHEMA_VERSION);
    expect(library.bundles).toHaveLength(1);
    expect(library.bundles[0]?.savedAt).toBe("2026-06-30T01:00:00.000Z");
    expect(exported).toContain("symtorch.policyLibrary.v1");
    expect(parsed).toEqual(library);
  });

  it("persists policy bundle libraries inside playground state", () => {
    const bundle = createPlaygroundPolicyBundle(DEFAULT_SCENARIO_ID, defaultRule, 0.42);
    const library = savePolicyBundleToLibrary(createPolicyLibrary(), bundle, "2026-06-30T00:00:00.000Z");
    const state = createPlaygroundState(DEFAULT_SCENARIO_ID, defaultRule, defaultCases(), 0.42, defaultTrainingExamples(), null, library);
    const roundTrip = parsePlaygroundState(JSON.stringify(state));

    expect(roundTrip?.policyLibrary).toEqual(library);
  });

  it("rejects tampered policy bundle imports", () => {
    const bundle = createPlaygroundPolicyBundle(DEFAULT_SCENARIO_ID, defaultRule, 0.5);
    const tampered = JSON.stringify({ ...bundle, rules: "escalate(X) :- approved(X)." });
    const parsed = parsePlaygroundPolicyBundle(tampered);

    expect(parsed.ok).toBe(false);
    expect(parsed.diagnostics[0]?.message).toContain("valid hash");
  });

  it("runs workbench decisions through the bundle materialization path", () => {
    const bundle = createPlaygroundPolicyBundle(DEFAULT_SCENARIO_ID, defaultRule, 0.7);
    const bundleAgent = buildPolicyBundleAgent(bundle, defaultCases());
    const loaded = loadPolicyBundle(bundle);
    const directAgent = new RuleAgent(loaded.program, new FuzzyRuleEngine(loaded.registry), 0.5);

    for (const item of defaultCases()) {
      directAgent.memory.observeEntity(item.entityId, {
        high_risk: item.high_risk,
        approved: item.approved
      });
    }

    expect(bundleAgent.decideEntitiesTrace()).toEqual(directAgent.decideEntitiesTrace());
  });

  it("ships a valid golden escalation policy bundle", () => {
    const serialized = readFileSync(new URL("../examples/policies/escalation.policy.json", import.meta.url), "utf8");
    const parsed = parsePlaygroundPolicyBundle(serialized);

    expect(parsed.ok).toBe(true);
    if (!parsed.ok) throw new Error("expected golden policy bundle");
    expect(parsed.bundle.name).toBe("Escalation Policy");
    expect(loadPolicyBundle(parsed.bundle).program.rules).toHaveLength(2);
  });

  it("reports scenario contract diagnostics", () => {
    const invalid = validatePlaygroundScenario({
      schemaVersion: SCENARIO_SCHEMA_VERSION,
      id: "",
      title: "Broken",
      description: "Missing useful pieces.",
      ruleSource: "broken(X) :- missing(X).",
      cases: [{ entityId: "case", high_risk: "bad", approved: 0.2 }],
      trainingExamples: [],
      trainedThreshold: Number.NaN
    });

    expect(invalid.ok).toBe(false);
    expect(invalid.diagnostics.map((item) => item.path)).toContain("$.id");
    expect(invalid.diagnostics.map((item) => item.path)).toContain("$.trainedThreshold");
    expect(invalid.diagnostics.map((item) => item.path)).toContain("$.cases[0]");
    expect(invalid.diagnostics.map((item) => item.path)).toContain("$.trainingExamples");
    expect(invalid.diagnostics.some((item) => item.path === "$.ruleSource" && item.message.includes("missing"))).toBe(true);
  });
});
