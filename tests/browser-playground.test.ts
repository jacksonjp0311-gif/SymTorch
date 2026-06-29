import { describe, expect, it } from "vitest";
import { RuleProgram } from "@symtorch/logic";
import {
  buildAgent,
  createFactRegistry,
  createPlaygroundState,
  createTrainingRun,
  DEFAULT_SCENARIO_ID,
  defaultCases,
  defaultRule,
  defaultScenario,
  defaultTrainingExamples,
  exportPlaygroundScenario,
  exportPlaygroundState,
  parsePlaygroundState,
  parsePlaygroundScenario,
  playgroundScenarios,
  SCENARIO_SCHEMA_VERSION,
  summarizeTrainingRun,
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

  it("exports readable versioned playground state", () => {
    const exported = exportPlaygroundState(defaultScenario().id, defaultRule, defaultCases(), 0.5, defaultTrainingExamples());
    const parsed = parsePlaygroundState(exported);

    expect(exported).toContain("\n");
    expect(exported).toContain("symtorch.playground.v1");
    expect(parsed?.scenarioId).toBe(DEFAULT_SCENARIO_ID);
    expect(parsed?.ruleSource).toBe(defaultRule);
    expect(parsed?.trainedThreshold).toBe(0.5);
    expect(parsed?.trainingExamples).toHaveLength(5);
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
