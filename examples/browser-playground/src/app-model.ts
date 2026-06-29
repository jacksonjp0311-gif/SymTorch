import { RuleAgent } from "@symtorch/agent";
import { tensor } from "@symtorch/core";
import {
  FactPredicate,
  FuzzyRuleEngine,
  type PredicateRegistry,
  PredicateRegistry as Registry,
  RuleProgram,
  RuleTrainer,
  ThresholdPredicate,
  validateProgram
} from "@symtorch/logic";

export type CaseFacts = {
  entityId: string;
  high_risk: number;
  approved: number;
};

export type TrainingResult = {
  beforeThreshold: number;
  afterThreshold: number;
  beforeScore: number;
  afterScore: number;
  finalLoss: number;
  historyLength: number;
  explanationPredicateCount: number;
  explanationJson: unknown;
};

export type BrowserPlaygroundState = {
  schemaVersion: typeof PLAYGROUND_STATE_VERSION;
  ruleSource: string;
  cases: CaseFacts[];
  trainedThreshold: number;
};

export const PLAYGROUND_STATE_VERSION = "symtorch.playground.v1";

export const defaultRule = `escalate(X) :- high_risk(X), not approved(X).
defer(X) :- approved(X).`;

export function defaultCases(): CaseFacts[] {
  return [
    { entityId: "case-low", high_risk: 0.2, approved: 0.1 },
    { entityId: "case-hot", high_risk: 0.9, approved: 0.1 },
    { entityId: "case-approved", high_risk: 0.9, approved: 0.95 },
    { entityId: "case-borderline", high_risk: 0.55, approved: 0.2 }
  ];
}

export function createFactRegistry(): PredicateRegistry {
  return new Registry()
    .register(new FactPredicate("high_risk"))
    .register(new FactPredicate("approved"));
}

export function buildAgent(program: RuleProgram, cases: readonly CaseFacts[], registry = createFactRegistry()): RuleAgent {
  const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
  for (const item of cases) {
    agent.memory.observeEntity(item.entityId, {
      high_risk: item.high_risk,
      approved: item.approved
    });
  }
  return agent;
}

export function validateRuleSource(source: string, registry = createFactRegistry()): ReturnType<typeof validateProgram> {
  return validateProgram(source, { registry });
}

export function createPlaygroundState(
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number
): BrowserPlaygroundState {
  return {
    schemaVersion: PLAYGROUND_STATE_VERSION,
    ruleSource,
    cases: cases.map((item) => ({ ...item })),
    trainedThreshold
  };
}

export function exportPlaygroundState(
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number
): string {
  return JSON.stringify(createPlaygroundState(ruleSource, cases, trainedThreshold), null, 2);
}

export function parsePlaygroundState(serialized: string | null): BrowserPlaygroundState | null {
  if (!serialized) return null;
  try {
    const value = JSON.parse(serialized) as Partial<BrowserPlaygroundState>;
    if (value.schemaVersion !== PLAYGROUND_STATE_VERSION) return null;
    if (typeof value.ruleSource !== "string") return null;
    if (typeof value.trainedThreshold !== "number" || !Number.isFinite(value.trainedThreshold)) return null;
    if (!Array.isArray(value.cases)) return null;
    const cases = value.cases.map(normalizeCase);
    if (cases.some((item) => item === null)) return null;
    return {
      schemaVersion: PLAYGROUND_STATE_VERSION,
      ruleSource: value.ruleSource,
      cases: cases as CaseFacts[],
      trainedThreshold: value.trainedThreshold
    };
  } catch {
    return null;
  }
}

export function trainHighRiskRule(ruleSource: string, threshold: number): TrainingResult {
  const validation = validateRuleSource(ruleSource);
  if (!validation.ok) {
    throw new Error(validation.diagnostics.map((item) => item.message).join("\n"));
  }

  const program = new RuleProgram(ruleSource);
  const highRisk = new ThresholdPredicate("high_risk", "risk", threshold, 10);
  const trainableRegistry = new Registry()
    .register(highRisk)
    .fixed("approved", (_call, context) => tensor(typeof context.approved === "number" ? context.approved : 0));
  const engine = new FuzzyRuleEngine(trainableRegistry);
  const trainer = new RuleTrainer(engine, program.rules[0]!, trainableRegistry, { learningRate: 0.2 });
  const beforeThreshold = highRisk.threshold.item();
  const beforeScore = trainer.predict({ risk: 0.82, approved: 0.08 }).score.item();
  const result = trainer.fit([
    { risk: 0.2, approved: 0.05, label: 0 },
    { risk: 0.35, approved: 0.15, label: 0 },
    { risk: 0.75, approved: 0.05, label: 1 },
    { risk: 0.9, approved: 0.1, label: 1 },
    { risk: 0.88, approved: 0.95, label: 0 }
  ], { epochs: 100 });
  const prediction = trainer.predict({ risk: 0.82, approved: 0.08 });

  return {
    beforeThreshold,
    afterThreshold: highRisk.threshold.item(),
    beforeScore,
    afterScore: prediction.score.item(),
    finalLoss: result.finalLoss,
    historyLength: result.history.length,
    explanationPredicateCount: prediction.explanation.predicates.length,
    explanationJson: prediction.explanation
  };
}

function normalizeCase(value: unknown): CaseFacts | null {
  if (!isRecord(value)) return null;
  if (typeof value.entityId !== "string") return null;
  if (typeof value.high_risk !== "number" || !Number.isFinite(value.high_risk)) return null;
  if (typeof value.approved !== "number" || !Number.isFinite(value.approved)) return null;
  return {
    entityId: value.entityId,
    high_risk: clamp01(value.high_risk),
    approved: clamp01(value.approved)
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
