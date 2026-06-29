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

export type TrainingExample = {
  risk: number;
  approved: number;
  label: number;
};

export type PlaygroundScenario = {
  id: string;
  title: string;
  description: string;
  ruleSource: string;
  cases: CaseFacts[];
  trainingExamples: TrainingExample[];
  trainedThreshold: number;
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
  scenarioId: string;
  ruleSource: string;
  cases: CaseFacts[];
  trainingExamples: TrainingExample[];
  trainedThreshold: number;
};

export const PLAYGROUND_STATE_VERSION = "symtorch.playground.v1";
export const DEFAULT_SCENARIO_ID = "case-escalation";

export const defaultRule = `escalate(X) :- high_risk(X), not approved(X).
defer(X) :- approved(X).`;

export const playgroundScenarios: readonly PlaygroundScenario[] = [
  {
    id: DEFAULT_SCENARIO_ID,
    title: "Case Escalation",
    description: "Escalate risky unapproved cases while deferring approved ones.",
    ruleSource: defaultRule,
    cases: defaultCases(),
    trainingExamples: defaultTrainingExamples(),
    trainedThreshold: 0.9
  },
  {
    id: "fraud-review",
    title: "Fraud Review",
    description: "Route suspicious unapproved transactions to review and clear approved transactions.",
    ruleSource: `review(X) :- high_risk(X), not approved(X).
clear(X) :- approved(X).`,
    cases: [
      { entityId: "txn-clean", high_risk: 0.15, approved: 0.8 },
      { entityId: "txn-watch", high_risk: 0.62, approved: 0.25 },
      { entityId: "txn-hot", high_risk: 0.94, approved: 0.05 },
      { entityId: "txn-approved", high_risk: 0.8, approved: 0.92 }
    ],
    trainingExamples: [
      { risk: 0.15, approved: 0.8, label: 0 },
      { risk: 0.45, approved: 0.3, label: 0 },
      { risk: 0.72, approved: 0.1, label: 1 },
      { risk: 0.95, approved: 0.05, label: 1 },
      { risk: 0.85, approved: 0.9, label: 0 }
    ],
    trainedThreshold: 0.88
  },
  {
    id: "support-routing",
    title: "Support Routing",
    description: "Send high-risk unresolved support issues to specialist review.",
    ruleSource: `route_specialist(X) :- high_risk(X), not approved(X).
resolve_standard(X) :- approved(X).`,
    cases: [
      { entityId: "ticket-simple", high_risk: 0.18, approved: 0.7 },
      { entityId: "ticket-billing", high_risk: 0.58, approved: 0.2 },
      { entityId: "ticket-outage", high_risk: 0.93, approved: 0.08 },
      { entityId: "ticket-resolved", high_risk: 0.72, approved: 0.9 }
    ],
    trainingExamples: [
      { risk: 0.1, approved: 0.75, label: 0 },
      { risk: 0.35, approved: 0.25, label: 0 },
      { risk: 0.78, approved: 0.08, label: 1 },
      { risk: 0.92, approved: 0.1, label: 1 },
      { risk: 0.8, approved: 0.85, label: 0 }
    ],
    trainedThreshold: 0.86
  }
];

export function defaultScenario(): PlaygroundScenario {
  return cloneScenario(scenarioById(DEFAULT_SCENARIO_ID)!);
}

export function scenarioById(id: string): PlaygroundScenario | null {
  return playgroundScenarios.find((scenario) => scenario.id === id) ?? null;
}

export function defaultCases(): CaseFacts[] {
  return [
    { entityId: "case-low", high_risk: 0.2, approved: 0.1 },
    { entityId: "case-hot", high_risk: 0.9, approved: 0.1 },
    { entityId: "case-approved", high_risk: 0.9, approved: 0.95 },
    { entityId: "case-borderline", high_risk: 0.55, approved: 0.2 }
  ];
}

export function defaultTrainingExamples(): TrainingExample[] {
  return [
    { risk: 0.2, approved: 0.05, label: 0 },
    { risk: 0.35, approved: 0.15, label: 0 },
    { risk: 0.75, approved: 0.05, label: 1 },
    { risk: 0.9, approved: 0.1, label: 1 },
    { risk: 0.88, approved: 0.95, label: 0 }
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
  scenarioId: string,
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number,
  trainingExamples: readonly TrainingExample[] = defaultTrainingExamples()
): BrowserPlaygroundState {
  return {
    schemaVersion: PLAYGROUND_STATE_VERSION,
    scenarioId,
    ruleSource,
    cases: cases.map((item) => ({ ...item })),
    trainingExamples: trainingExamples.map((item) => ({ ...item })),
    trainedThreshold
  };
}

export function exportPlaygroundState(
  scenarioId: string,
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number,
  trainingExamples: readonly TrainingExample[] = defaultTrainingExamples()
): string {
  return JSON.stringify(createPlaygroundState(scenarioId, ruleSource, cases, trainedThreshold, trainingExamples), null, 2);
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
    const trainingExamples = Array.isArray(value.trainingExamples)
      ? value.trainingExamples.map(normalizeTrainingExample)
      : defaultTrainingExamples();
    if (trainingExamples.some((item) => item === null)) return null;
    return {
      schemaVersion: PLAYGROUND_STATE_VERSION,
      scenarioId: typeof value.scenarioId === "string" ? value.scenarioId : DEFAULT_SCENARIO_ID,
      ruleSource: value.ruleSource,
      cases: cases as CaseFacts[],
      trainingExamples: trainingExamples as TrainingExample[],
      trainedThreshold: value.trainedThreshold
    };
  } catch {
    return null;
  }
}

function cloneScenario(scenario: PlaygroundScenario): PlaygroundScenario {
  return {
    ...scenario,
    cases: scenario.cases.map((item) => ({ ...item })),
    trainingExamples: scenario.trainingExamples.map((item) => ({ ...item }))
  };
}

export function trainHighRiskRule(
  ruleSource: string,
  threshold: number,
  examples: readonly TrainingExample[] = defaultTrainingExamples()
): TrainingResult {
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
  const result = trainer.fit(examples.map((example) => ({ ...example })), { epochs: 100 });
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

function normalizeTrainingExample(value: unknown): TrainingExample | null {
  if (!isRecord(value)) return null;
  if (typeof value.risk !== "number" || !Number.isFinite(value.risk)) return null;
  if (typeof value.approved !== "number" || !Number.isFinite(value.approved)) return null;
  if (typeof value.label !== "number" || !Number.isFinite(value.label)) return null;
  return {
    risk: clamp01(value.risk),
    approved: clamp01(value.approved),
    label: value.label >= 0.5 ? 1 : 0
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
