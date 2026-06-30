import { createPolicyAgent, RuleAgent, type SerializedEntityDecision } from "@symtorch/agent";
import { tensor } from "@symtorch/core";
import {
  createPolicyBundle,
  FactPredicate,
  FuzzyRuleEngine,
  isSerializedPolicyBundle,
  loadPolicyBundle,
  POLICY_BUNDLE_SCHEMA_VERSION,
  type PredicateRegistry,
  PredicateRegistry as Registry,
  RuleProgram,
  RuleTrainer,
  type SerializedPolicyBundle,
  ThresholdPredicate,
  validateProgram,
  verifyPolicyBundleHash
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
  schemaVersion: typeof SCENARIO_SCHEMA_VERSION;
  id: string;
  title: string;
  description: string;
  ruleSource: string;
  cases: CaseFacts[];
  trainingExamples: TrainingExample[];
  trainedThreshold: number;
};

export type ScenarioValidationDiagnostic = {
  path: string;
  message: string;
};

export type ScenarioValidationResult =
  | { ok: true; scenario: PlaygroundScenario; diagnostics: [] }
  | { ok: false; scenario: null; diagnostics: ScenarioValidationDiagnostic[] };

export type PolicyBundleValidationResult =
  | { ok: true; bundle: SerializedPolicyBundle; diagnostics: [] }
  | { ok: false; bundle: null; diagnostics: ScenarioValidationDiagnostic[] };

export type PolicyHealth = {
  schemaVersion: string;
  hash: string;
  hashVerified: boolean;
  ruleCount: number;
  predicateCount: number;
  lastDecisionStatus: string;
  replayStatus: "PASS" | "FAIL" | "NOT_RUN";
};

export type SavedPolicyBundle = {
  id: string;
  savedAt: string;
  bundle: SerializedPolicyBundle;
};

export type PolicyBundleLibrary = {
  schemaVersion: typeof POLICY_LIBRARY_SCHEMA_VERSION;
  bundles: SavedPolicyBundle[];
};

export type MigrationResult<T> =
  | { ok: true; migrated: boolean; value: T; diagnostics: [] }
  | { ok: false; migrated: false; value: null; diagnostics: ScenarioValidationDiagnostic[] };

export type TrainingResult = {
  beforeThreshold: number;
  afterThreshold: number;
  beforeScore: number;
  afterScore: number;
  finalLoss: number;
  initialLoss: number;
  historyLength: number;
  history: TrainingHistoryItem[];
  explanationPredicateCount: number;
  explanationJson: unknown;
};

export type TrainingHistoryItem = {
  epoch: number;
  loss: number;
};

export type TrainingRun = {
  schemaVersion: typeof TRAINING_RUN_SCHEMA_VERSION;
  scenarioId: string;
  startedThreshold: number;
  finalThreshold: number;
  beforeScore: number;
  afterScore: number;
  initialLoss: number;
  finalLoss: number;
  history: TrainingHistoryItem[];
};

export type BrowserPlaygroundState = {
  schemaVersion: typeof PLAYGROUND_STATE_VERSION;
  scenarioId: string;
  ruleSource: string;
  cases: CaseFacts[];
  trainingExamples: TrainingExample[];
  trainedThreshold: number;
  lastTrainingRun: TrainingRun | null;
  policyLibrary: PolicyBundleLibrary;
};

export const PLAYGROUND_STATE_VERSION = "symtorch.playground.v1";
export const POLICY_LIBRARY_SCHEMA_VERSION = "symtorch.policyLibrary.v1";
export const SCENARIO_SCHEMA_VERSION = "symtorch.scenario.v1";
export const TRAINING_RUN_SCHEMA_VERSION = "symtorch.trainingRun.v1";
export const DEFAULT_SCENARIO_ID = "case-escalation";

export const defaultRule = `escalate(X) :- high_risk(X), not approved(X).
defer(X) :- approved(X).`;

export const playgroundScenarios: readonly PlaygroundScenario[] = [
  {
    schemaVersion: SCENARIO_SCHEMA_VERSION,
    id: DEFAULT_SCENARIO_ID,
    title: "Case Escalation",
    description: "Escalate risky unapproved cases while deferring approved ones.",
    ruleSource: defaultRule,
    cases: defaultCases(),
    trainingExamples: defaultTrainingExamples(),
    trainedThreshold: 0.9
  },
  {
    schemaVersion: SCENARIO_SCHEMA_VERSION,
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
    schemaVersion: SCENARIO_SCHEMA_VERSION,
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

export function exportPlaygroundScenario(scenario: PlaygroundScenario): string {
  return JSON.stringify(cloneScenario(scenario), null, 2);
}

export function parsePlaygroundScenario(serialized: string | null): ScenarioValidationResult {
  if (!serialized) {
    return scenarioError("$", "Expected scenario JSON.");
  }
  try {
    return validatePlaygroundScenario(JSON.parse(serialized));
  } catch {
    return scenarioError("$", "Expected valid JSON.");
  }
}

export function validatePlaygroundScenario(value: unknown): ScenarioValidationResult {
  const diagnostics: ScenarioValidationDiagnostic[] = [];
  if (!isRecord(value)) return scenarioError("$", "Expected an object.");
  if (value.schemaVersion !== SCENARIO_SCHEMA_VERSION) {
    diagnostics.push({ path: "$.schemaVersion", message: `Expected ${SCENARIO_SCHEMA_VERSION}.` });
  }
  const id = readString(value, "id", diagnostics);
  const title = readString(value, "title", diagnostics);
  const description = readString(value, "description", diagnostics);
  const ruleSource = readString(value, "ruleSource", diagnostics);
  const trainedThreshold = readNumber(value, "trainedThreshold", diagnostics);
  const cases = readCases(value.cases, diagnostics);
  const trainingExamples = readTrainingExamples(value.trainingExamples, diagnostics);

  if (ruleSource) {
    const validation = validateRuleSource(ruleSource);
    if (!validation.ok) {
      for (const diagnostic of validation.diagnostics) {
        diagnostics.push({ path: "$.ruleSource", message: diagnostic.message });
      }
    }
  }

  if (diagnostics.length > 0 || !id || !title || !description || !ruleSource || trainedThreshold === null || !cases || !trainingExamples) {
    return { ok: false, scenario: null, diagnostics };
  }

  return {
    ok: true,
    diagnostics: [],
    scenario: {
      schemaVersion: SCENARIO_SCHEMA_VERSION,
      id,
      title,
      description,
      ruleSource,
      cases,
      trainingExamples,
      trainedThreshold: clamp01(trainedThreshold)
    }
  };
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

export function createPlaygroundPolicyBundle(
  scenarioId: string,
  ruleSource: string,
  trainedThreshold: number
): SerializedPolicyBundle {
  const scenario = scenarioById(scenarioId);
  return createPolicyBundle({
    name: scenario?.title ?? "SymTorch Playground Policy",
    version: "playground",
    rules: ruleSource,
    predicates: [
      { kind: "threshold", name: "high_risk", valueKey: "high_risk", threshold: clamp01(trainedThreshold), slope: 10 },
      { kind: "fact", name: "approved" }
    ],
    metadata: {
      scenarioId,
      source: "browser-playground"
    }
  });
}

export function exportPlaygroundPolicyBundle(
  scenarioId: string,
  ruleSource: string,
  trainedThreshold: number
): string {
  return JSON.stringify(createPlaygroundPolicyBundle(scenarioId, ruleSource, trainedThreshold), null, 2);
}

export function parsePlaygroundPolicyBundle(serialized: string | null): PolicyBundleValidationResult {
  if (!serialized) {
    return policyBundleError("$", "Expected policy bundle JSON.");
  }
  try {
    const value = JSON.parse(serialized) as unknown;
    if (!isSerializedPolicyBundle(value)) {
      return policyBundleError("$", `Expected ${POLICY_BUNDLE_SCHEMA_VERSION} bundle with a valid hash.`);
    }
    return { ok: true, bundle: value, diagnostics: [] };
  } catch {
    return policyBundleError("$", "Expected valid JSON.");
  }
}

export function buildPolicyBundleAgent(bundle: SerializedPolicyBundle, cases: readonly CaseFacts[]): RuleAgent {
  const agent = createPolicyAgent(bundle, {
    threshold: 0.5,
    limits: {
      maxRuleSourceLength: 10_000,
      maxEntitiesPerBatch: 100
    }
  });
  for (const item of cases) {
    agent.memory.observeEntity(item.entityId, {
      high_risk: item.high_risk,
      approved: item.approved
    });
  }
  return agent;
}

export function createPolicyHealth(
  bundle: SerializedPolicyBundle,
  lastDecision: SerializedEntityDecision | null = null,
  replayOk: boolean | null = null
): PolicyHealth {
  const hashVerified = verifyPolicyBundleHash(bundle);
  let ruleCount = 0;
  let predicateCount = bundle.predicates.length;
  if (hashVerified) {
    const loaded = loadPolicyBundle(bundle);
    ruleCount = loaded.program.rules.length;
    predicateCount = bundle.predicates.length;
  }
  return {
    schemaVersion: bundle.schemaVersion,
    hash: bundle.hash,
    hashVerified,
    ruleCount,
    predicateCount,
    lastDecisionStatus: lastDecision
      ? `${lastDecision.accepted ? "ACCEPTED" : "REJECTED"} ${lastDecision.action} ${lastDecision.score.toFixed(4)}`
      : "NOT_RUN",
    replayStatus: replayOk === null ? "NOT_RUN" : replayOk ? "PASS" : "FAIL"
  };
}

export function thresholdFromPolicyBundle(bundle: SerializedPolicyBundle): number | null {
  const predicate = bundle.predicates.find((item) => item.kind === "threshold" && item.name === "high_risk");
  return predicate?.kind === "threshold" ? clamp01(predicate.threshold) : null;
}

export function scenarioIdFromPolicyBundle(bundle: SerializedPolicyBundle): string | null {
  const scenarioId = bundle.metadata.scenarioId;
  return typeof scenarioId === "string" && scenarioId.trim() !== "" ? scenarioId : null;
}

export function createPolicyLibrary(bundles: readonly SavedPolicyBundle[] = []): PolicyBundleLibrary {
  return {
    schemaVersion: POLICY_LIBRARY_SCHEMA_VERSION,
    bundles: dedupeSavedBundles(bundles)
  };
}

export function savePolicyBundleToLibrary(
  library: PolicyBundleLibrary,
  bundle: SerializedPolicyBundle,
  savedAt = new Date().toISOString()
): PolicyBundleLibrary {
  const id = policyBundleLibraryId(bundle);
  return createPolicyLibrary([
    { id, savedAt, bundle },
    ...library.bundles.filter((item) => item.id !== id)
  ]);
}

export function exportPolicyBundleLibrary(library: PolicyBundleLibrary): string {
  return JSON.stringify(createPolicyLibrary(library.bundles), null, 2);
}

export function parsePolicyBundleLibrary(serialized: string | null): PolicyBundleLibrary | null {
  if (!serialized) return null;
  try {
    const migration = migratePolicyBundleLibrary(JSON.parse(serialized));
    return migration.ok ? migration.value : null;
  } catch {
    return null;
  }
}

export function migratePolicyBundleLibrary(value: unknown): MigrationResult<PolicyBundleLibrary> {
  const normalized = normalizePolicyBundleLibrary(value);
  if (normalized) {
    return { ok: true, migrated: false, value: normalized, diagnostics: [] };
  }
  if (Array.isArray(value)) {
    const bundles = value.map(normalizeSavedPolicyBundle);
    if (bundles.some((item) => item === null)) {
      return migrationError("$", `Expected ${POLICY_LIBRARY_SCHEMA_VERSION} library or an array of saved policy bundles.`);
    }
    return {
      ok: true,
      migrated: true,
      value: createPolicyLibrary(bundles as SavedPolicyBundle[]),
      diagnostics: []
    };
  }
  return migrationError("$", `Expected ${POLICY_LIBRARY_SCHEMA_VERSION} library.`);
}

export function policyBundleLibraryId(bundle: SerializedPolicyBundle): string {
  return `${bundle.name}:${bundle.version}:${bundle.hash}`;
}

export function validateRuleSource(source: string, registry = createFactRegistry()): ReturnType<typeof validateProgram> {
  return validateProgram(source, { registry });
}

export function createPlaygroundState(
  scenarioId: string,
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number,
  trainingExamples: readonly TrainingExample[] = defaultTrainingExamples(),
  lastTrainingRun: TrainingRun | null = null,
  policyLibrary: PolicyBundleLibrary = createPolicyLibrary()
): BrowserPlaygroundState {
  return {
    schemaVersion: PLAYGROUND_STATE_VERSION,
    scenarioId,
    ruleSource,
    cases: cases.map((item) => ({ ...item })),
    trainingExamples: trainingExamples.map((item) => ({ ...item })),
    trainedThreshold,
    lastTrainingRun: lastTrainingRun ? cloneTrainingRun(lastTrainingRun) : null,
    policyLibrary: createPolicyLibrary(policyLibrary.bundles)
  };
}

export function exportPlaygroundState(
  scenarioId: string,
  ruleSource: string,
  cases: readonly CaseFacts[],
  trainedThreshold: number,
  trainingExamples: readonly TrainingExample[] = defaultTrainingExamples(),
  lastTrainingRun: TrainingRun | null = null,
  policyLibrary: PolicyBundleLibrary = createPolicyLibrary()
): string {
  return JSON.stringify(createPlaygroundState(scenarioId, ruleSource, cases, trainedThreshold, trainingExamples, lastTrainingRun, policyLibrary), null, 2);
}

export function parsePlaygroundState(serialized: string | null): BrowserPlaygroundState | null {
  if (!serialized) return null;
  try {
    const migration = migratePlaygroundState(JSON.parse(serialized));
    return migration.ok ? migration.value : null;
  } catch {
    return null;
  }
}

export function migratePlaygroundState(value: unknown): MigrationResult<BrowserPlaygroundState> {
  if (!isRecord(value)) return migrationError("$", "Expected an object.");
  if (value.schemaVersion !== PLAYGROUND_STATE_VERSION) {
    return migrationError("$.schemaVersion", `Expected ${PLAYGROUND_STATE_VERSION}.`);
  }
  if (typeof value.ruleSource !== "string") return migrationError("$.ruleSource", "Expected a string.");
  if (typeof value.trainedThreshold !== "number" || !Number.isFinite(value.trainedThreshold)) {
    return migrationError("$.trainedThreshold", "Expected a finite number.");
  }
  if (!Array.isArray(value.cases)) return migrationError("$.cases", "Expected an array.");
  const cases = value.cases.map(normalizeCase);
  if (cases.some((item) => item === null)) return migrationError("$.cases", "Expected entityId, high_risk, and approved.");
  const trainingExamples = Array.isArray(value.trainingExamples)
    ? value.trainingExamples.map(normalizeTrainingExample)
    : defaultTrainingExamples();
  if (trainingExamples.some((item) => item === null)) {
    return migrationError("$.trainingExamples", "Expected risk, approved, and label.");
  }
  const lastTrainingRun = value.lastTrainingRun === null || value.lastTrainingRun === undefined
    ? null
    : normalizeTrainingRun(value.lastTrainingRun);
  if (value.lastTrainingRun !== null && value.lastTrainingRun !== undefined && !lastTrainingRun) {
    return migrationError("$.lastTrainingRun", `Expected ${TRAINING_RUN_SCHEMA_VERSION}.`);
  }
  const policyLibraryMigration = value.policyLibrary === undefined
    ? { ok: true as const, migrated: true, value: createPolicyLibrary(), diagnostics: [] as [] }
    : migratePolicyBundleLibrary(value.policyLibrary);
  if (!policyLibraryMigration.ok) {
    return migrationError("$.policyLibrary", policyLibraryMigration.diagnostics[0]?.message ?? `Expected ${POLICY_LIBRARY_SCHEMA_VERSION}.`);
  }
  return {
    ok: true,
    migrated: policyLibraryMigration.migrated || value.trainingExamples === undefined,
    diagnostics: [],
    value: {
      schemaVersion: PLAYGROUND_STATE_VERSION,
      scenarioId: typeof value.scenarioId === "string" ? value.scenarioId : DEFAULT_SCENARIO_ID,
      ruleSource: value.ruleSource,
      cases: cases as CaseFacts[],
      trainingExamples: trainingExamples as TrainingExample[],
      trainedThreshold: value.trainedThreshold,
      lastTrainingRun,
      policyLibrary: policyLibraryMigration.value
    }
  };
}

function cloneScenario(scenario: PlaygroundScenario): PlaygroundScenario {
  return {
    ...scenario,
    cases: scenario.cases.map((item) => ({ ...item })),
    trainingExamples: scenario.trainingExamples.map((item) => ({ ...item }))
  };
}

function scenarioError(path: string, message: string): ScenarioValidationResult {
  return {
    ok: false,
    scenario: null,
    diagnostics: [{ path, message }]
  };
}

function policyBundleError(path: string, message: string): PolicyBundleValidationResult {
  return {
    ok: false,
    bundle: null,
    diagnostics: [{ path, message }]
  };
}

function migrationError<T>(path: string, message: string): MigrationResult<T> {
  return {
    ok: false,
    migrated: false,
    value: null,
    diagnostics: [{ path, message }]
  };
}

function readString(value: Record<string, unknown>, key: string, diagnostics: ScenarioValidationDiagnostic[]): string | null {
  const item = value[key];
  if (typeof item !== "string" || item.trim() === "") {
    diagnostics.push({ path: `$.${key}`, message: "Expected a non-empty string." });
    return null;
  }
  return item;
}

function readNumber(value: Record<string, unknown>, key: string, diagnostics: ScenarioValidationDiagnostic[]): number | null {
  const item = value[key];
  if (typeof item !== "number" || !Number.isFinite(item)) {
    diagnostics.push({ path: `$.${key}`, message: "Expected a finite number." });
    return null;
  }
  return item;
}

function readCases(value: unknown, diagnostics: ScenarioValidationDiagnostic[]): CaseFacts[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    diagnostics.push({ path: "$.cases", message: "Expected at least one case." });
    return null;
  }
  const cases = value.map(normalizeCase);
  cases.forEach((item, index) => {
    if (!item) diagnostics.push({ path: `$.cases[${index}]`, message: "Expected entityId, high_risk, and approved." });
  });
  return cases.some((item) => item === null) ? null : cases as CaseFacts[];
}

function readTrainingExamples(value: unknown, diagnostics: ScenarioValidationDiagnostic[]): TrainingExample[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    diagnostics.push({ path: "$.trainingExamples", message: "Expected at least one training example." });
    return null;
  }
  const examples = value.map(normalizeTrainingExample);
  examples.forEach((item, index) => {
    if (!item) diagnostics.push({ path: `$.trainingExamples[${index}]`, message: "Expected risk, approved, and label." });
  });
  return examples.some((item) => item === null) ? null : examples as TrainingExample[];
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
    initialLoss: result.history[0]?.loss ?? Number.NaN,
    finalLoss: result.finalLoss,
    historyLength: result.history.length,
    history: result.history.map((item) => ({ ...item })),
    explanationPredicateCount: prediction.explanation.predicates.length,
    explanationJson: prediction.explanation
  };
}

export function createTrainingRun(scenarioId: string, result: TrainingResult): TrainingRun {
  return {
    schemaVersion: TRAINING_RUN_SCHEMA_VERSION,
    scenarioId,
    startedThreshold: result.beforeThreshold,
    finalThreshold: result.afterThreshold,
    beforeScore: result.beforeScore,
    afterScore: result.afterScore,
    initialLoss: result.initialLoss,
    finalLoss: result.finalLoss,
    history: result.history.map((item) => ({ ...item }))
  };
}

export function summarizeTrainingRun(run: TrainingRun | null): string {
  if (!run) return "Not trained yet.";
  const lossDelta = run.initialLoss - run.finalLoss;
  return [
    `threshold: ${run.startedThreshold.toFixed(4)} -> ${run.finalThreshold.toFixed(4)}`,
    `score: ${run.beforeScore.toFixed(4)} -> ${run.afterScore.toFixed(4)}`,
    `loss: ${run.initialLoss.toFixed(4)} -> ${run.finalLoss.toFixed(4)} (${lossDelta >= 0 ? "-" : "+"}${Math.abs(lossDelta).toFixed(4)})`,
    `epochs: ${run.history.length}`
  ].join("\n");
}

function cloneTrainingRun(run: TrainingRun): TrainingRun {
  return {
    ...run,
    history: run.history.map((item) => ({ ...item }))
  };
}

function normalizePolicyBundleLibrary(value: unknown): PolicyBundleLibrary | null {
  if (!isRecord(value)) return null;
  if (value.schemaVersion !== POLICY_LIBRARY_SCHEMA_VERSION) return null;
  if (!Array.isArray(value.bundles)) return null;
  const bundles = value.bundles.map(normalizeSavedPolicyBundle);
  if (bundles.some((item) => item === null)) return null;
  return createPolicyLibrary(bundles as SavedPolicyBundle[]);
}

function normalizeSavedPolicyBundle(value: unknown): SavedPolicyBundle | null {
  if (!isRecord(value)) return null;
  if (typeof value.id !== "string" || typeof value.savedAt !== "string") return null;
  if (!isSerializedPolicyBundle(value.bundle)) return null;
  return {
    id: policyBundleLibraryId(value.bundle),
    savedAt: value.savedAt,
    bundle: value.bundle
  };
}

function dedupeSavedBundles(bundles: readonly SavedPolicyBundle[]): SavedPolicyBundle[] {
  const seen = new Set<string>();
  const result: SavedPolicyBundle[] = [];
  for (const item of bundles) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    result.push({
      id: item.id,
      savedAt: item.savedAt,
      bundle: item.bundle
    });
  }
  return result;
}

function normalizeTrainingRun(value: unknown): TrainingRun | null {
  if (!isRecord(value)) return null;
  if (value.schemaVersion !== TRAINING_RUN_SCHEMA_VERSION) return null;
  if (typeof value.scenarioId !== "string") return null;
  const startedThreshold = finiteNumber(value.startedThreshold);
  const finalThreshold = finiteNumber(value.finalThreshold);
  const beforeScore = finiteNumber(value.beforeScore);
  const afterScore = finiteNumber(value.afterScore);
  const initialLoss = finiteNumber(value.initialLoss);
  const finalLoss = finiteNumber(value.finalLoss);
  if (
    startedThreshold === null ||
    finalThreshold === null ||
    beforeScore === null ||
    afterScore === null ||
    initialLoss === null ||
    finalLoss === null ||
    !Array.isArray(value.history)
  ) return null;
  const history = value.history.map(normalizeTrainingHistoryItem);
  if (history.some((item) => item === null)) return null;
  return {
    schemaVersion: TRAINING_RUN_SCHEMA_VERSION,
    scenarioId: value.scenarioId,
    startedThreshold,
    finalThreshold,
    beforeScore,
    afterScore,
    initialLoss,
    finalLoss,
    history: history as TrainingHistoryItem[]
  };
}

function normalizeTrainingHistoryItem(value: unknown): TrainingHistoryItem | null {
  if (!isRecord(value)) return null;
  const epoch = finiteNumber(value.epoch);
  const loss = finiteNumber(value.loss);
  if (epoch === null || loss === null) return null;
  return { epoch, loss };
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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
