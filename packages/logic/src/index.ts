import { add, matmul, mul, ResourceLimitError, sigmoid, sub, SymTorchError, Tensor, tensor } from "@symtorch/core";
import { mseLoss, Optimizer, Parameter, SGD } from "@symtorch/nn";

export type Term = {
  kind: "variable" | "constant";
  name: string;
};

export type PredicateCall = {
  name: string;
  terms: readonly Term[];
  negated: boolean;
};

export type RuleAst = {
  head: PredicateCall;
  body: readonly PredicateCall[];
  source: string;
};

export class RuleParseError extends Error {
  readonly line: number;
  readonly column: number;
  readonly snippet: string;

  constructor(reason: string, readonly source: string, readonly index: number) {
    const location = locateSource(source, index);
    super(`Rule parse error at line ${location.line}, column ${location.column}: ${reason}\n${location.snippet}\n${" ".repeat(location.column - 1)}^`);
    this.name = "RuleParseError";
    this.line = location.line;
    this.column = location.column;
    this.snippet = location.snippet;
  }
}

export class RuleValidationError extends SymTorchError {
  constructor(message: string) {
    super("ERR_RULE_VALIDATION", message);
    this.name = "RuleValidationError";
  }
}

export class PredicateEvaluationError extends SymTorchError {
  constructor(readonly predicate: string, message: string) {
    super("ERR_PREDICATE", `Predicate "${predicate}" failed: ${message}`);
    this.name = "PredicateEvaluationError";
  }
}

export type RuleValidationResult =
  | { ok: true; rules: RuleAst[]; diagnostics: RuleDiagnostic[] }
  | { ok: false; rules: RuleAst[]; diagnostics: RuleDiagnostic[]; error: RuleParseError | RuleDiagnostic };

export type RuleDiagnostic = {
  code: "parse_error" | "missing_predicate";
  message: string;
  severity: "error";
  line?: number;
  column?: number;
  snippet?: string;
  predicate?: string;
};

export type BatchRuleValidationItem = {
  id: string;
  source: string;
  result: RuleValidationResult;
};

export type RuleValidationOptions = {
  registry?: PredicateRegistry;
  limits?: LogicRuntimeLimits;
};

export type RuleValidationInput = string | {
  id: string;
  source: string;
};

export type PredicateContext = Record<string, unknown>;
export type PredicateResolver = (call: PredicateCall, context: PredicateContext) => Tensor;

export type PredicateResolution = {
  score: Tensor;
  kind: "fixed" | "learnable";
  detail?: Record<string, number | string>;
};

export const EXPLANATION_SCHEMA_VERSION = "symtorch.explanation.v1" as const;
export type ExplanationSchemaVersion = typeof EXPLANATION_SCHEMA_VERSION;

export interface Predicate {
  readonly name: string;
  readonly kind: "fixed" | "learnable";
  evaluate(call: PredicateCall, context: PredicateContext): Tensor;
  parameters(): Parameter[];
  describe?(): Record<string, number | string>;
}

export type RuleExplanation = {
  rule: string;
  head: string;
  score: number;
  predicates: PredicateTrace[];
};

export type PredicateTrace = {
  name: string;
  negated: boolean;
  value: number;
  contribution: number;
  kind?: "fixed" | "learnable";
  detail?: Record<string, number | string>;
};

export type RuleResult = {
  score: Tensor;
  explanation: RuleExplanation;
};

export type AggregatedRuleExplanation = {
  head: string;
  score: number;
  ruleCount: number;
  rules: RuleExplanation[];
};

export type AggregatedRuleResult = {
  head: string;
  score: Tensor;
  explanation: AggregatedRuleExplanation;
};

export type SerializedPredicateTrace = {
  name: string;
  negated: boolean;
  value: number;
  contribution: number;
  kind?: "fixed" | "learnable";
  detail?: Record<string, number | string>;
};

export type SerializedRuleExplanation = {
  schemaVersion: ExplanationSchemaVersion;
  type: "rule";
  rule: string;
  head: string;
  score: number;
  predicates: SerializedPredicateTrace[];
};

export type SerializedAggregatedRuleExplanation = {
  schemaVersion: ExplanationSchemaVersion;
  type: "aggregate";
  head: string;
  score: number;
  ruleCount: number;
  rules: SerializedRuleExplanation[];
};

export type SerializedExplanation = SerializedRuleExplanation | SerializedAggregatedRuleExplanation;

export type EntityRuleResult = {
  entityId: string;
  results: AggregatedRuleResult[];
};

export type RankedEntityResult = {
  entityId: string;
  result: AggregatedRuleResult;
};

export type RuleEvaluationEvent = {
  kind: "rule.evaluate";
  rule: string;
  head: string;
  score: number;
  predicateCount: number;
  contextKeys: string[];
  durationMs: number;
};

export type ProgramEvaluationEvent = {
  kind: "program.evaluate";
  ruleCount: number;
  groupCount: number;
  contextKeys: string[];
  durationMs: number;
};

export type LogicObserver = {
  onRuleEvaluate?(event: RuleEvaluationEvent): void;
  onProgramEvaluate?(event: ProgramEvaluationEvent): void;
};

export type FuzzyRuleEngineOptions = {
  observer?: LogicObserver;
  limits?: LogicRuntimeLimits;
};

export type LogicRuntimeLimits = {
  maxRuleSourceLength?: number;
  maxRules?: number;
  maxPredicatesPerRule?: number;
  maxEntitiesPerEvaluation?: number;
};

export const DOMAIN_CONTRACT_SCHEMA_VERSION = "symtorch.domainContract.v1" as const;
export type DomainContractSchemaVersion = typeof DOMAIN_CONTRACT_SCHEMA_VERSION;
export type DomainFieldType = "number" | "string" | "boolean";

export type DomainField = {
  type: DomainFieldType;
  required?: boolean;
  min?: number;
  max?: number;
};

export type DomainEntity = {
  fields: Record<string, DomainField>;
};

export type DomainContract = {
  schemaVersion: DomainContractSchemaVersion;
  entities: Record<string, DomainEntity>;
};

export type DomainValidationDiagnostic = {
  path: string;
  message: string;
};

export type DomainValidationResult =
  | { ok: true; diagnostics: [] }
  | { ok: false; diagnostics: DomainValidationDiagnostic[] };

export const POLICY_BUNDLE_SCHEMA_VERSION = "symtorch.policyBundle.v1" as const;
export type PolicyBundleSchemaVersion = typeof POLICY_BUNDLE_SCHEMA_VERSION;
export const POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION = "symtorch.policyBundleSignature.v1" as const;
export type PolicyBundleSignatureSchemaVersion = typeof POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION;
export const PRODUCTION_READINESS_SCHEMA_VERSION = "symtorch.productionReadiness.v1" as const;
export type ProductionReadinessSchemaVersion = typeof PRODUCTION_READINESS_SCHEMA_VERSION;

export type PolicyBundlePredicate =
  | { kind: "fact"; name: string; key?: string }
  | { kind: "threshold"; name: string; valueKey: string; threshold: number; slope: number }
  | { kind: "linear"; name: string; featureKey: string; featureCount: number; weights: number[]; bias: number };

export type SerializedPolicyBundle = {
  schemaVersion: PolicyBundleSchemaVersion;
  name: string;
  version: string;
  rules: string;
  predicates: PolicyBundlePredicate[];
  metadata: Record<string, string | number | boolean>;
  hash: string;
};

export type PolicyBundleSignature = {
  schemaVersion: PolicyBundleSignatureSchemaVersion;
  algorithm: "symtorch-dev-fnv1a32";
  keyId: string;
  signature: string;
};

export type SignedPolicyBundle = SerializedPolicyBundle & {
  signature: PolicyBundleSignature;
};

export type SignedPolicyBundleVerificationResult =
  | { ok: true; keyId: string; algorithm: PolicyBundleSignature["algorithm"] }
  | { ok: false; reason: "invalid_bundle" | "missing_signature" | "invalid_signature_schema" | "unknown_key" | "signature_mismatch" };

export type ProductionTrackId =
  | "typed_domains"
  | "bundle_signing"
  | "durable_persistence"
  | "trace_snapshots"
  | "runtime_limits"
  | "error_taxonomy"
  | "cpu_gpu_parity"
  | "api_stability"
  | "security_model"
  | "real_apps";

export type ProductionTrackStatus = "implemented" | "alpha" | "planned";

export type ProductionReadinessTrack = {
  id: ProductionTrackId;
  status: ProductionTrackStatus;
  evidence: string[];
  remaining: string[];
};

export type ProductionReadinessReport = {
  schemaVersion: ProductionReadinessSchemaVersion;
  version: string;
  tracks: ProductionReadinessTrack[];
  productionReady: false;
};

export type PolicyBundleSecurityAssessment = {
  schemaVersion: ProductionReadinessSchemaVersion;
  ok: boolean;
  diagnostics: {
    code: "invalid_hash" | "invalid_signature" | "untrusted_key" | "rule_validation" | "runtime_limit" | "security_boundary";
    message: string;
  }[];
};

export type PolicyBundleInput = Omit<SerializedPolicyBundle, "schemaVersion" | "hash">;

export type LoadedPolicyBundle = {
  bundle: SerializedPolicyBundle;
  program: RuleProgram;
  registry: PredicateRegistry;
  engine: FuzzyRuleEngine;
};

export type LoadPolicyBundleOptions = {
  limits?: LogicRuntimeLimits;
  observer?: LogicObserver;
};

export type LabeledRuleExample = PredicateContext & {
  label: number;
};

export type RuleTrainerOptions = {
  learningRate?: number;
  epochs?: number;
  optimizer?: Optimizer;
};

export type RuleTrainerHistoryItem = {
  epoch: number;
  loss: number;
};

export type RuleTrainerResult = {
  history: RuleTrainerHistoryItem[];
  finalLoss: number;
};

export class RuleProgram {
  readonly rules: readonly RuleAst[];

  constructor(source: string, options: { limits?: LogicRuntimeLimits } = {}) {
    this.rules = parseProgram(source, options);
  }
}

export class FactStore {
  private readonly facts = new Map<string, unknown>();

  constructor(initial?: PredicateContext) {
    if (initial) this.observe(initial);
  }

  set(key: string, value: unknown): this {
    this.facts.set(key, value);
    return this;
  }

  get<T = unknown>(key: string): T | undefined {
    return this.facts.get(key) as T | undefined;
  }

  observe(facts: PredicateContext): this {
    for (const [key, value] of Object.entries(facts)) this.set(key, value);
    return this;
  }

  setEntity(entityId: string, facts: PredicateContext): this {
    for (const [key, value] of Object.entries(facts)) this.set(`${entityId}.${key}`, value);
    return this;
  }

  entityIds(): string[] {
    const ids = new Set<string>();
    for (const key of this.facts.keys()) {
      const dot = key.indexOf(".");
      if (dot > 0) ids.add(key.slice(0, dot));
    }
    return Array.from(ids).sort();
  }

  context(extra: PredicateContext = {}): PredicateContext {
    return { ...Object.fromEntries(this.facts.entries()), ...extra };
  }

  entityContext(entityId: string, extra: PredicateContext = {}): PredicateContext {
    const prefix = `${entityId}.`;
    const scoped: PredicateContext = { entity: entityId };
    for (const [key, value] of this.facts.entries()) {
      if (key.startsWith(prefix)) scoped[key.slice(prefix.length)] = value;
    }
    return { ...scoped, ...extra };
  }

  clear(): void {
    this.facts.clear();
  }
}

export class FuzzyRuleEngine {
  constructor(
    private readonly resolver: PredicateResolver | PredicateRegistry,
    private readonly options: FuzzyRuleEngineOptions = {}
  ) {}

  evaluate(rule: RuleAst, context: PredicateContext = {}): RuleResult {
    enforcePredicatesPerRuleLimit(rule, this.options.limits);
    const startedAt = nowMs();
    let score = tensor(1);
    const traces: PredicateTrace[] = [];
    for (const call of rule.body) {
      const resolved = this.resolve(call, context);
      const raw = resolved.score;
      const value = call.negated ? sub(1, raw) : raw;
      score = mul(score, value);
      const trace: PredicateTrace = {
        name: formatPredicate(call),
        negated: call.negated,
        value: raw.item(),
        contribution: value.item(),
        kind: resolved.kind
      };
      if (resolved.detail) trace.detail = resolved.detail;
      traces.push(trace);
    }
    const result = {
      score,
      explanation: {
        rule: rule.source,
        head: formatPredicate(rule.head),
        score: score.item(),
        predicates: traces
      }
    };
    this.options.observer?.onRuleEvaluate?.({
      kind: "rule.evaluate",
      rule: result.explanation.rule,
      head: result.explanation.head,
      score: result.explanation.score,
      predicateCount: traces.length,
      contextKeys: stableKeys(context),
      durationMs: nowMs() - startedAt
    });
    return result;
  }

  evaluateProgram(program: RuleProgram, context: PredicateContext = {}): RuleResult[] {
    return program.rules.map((rule) => this.evaluate(rule, context));
  }

  evaluateProgramGrouped(program: RuleProgram, context: PredicateContext = {}): AggregatedRuleResult[] {
    const startedAt = nowMs();
    const groups = new Map<string, RuleResult[]>();
    for (const rule of program.rules) {
      const result = this.evaluate(rule, context);
      const group = groups.get(result.explanation.head) ?? [];
      group.push(result);
      groups.set(result.explanation.head, group);
    }

    const aggregated = Array.from(groups.entries()).map(([head, results]) => {
      const score = results.reduce((acc, result) => probabilisticOr(acc, result.score), tensor(0));
      return {
        head,
        score,
        explanation: {
          head,
          score: score.item(),
          ruleCount: results.length,
          rules: results.map((result) => result.explanation)
        }
      };
    });
    this.options.observer?.onProgramEvaluate?.({
      kind: "program.evaluate",
      ruleCount: program.rules.length,
      groupCount: aggregated.length,
      contextKeys: stableKeys(context),
      durationMs: nowMs() - startedAt
    });
    return aggregated;
  }

  evaluateEntities(program: RuleProgram, facts: FactStore, entityIds = facts.entityIds()): EntityRuleResult[] {
    enforceEntityEvaluationLimit(entityIds.length, this.options.limits);
    return entityIds.map((entityId) => ({
      entityId,
      results: this.evaluateProgramGrouped(program, facts.entityContext(entityId))
    }));
  }

  rankEntitiesByHead(program: RuleProgram, facts: FactStore, head: string, entityIds = facts.entityIds()): RankedEntityResult[] {
    return this.evaluateEntities(program, facts, entityIds)
      .map((entity) => {
        const result = entity.results.find((candidate) => candidate.head === head);
        return result ? { entityId: entity.entityId, result } : null;
      })
      .filter((item): item is RankedEntityResult => item !== null)
      .sort((a, b) => b.result.score.item() - a.result.score.item());
  }

  private resolve(call: PredicateCall, context: PredicateContext): PredicateResolution {
    if (this.resolver instanceof PredicateRegistry) return this.resolver.resolve(call, context);
    return { score: this.resolver(call, context), kind: "fixed" };
  }
}

export class PredicateRegistry {
  private readonly predicates = new Map<string, Predicate>();

  register(predicate: Predicate): this {
    this.predicates.set(predicate.name, predicate);
    return this;
  }

  fixed(name: string, resolver: PredicateResolver): this {
    return this.register(new FixedPredicate(name, resolver));
  }

  has(name: string): boolean {
    return this.predicates.has(name);
  }

  names(): string[] {
    return Array.from(this.predicates.keys()).sort();
  }

  resolve(call: PredicateCall, context: PredicateContext): PredicateResolution {
    const predicate = this.predicates.get(call.name);
    if (!predicate) throw new PredicateEvaluationError(call.name, "not registered.");
    const resolution: PredicateResolution = {
      score: evaluatePredicate(predicate, call, context),
      kind: predicate.kind
    };
    const detail = predicate.describe?.();
    if (detail) resolution.detail = detail;
    return resolution;
  }

  parameters(): Parameter[] {
    return Array.from(this.predicates.values()).flatMap((predicate) => predicate.parameters());
  }
}

export class FixedPredicate implements Predicate {
  readonly kind = "fixed";

  constructor(readonly name: string, private readonly resolver: PredicateResolver) {}

  evaluate(call: PredicateCall, context: PredicateContext): Tensor {
    return this.resolver(call, context);
  }

  parameters(): Parameter[] {
    return [];
  }
}

export class FactPredicate implements Predicate {
  readonly kind = "fixed";

  constructor(readonly name: string, readonly key = name) {}

  evaluate(_call: PredicateCall, context: PredicateContext): Tensor {
    const value = context[this.key];
    if (value instanceof Tensor) return value;
    if (typeof value !== "number") throw new Error(`FactPredicate "${this.name}" expected numeric context key "${this.key}".`);
    return tensor(value);
  }

  parameters(): Parameter[] {
    return [];
  }

  describe(): Record<string, number | string> {
    return { key: this.key };
  }
}

export class ThresholdPredicate implements Predicate {
  readonly kind = "learnable";
  readonly threshold: Parameter;

  constructor(
    readonly name: string,
    readonly valueKey: string,
    initialThreshold = 0.5,
    readonly slope = 8
  ) {
    this.threshold = new Parameter([initialThreshold], []);
  }

  evaluate(_call: PredicateCall, context: PredicateContext): Tensor {
    const value = readScalar(context, this.valueKey);
    return sigmoid(mul(sub(tensor(value), this.threshold), this.slope));
  }

  parameters(): Parameter[] {
    return [this.threshold];
  }

  describe(): Record<string, number | string> {
    return {
      valueKey: this.valueKey,
      threshold: this.threshold.item(),
      slope: this.slope
    };
  }
}

export class LinearPredicate implements Predicate {
  readonly kind = "learnable";
  readonly weight: Parameter;
  readonly bias: Parameter;

  constructor(
    readonly name: string,
    readonly featureKey: string,
    readonly featureCount: number,
    initialWeights?: readonly number[],
    initialBias = 0
  ) {
    if (initialWeights && initialWeights.length !== featureCount) {
      throw new Error(`LinearPredicate expected ${featureCount} initial weights, got ${initialWeights.length}.`);
    }
    this.weight = new Parameter(initialWeights ?? Array.from({ length: featureCount }, () => 0), [featureCount, 1]);
    this.bias = new Parameter([initialBias], []);
  }

  evaluate(_call: PredicateCall, context: PredicateContext): Tensor {
    const features = readFeatureTensor(context, this.featureKey, this.featureCount);
    return sigmoid(add(matmul(features, this.weight), this.bias));
  }

  parameters(): Parameter[] {
    return [this.weight, this.bias];
  }

  describe(): Record<string, number | string> {
    return {
      featureKey: this.featureKey,
      featureCount: this.featureCount,
      bias: this.bias.item()
    };
  }
}

export class RuleTrainer {
  private readonly optimizer: Optimizer;

  constructor(
    readonly engine: FuzzyRuleEngine,
    readonly rule: RuleAst,
    readonly registry: PredicateRegistry,
    options: RuleTrainerOptions = {}
  ) {
    this.optimizer = options.optimizer ?? new SGD(registry.parameters(), options.learningRate ?? 0.1);
  }

  fit(examples: readonly LabeledRuleExample[], options: RuleTrainerOptions = {}): RuleTrainerResult {
    if (examples.length === 0) throw new Error("RuleTrainer.fit requires at least one example.");
    const epochs = options.epochs ?? 50;
    const history: RuleTrainerHistoryItem[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      for (const example of examples) {
        this.optimizer.zeroGrad();
        const result = this.engine.evaluate(this.rule, example);
        const loss = mseLoss(result.score, tensor(example.label));
        totalLoss += loss.item();
        loss.backward();
        this.optimizer.step();
      }
      history.push({ epoch, loss: totalLoss / examples.length });
    }

    return {
      history,
      finalLoss: history[history.length - 1]?.loss ?? Number.NaN
    };
  }

  predict(context: PredicateContext): RuleResult {
    return this.engine.evaluate(this.rule, context);
  }
}

export function renderRuleExplanation(explanation: RuleExplanation): string {
  const lines = [
    `${explanation.head}: ${formatScore(explanation.score)}`,
    `rule: ${explanation.rule}`
  ];
  for (const predicate of explanation.predicates) {
    const kind = predicate.kind ? ` ${predicate.kind}` : "";
    const detail = predicate.detail ? ` ${formatDetail(predicate.detail)}` : "";
    lines.push(
      `- ${predicate.name}${kind}: value=${formatScore(predicate.value)} contribution=${formatScore(predicate.contribution)}${detail}`
    );
  }
  return lines.join("\n");
}

export function renderAggregatedExplanation(explanation: AggregatedRuleExplanation): string {
  const lines = [
    `${explanation.head}: ${formatScore(explanation.score)} from ${explanation.ruleCount} rule${explanation.ruleCount === 1 ? "" : "s"}`
  ];
  for (const rule of explanation.rules) {
    lines.push(indent(renderRuleExplanation(rule), "  "));
  }
  return lines.join("\n");
}

export function decisionCard(result: RuleResult | AggregatedRuleResult): string {
  if (isAggregatedExplanation(result.explanation)) return renderAggregatedExplanation(result.explanation);
  return renderRuleExplanation(result.explanation);
}

export function decisionTrace(result: RuleResult): SerializedRuleExplanation;
export function decisionTrace(result: AggregatedRuleResult): SerializedAggregatedRuleExplanation;
export function decisionTrace(result: RuleResult | AggregatedRuleResult): SerializedExplanation {
  return serializeExplanation(result.explanation);
}

export function serializeExplanation(explanation: RuleExplanation): SerializedRuleExplanation;
export function serializeExplanation(explanation: AggregatedRuleExplanation): SerializedAggregatedRuleExplanation;
export function serializeExplanation(explanation: RuleExplanation | AggregatedRuleExplanation): SerializedExplanation;
export function serializeExplanation(explanation: RuleExplanation | AggregatedRuleExplanation): SerializedExplanation {
  if (isAggregatedExplanation(explanation)) {
    return {
      schemaVersion: EXPLANATION_SCHEMA_VERSION,
      type: "aggregate",
      head: explanation.head,
      score: explanation.score,
      ruleCount: explanation.ruleCount,
      rules: explanation.rules.map((rule) => serializeRuleExplanation(rule))
    };
  }
  return serializeRuleExplanation(explanation);
}

export function parseProgram(source: string, options: { limits?: LogicRuntimeLimits } = {}): RuleAst[] {
  enforceRuleSourceLimit(source, options.limits);
  const rules = splitRuleSources(source).map((rule) => parseRuleAt(rule.text, rule.start, source));
  enforceRuleCountLimit(rules.length, options.limits);
  for (const rule of rules) enforcePredicatesPerRuleLimit(rule, options.limits);
  return rules;
}

export function validateProgram(source: string, options: RuleValidationOptions = {}): RuleValidationResult {
  try {
    const rules = parseProgram(source, options.limits ? { limits: options.limits } : {});
    const diagnostics = options.registry ? validatePredicateBindings(rules, options.registry) : [];
    if (diagnostics.length > 0) return { ok: false, rules, diagnostics, error: diagnostics[0]! };
    return { ok: true, rules, diagnostics };
  } catch (error) {
    if (error instanceof RuleParseError) {
      const diagnostic = diagnosticFromParseError(error);
      return { ok: false, rules: [], diagnostics: [diagnostic], error };
    }
    throw error;
  }
}

export function validatePrograms(inputs: readonly RuleValidationInput[], options: RuleValidationOptions = {}): BatchRuleValidationItem[] {
  return inputs.map((input, index) => {
    const item = typeof input === "string" ? { id: `rule-${index + 1}`, source: input } : input;
    return {
      id: item.id,
      source: item.source,
      result: validateProgram(item.source, options)
    };
  });
}

export function parseRule(source: string): RuleAst {
  return parseRuleAt(source, 0, source);
}

export function createPolicyBundle(input: PolicyBundleInput): SerializedPolicyBundle {
  const bundleWithoutHash = {
    schemaVersion: POLICY_BUNDLE_SCHEMA_VERSION,
    name: input.name,
    version: input.version,
    rules: input.rules,
    predicates: [...input.predicates].sort((left, right) => left.name.localeCompare(right.name)),
    metadata: stableMetadata(input.metadata)
  };
  return {
    ...bundleWithoutHash,
    hash: stableHash(stableStringify(bundleWithoutHash))
  };
}

export function isSerializedPolicyBundle(value: unknown): value is SerializedPolicyBundle {
  if (!isRecord(value)) return false;
  if (value.schemaVersion !== POLICY_BUNDLE_SCHEMA_VERSION) return false;
  if (typeof value.name !== "string" || typeof value.version !== "string" || typeof value.rules !== "string") return false;
  if (!Array.isArray(value.predicates) || !value.predicates.every(isPolicyBundlePredicate)) return false;
  if (!isRecord(value.metadata) || !Object.values(value.metadata).every(isJsonPrimitive)) return false;
  if (typeof value.hash !== "string") return false;
  return verifyPolicyBundleHash({
    schemaVersion: value.schemaVersion,
    name: value.name,
    version: value.version,
    rules: value.rules,
    predicates: value.predicates,
    metadata: value.metadata as Record<string, string | number | boolean>,
    hash: value.hash
  });
}

export function verifyPolicyBundleHash(bundle: SerializedPolicyBundle): boolean {
  const { hash: _hash, ...withoutHash } = bundle;
  return stableHash(stableStringify({
    ...withoutHash,
    predicates: [...bundle.predicates].sort((left, right) => left.name.localeCompare(right.name)),
    metadata: stableMetadata(bundle.metadata)
  })) === bundle.hash;
}

export function createDomainContract(entities: Record<string, DomainEntity>): DomainContract {
  return {
    schemaVersion: DOMAIN_CONTRACT_SCHEMA_VERSION,
    entities
  };
}

export function validateDomainContext(contract: DomainContract, entityName: string, context: PredicateContext): DomainValidationResult {
  if (contract.schemaVersion !== DOMAIN_CONTRACT_SCHEMA_VERSION) {
    return { ok: false, diagnostics: [{ path: "$.schemaVersion", message: `Expected ${DOMAIN_CONTRACT_SCHEMA_VERSION}.` }] };
  }
  const entity = contract.entities[entityName];
  if (!entity) {
    return { ok: false, diagnostics: [{ path: `$.entities.${entityName}`, message: "Expected a declared entity." }] };
  }
  const diagnostics: DomainValidationDiagnostic[] = [];
  for (const [fieldName, field] of Object.entries(entity.fields)) {
    const value = context[fieldName];
    const path = `$.${entityName}.${fieldName}`;
    if (value === undefined || value === null) {
      if (field.required !== false) diagnostics.push({ path, message: "Expected a value." });
      continue;
    }
    if (typeof value !== field.type) {
      diagnostics.push({ path, message: `Expected ${field.type}.` });
      continue;
    }
    if (field.type === "number") {
      const numeric = value as number;
      if (!Number.isFinite(numeric)) diagnostics.push({ path, message: "Expected a finite number." });
      if (field.min !== undefined && numeric < field.min) diagnostics.push({ path, message: `Expected >= ${field.min}.` });
      if (field.max !== undefined && numeric > field.max) diagnostics.push({ path, message: `Expected <= ${field.max}.` });
    }
  }
  return diagnostics.length === 0 ? { ok: true, diagnostics: [] } : { ok: false, diagnostics };
}

export function signPolicyBundle(bundle: SerializedPolicyBundle, keyId: string, secret: string): SignedPolicyBundle {
  if (!verifyPolicyBundleHash(bundle)) {
    throw new RuleValidationError(`Expected ${POLICY_BUNDLE_SCHEMA_VERSION} bundle with a valid hash before signing.`);
  }
  return {
    ...bundle,
    signature: {
      schemaVersion: POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION,
      algorithm: "symtorch-dev-fnv1a32",
      keyId,
      signature: stableHash(`${bundle.hash}:${keyId}:${secret}`)
    }
  };
}

export function verifySignedPolicyBundle(bundle: SignedPolicyBundle, secrets: Record<string, string>): boolean {
  return verifySignedPolicyBundleDetailed(bundle, secrets).ok;
}

export function verifySignedPolicyBundleDetailed(bundle: unknown, secrets: Record<string, string>): SignedPolicyBundleVerificationResult {
  if (!isSerializedPolicyBundle(bundle)) return { ok: false, reason: "invalid_bundle" };
  if (!isRecord(bundle) || !("signature" in bundle)) return { ok: false, reason: "missing_signature" };
  if (!isPolicyBundleSignature(bundle.signature)) return { ok: false, reason: "invalid_signature_schema" };
  const secret = secrets[bundle.signature.keyId];
  if (secret === undefined) return { ok: false, reason: "unknown_key" };
  if (bundle.signature.signature !== stableHash(`${bundle.hash}:${bundle.signature.keyId}:${secret}`)) {
    return { ok: false, reason: "signature_mismatch" };
  }
  return { ok: true, keyId: bundle.signature.keyId, algorithm: bundle.signature.algorithm };
}

export function productionRuntimeLimits(overrides: LogicRuntimeLimits = {}): Required<LogicRuntimeLimits> {
  return {
    maxRuleSourceLength: overrides.maxRuleSourceLength ?? 16_384,
    maxRules: overrides.maxRules ?? 256,
    maxPredicatesPerRule: overrides.maxPredicatesPerRule ?? 32,
    maxEntitiesPerEvaluation: overrides.maxEntitiesPerEvaluation ?? 10_000
  };
}

export function assessPolicyBundleSecurity(
  bundle: unknown,
  options: { secrets?: Record<string, string>; trustedKeyIds?: readonly string[]; limits?: LogicRuntimeLimits } = {}
): PolicyBundleSecurityAssessment {
  const diagnostics: PolicyBundleSecurityAssessment["diagnostics"] = [];
  if (!isSerializedPolicyBundle(bundle)) {
    diagnostics.push({ code: "invalid_hash", message: `Expected ${POLICY_BUNDLE_SCHEMA_VERSION} bundle with a valid deterministic hash.` });
    return { schemaVersion: PRODUCTION_READINESS_SCHEMA_VERSION, ok: false, diagnostics };
  }

  const limits = productionRuntimeLimits(options.limits);
  const validation = validateProgram(bundle.rules, { limits });
  if (!validation.ok) diagnostics.push({ code: "rule_validation", message: validation.error.message });

  if (bundle.rules.length > limits.maxRuleSourceLength || bundle.predicates.length > limits.maxRules) {
    diagnostics.push({ code: "runtime_limit", message: "Bundle exceeds configured production runtime limits." });
  }

  if (options.secrets) {
    const signature = verifySignedPolicyBundleDetailed(bundle, options.secrets);
    if (!signature.ok) diagnostics.push({ code: "invalid_signature", message: `Bundle signature verification failed: ${signature.reason}.` });
    if (signature.ok && options.trustedKeyIds && !options.trustedKeyIds.includes(signature.keyId)) {
      diagnostics.push({ code: "untrusted_key", message: `Bundle key "${signature.keyId}" is not in the trusted key set.` });
    }
  } else {
    diagnostics.push({
      code: "security_boundary",
      message: "No signature secrets were provided. Treat this as local integrity validation, not trusted policy admission."
    });
  }

  return {
    schemaVersion: PRODUCTION_READINESS_SCHEMA_VERSION,
    ok: diagnostics.length === 0,
    diagnostics
  };
}

export function getProductionReadinessReport(version = "0.29.0"): ProductionReadinessReport {
  return {
    schemaVersion: PRODUCTION_READINESS_SCHEMA_VERSION,
    version,
    productionReady: false,
    tracks: [
      {
        id: "typed_domains",
        status: "alpha",
        evidence: ["symtorch.domainContract.v1", "validateDomainContext()"],
        remaining: ["entity-aware grounding and schema-driven predicate binding"]
      },
      {
        id: "bundle_signing",
        status: "alpha",
        evidence: ["symtorch.policyBundleSignature.v1", "verifySignedPolicyBundleDetailed()"],
        remaining: ["replace development FNV signatures with audited cryptographic signatures and key rotation"]
      },
      {
        id: "durable_persistence",
        status: "alpha",
        evidence: ["decision ledger snapshots", "append-oriented Node sink"],
        remaining: ["transactional adapters, corruption recovery, IndexedDB and SQLite production adapters"]
      },
      {
        id: "trace_snapshots",
        status: "alpha",
        evidence: ["versioned explanations", "expected decision fixtures"],
        remaining: ["full golden trace snapshot corpus across policies and ledgers"]
      },
      {
        id: "runtime_limits",
        status: "alpha",
        evidence: ["productionRuntimeLimits()", "core, logic, and agent limit hooks"],
        remaining: ["abortable execution and browser workload timeouts"]
      },
      {
        id: "error_taxonomy",
        status: "alpha",
        evidence: ["SymTorchError codes for backend, policy, predicate, replay, resource, validation, and shape failures"],
        remaining: ["package-wide conversion of generic errors to typed errors"]
      },
      {
        id: "cpu_gpu_parity",
        status: "planned",
        evidence: ["CPU correctness oracle", "explicit WebGPU kernel prototypes"],
        remaining: ["backend-routed WebGPU execution and forward/gradient parity gates"]
      },
      {
        id: "api_stability",
        status: "alpha",
        evidence: ["docs/api-surface.md", "release manifest schema checks"],
        remaining: ["export snapshots and formal deprecation policy"]
      },
      {
        id: "security_model",
        status: "alpha",
        evidence: ["assessPolicyBundleSecurity()", "explicit non-claims"],
        remaining: ["sandbox boundary, trusted registries, and resource-exhaustion controls"]
      },
      {
        id: "real_apps",
        status: "alpha",
        evidence: ["browser policy workbench", "checked-in policy fixtures"],
        remaining: ["production app shell, auth integration boundaries, and deployment documentation"]
      }
    ]
  };
}

export function loadPolicyBundle(bundle: SerializedPolicyBundle, options: LoadPolicyBundleOptions = {}): LoadedPolicyBundle {
  if (!isSerializedPolicyBundle(bundle)) {
    throw new RuleValidationError(`Expected ${POLICY_BUNDLE_SCHEMA_VERSION} bundle with a valid hash.`);
  }
  const program = new RuleProgram(bundle.rules, options.limits ? { limits: options.limits } : {});
  const registry = new PredicateRegistry();
  for (const predicate of bundle.predicates) registry.register(predicateFromBundle(predicate));
  const engine = new FuzzyRuleEngine(registry, {
    ...(options.limits ? { limits: options.limits } : {}),
    ...(options.observer ? { observer: options.observer } : {})
  });
  return { bundle, program, registry, engine };
}

export function productAnd(values: readonly Tensor[]): Tensor {
  return values.reduce((acc, value) => mul(acc, value), tensor(1));
}

export function probabilisticOr(a: Tensor, b: Tensor): Tensor {
  return sub(sub(a, mul(a, b)), sub(tensor(0), b));
}

export function fuzzyNot(value: Tensor): Tensor {
  return sub(1, value);
}

export function formatPredicate(call: PredicateCall): string {
  const prefix = call.negated ? "not " : "";
  return `${prefix}${call.name}(${call.terms.map((term) => term.name).join(", ")})`;
}

function parseRuleAt(source: string, startIndex: number, fullSource: string): RuleAst {
  const trimmedStart = leadingWhitespaceLength(source);
  const trimmed = source.trim();
  const normalized = trimmed.replace(/\.$/, "");
  const localBase = startIndex + trimmedStart;
  const separatorIndex = normalized.indexOf(":-");
  if (separatorIndex !== normalized.lastIndexOf(":-")) {
    throw new RuleParseError("Rule contains more than one body separator \":-\".", fullSource, localBase + normalized.lastIndexOf(":-"));
  }
  const headText = separatorIndex >= 0 ? normalized.slice(0, separatorIndex).trim() : normalized.trim();
  const headStart = localBase + (separatorIndex >= 0 ? leadingWhitespaceLength(normalized.slice(0, separatorIndex)) : leadingWhitespaceLength(normalized));
  if (!headText) throw new RuleParseError("Rule is missing a head.", fullSource, localBase);
  const head = parsePredicateAt(headText, fullSource, headStart);
  const bodyText = separatorIndex >= 0 ? normalized.slice(separatorIndex + 2) : "";
  const bodyStart = localBase + separatorIndex + 2;
  const body = bodyText ? splitTopLevelWithOffsets(bodyText, fullSource, bodyStart).map((part) => parsePredicateAt(part.text, fullSource, part.start)) : [];
  return { head, body, source: trimmed };
}

function parsePredicate(text: string): PredicateCall {
  return parsePredicateAt(text, text, 0);
}

function parsePredicateAt(text: string, fullSource: string, startIndex: number): PredicateCall {
  const trimmedStart = leadingWhitespaceLength(text);
  const trimmed = text.trim();
  const absoluteStart = startIndex + trimmedStart;
  const negated = trimmed.startsWith("not ");
  const body = negated ? trimmed.slice(4).trim() : trimmed;
  const bodyStart = absoluteStart + (negated ? 4 + leadingWhitespaceLength(trimmed.slice(4)) : 0);
  const match = /^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$/.exec(body);
  if (!match) throw new RuleParseError(`Invalid predicate call "${trimmed}". Expected name(term, ...).`, fullSource, absoluteStart);
  const name = match[1];
  if (!name) throw new RuleParseError(`Invalid predicate name "${trimmed}".`, fullSource, bodyStart);
  const termsText = match[2] ?? "";
  const termsStart = bodyStart + name.length + 1;
  const terms = splitTermsWithOffsets(termsText, termsStart)
    .filter((term) => term.text.trim().length > 0)
    .map((term) => parseTermAt(term.text, fullSource, term.start));
  return { name, terms, negated };
}

function parseTerm(text: string): Term {
  return parseTermAt(text, text, 0);
}

function parseTermAt(text: string, fullSource: string, startIndex: number): Term {
  const trimmedStart = leadingWhitespaceLength(text);
  const name = text.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new RuleParseError(`Invalid term "${name}". Terms must be identifiers.`, fullSource, startIndex + trimmedStart);
  }
  return {
    kind: /^[A-Z_]/.test(name) ? "variable" : "constant",
    name
  };
}

function splitTopLevel(text: string): string[] {
  return splitTopLevelWithOffsets(text, text, 0).map((part) => part.text);
}

function splitTopLevelWithOffsets(text: string, fullSource: string, startIndex: number): { text: string; start: number }[] {
  const parts: string[] = [];
  const offsets: number[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === "(") depth += 1;
    if (char === ")") {
      depth -= 1;
      if (depth < 0) throw new RuleParseError("Unexpected closing parenthesis.", fullSource, startIndex + i);
    }
    if (char === "," && depth === 0) {
      const raw = text.slice(start, i);
      parts.push(raw.trim());
      offsets.push(startIndex + start + leadingWhitespaceLength(raw));
      start = i + 1;
    }
  }
  if (depth > 0) throw new RuleParseError("Unclosed parenthesis in rule body.", fullSource, startIndex + text.length - 1);
  const raw = text.slice(start);
  parts.push(raw.trim());
  offsets.push(startIndex + start + leadingWhitespaceLength(raw));
  return parts
    .map((part, index) => ({ text: part, start: offsets[index] ?? startIndex }))
    .filter((part) => part.text.length > 0);
}

function splitTermsWithOffsets(text: string, startIndex: number): { text: string; start: number }[] {
  const terms: { text: string; start: number }[] = [];
  let start = 0;
  for (let i = 0; i <= text.length; i++) {
    if (i === text.length || text[i] === ",") {
      const raw = text.slice(start, i);
      terms.push({ text: raw, start: startIndex + start + leadingWhitespaceLength(raw) });
      start = i + 1;
    }
  }
  return terms;
}

function splitRuleSources(source: string): { text: string; start: number }[] {
  const rules: { text: string; start: number }[] = [];
  let start = 0;
  for (let i = 0; i < source.length; i++) {
    if (source[i] === ".") {
      const raw = source.slice(start, i + 1);
      const trimmedStart = leadingWhitespaceLength(raw);
      const text = raw.trim();
      if (text) rules.push({ text, start: start + trimmedStart });
      start = i + 1;
    }
  }
  const raw = source.slice(start);
  const trimmedStart = leadingWhitespaceLength(raw);
  const text = raw.trim();
  if (text) rules.push({ text: `${text}.`, start: start + trimmedStart });
  return rules;
}

function validatePredicateBindings(rules: readonly RuleAst[], registry: PredicateRegistry): RuleDiagnostic[] {
  const diagnostics: RuleDiagnostic[] = [];
  const seen = new Set<string>();
  for (const rule of rules) {
    for (const call of rule.body) {
      if (registry.has(call.name)) continue;
      const key = `${rule.source}:${call.name}`;
      if (seen.has(key)) continue;
      seen.add(key);
      diagnostics.push({
        code: "missing_predicate",
        severity: "error",
        predicate: call.name,
        message: `Predicate "${call.name}" is not registered.`
      });
    }
  }
  return diagnostics;
}

function evaluatePredicate(predicate: Predicate, call: PredicateCall, context: PredicateContext): Tensor {
  try {
    return predicate.evaluate(call, context);
  } catch (error) {
    if (error instanceof PredicateEvaluationError) throw error;
    const message = error instanceof Error ? error.message : String(error);
    throw new PredicateEvaluationError(call.name, message);
  }
}

function enforceRuleSourceLimit(source: string, limits?: LogicRuntimeLimits): void {
  const max = limits?.maxRuleSourceLength;
  if (max !== undefined && source.length > max) {
    throw new ResourceLimitError(`Rule source length ${source.length} exceeds maxRuleSourceLength=${max}.`);
  }
}

function enforceRuleCountLimit(count: number, limits?: LogicRuntimeLimits): void {
  const max = limits?.maxRules;
  if (max !== undefined && count > max) {
    throw new ResourceLimitError(`Rule count ${count} exceeds maxRules=${max}.`);
  }
}

function enforcePredicatesPerRuleLimit(rule: RuleAst, limits?: LogicRuntimeLimits): void {
  const max = limits?.maxPredicatesPerRule;
  if (max !== undefined && rule.body.length > max) {
    throw new ResourceLimitError(`Rule "${rule.source}" has ${rule.body.length} predicates, exceeding maxPredicatesPerRule=${max}.`);
  }
}

function enforceEntityEvaluationLimit(count: number, limits?: LogicRuntimeLimits): void {
  const max = limits?.maxEntitiesPerEvaluation;
  if (max !== undefined && count > max) {
    throw new ResourceLimitError(`Entity evaluation count ${count} exceeds maxEntitiesPerEvaluation=${max}.`);
  }
}

function diagnosticFromParseError(error: RuleParseError): RuleDiagnostic {
  return {
    code: "parse_error",
    severity: "error",
    message: error.message,
    line: error.line,
    column: error.column,
    snippet: error.snippet
  };
}

function leadingWhitespaceLength(value: string): number {
  return value.length - value.trimStart().length;
}

function locateSource(source: string, index: number): { line: number; column: number; snippet: string } {
  const safeIndex = Math.max(0, Math.min(index, Math.max(0, source.length - 1)));
  const before = source.slice(0, safeIndex);
  const line = before.split("\n").length;
  const lineStart = Math.max(source.lastIndexOf("\n", safeIndex - 1) + 1, 0);
  const nextLine = source.indexOf("\n", safeIndex);
  const lineEnd = nextLine === -1 ? source.length : nextLine;
  return {
    line,
    column: safeIndex - lineStart + 1,
    snippet: source.slice(lineStart, lineEnd)
  };
}

function readScalar(context: PredicateContext, key: string): number {
  const value = context[key];
  if (typeof value !== "number") throw new Error(`Predicate context key "${key}" must be a number.`);
  return value;
}

function readFeatureTensor(context: PredicateContext, key: string, expectedLength: number): Tensor {
  const value = context[key];
  if (value instanceof Tensor) {
    if (value.shape.length === 1 && value.shape[0] === expectedLength) return value.reshape([1, expectedLength]);
    if (value.shape.length === 2 && value.shape[0] === 1 && value.shape[1] === expectedLength) return value;
    throw new Error(`Predicate context tensor "${key}" must have shape [${expectedLength}] or [1, ${expectedLength}].`);
  }
  if (!Array.isArray(value)) throw new Error(`Predicate context key "${key}" must be a number array or Tensor.`);
  if (value.length !== expectedLength || !value.every((item) => typeof item === "number")) {
    throw new Error(`Predicate context key "${key}" must contain ${expectedLength} numbers.`);
  }
  return tensor(value as number[], { shape: [1, expectedLength] });
}

function formatScore(value: number): string {
  return value.toFixed(4);
}

function isAggregatedExplanation(explanation: RuleExplanation | AggregatedRuleExplanation): explanation is AggregatedRuleExplanation {
  return "rules" in explanation;
}

function serializeRuleExplanation(explanation: RuleExplanation): SerializedRuleExplanation {
  return {
    schemaVersion: EXPLANATION_SCHEMA_VERSION,
    type: "rule",
    rule: explanation.rule,
    head: explanation.head,
    score: explanation.score,
    predicates: explanation.predicates.map(serializePredicateTrace)
  };
}

function serializePredicateTrace(trace: PredicateTrace): SerializedPredicateTrace {
  const serialized: SerializedPredicateTrace = {
    name: trace.name,
    negated: trace.negated,
    value: trace.value,
    contribution: trace.contribution
  };
  if (trace.kind) serialized.kind = trace.kind;
  if (trace.detail) serialized.detail = stableDetail(trace.detail);
  return serialized;
}

function stableDetail(detail: Record<string, number | string>): Record<string, number | string> {
  return Object.fromEntries(Object.entries(detail).sort(([left], [right]) => left.localeCompare(right)));
}

function stableKeys(context: PredicateContext): string[] {
  return Object.keys(context).sort();
}

function nowMs(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

function formatDetail(detail: Record<string, number | string>): string {
  const entries = Object.entries(detail).map(([key, value]) => `${key}=${typeof value === "number" ? formatScore(value) : value}`);
  return entries.length > 0 ? `(${entries.join(", ")})` : "";
}

function indent(value: string, prefix: string): string {
  return value
    .split("\n")
    .map((line) => `${prefix}${line}`)
    .join("\n");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isJsonPrimitive(value: unknown): value is string | number | boolean {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

function isPolicyBundlePredicate(value: unknown): value is PolicyBundlePredicate {
  if (!isRecord(value) || typeof value.name !== "string" || typeof value.kind !== "string") return false;
  if (value.kind === "fact") return value.key === undefined || typeof value.key === "string";
  if (value.kind === "threshold") {
    return typeof value.valueKey === "string" && typeof value.threshold === "number" && typeof value.slope === "number";
  }
  if (value.kind === "linear") {
    return typeof value.featureKey === "string" &&
      typeof value.featureCount === "number" &&
      Array.isArray(value.weights) &&
      value.weights.every((item) => typeof item === "number") &&
      typeof value.bias === "number";
  }
  return false;
}

function isPolicyBundleSignature(value: unknown): value is PolicyBundleSignature {
  return isRecord(value) &&
    value.schemaVersion === POLICY_BUNDLE_SIGNATURE_SCHEMA_VERSION &&
    value.algorithm === "symtorch-dev-fnv1a32" &&
    typeof value.keyId === "string" &&
    typeof value.signature === "string";
}

function predicateFromBundle(predicate: PolicyBundlePredicate): Predicate {
  if (predicate.kind === "fact") return new FactPredicate(predicate.name, predicate.key ?? predicate.name);
  if (predicate.kind === "threshold") {
    return new ThresholdPredicate(predicate.name, predicate.valueKey, predicate.threshold, predicate.slope);
  }
  if (predicate.weights.length !== predicate.featureCount) {
    throw new RuleValidationError(`Linear predicate "${predicate.name}" expected ${predicate.featureCount} weights, received ${predicate.weights.length}.`);
  }
  return new LinearPredicate(
    predicate.name,
    predicate.featureKey,
    predicate.featureCount,
    predicate.weights,
    predicate.bias
  );
}

function stableMetadata(metadata: Record<string, string | number | boolean>): Record<string, string | number | boolean> {
  return Object.fromEntries(Object.entries(metadata).sort(([left], [right]) => left.localeCompare(right)));
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function stableHash(value: string): string {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a32:${(hash >>> 0).toString(16).padStart(8, "0")}`;
}
