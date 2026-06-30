import { add, matmul, mul, sigmoid, sub, Tensor, tensor } from "@symtorch/core";
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

  constructor(source: string) {
    this.rules = parseProgram(source);
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
    if (!predicate) throw new Error(`No predicate registered for "${call.name}".`);
    const resolution: PredicateResolution = {
      score: predicate.evaluate(call, context),
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

export function parseProgram(source: string): RuleAst[] {
  return splitRuleSources(source).map((rule) => parseRuleAt(rule.text, rule.start, source));
}

export function validateProgram(source: string, options: RuleValidationOptions = {}): RuleValidationResult {
  try {
    const rules = parseProgram(source);
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
