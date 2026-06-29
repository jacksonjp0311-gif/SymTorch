import { add, bind, mul, sum, tensor, type Tensor, unbind, zeros } from "@symtorch/core";
import { decisionTrace, FactStore, type AggregatedRuleResult, type FuzzyRuleEngine, type PredicateContext, type RuleProgram, type SerializedAggregatedRuleExplanation } from "@symtorch/logic";

export type Observation = Record<string, unknown>;

export type AgentDecision = {
  action: string;
  results: readonly AggregatedRuleResult[];
};

export const AGENT_DECISION_SCHEMA_VERSION = "symtorch.agentDecision.v1" as const;
export type AgentDecisionSchemaVersion = typeof AGENT_DECISION_SCHEMA_VERSION;

export type SerializedAgentDecision = {
  schemaVersion: AgentDecisionSchemaVersion;
  action: string;
  selectedHead: string | null;
  score: number;
  threshold: number;
  accepted: boolean;
  trace: SerializedAggregatedRuleExplanation | null;
  results: SerializedAggregatedRuleExplanation[];
};

export type SerializedEntityDecision = SerializedAgentDecision & {
  entityId: string;
};

export type EntityDecisionOptions = {
  entityIds?: readonly string[];
  minScore?: number;
  acceptedOnly?: boolean;
  topK?: number;
};

export type DecisionLedgerEntry = {
  id: string;
  createdAt: string;
  kind: "agent" | "entity";
  context: PredicateContext;
  decision: SerializedAgentDecision | SerializedEntityDecision;
};

export class DecisionLedger {
  private readonly entries: DecisionLedgerEntry[] = [];
  private nextId = 1;

  append(entry: Omit<DecisionLedgerEntry, "id" | "createdAt">, createdAt = new Date()): DecisionLedgerEntry {
    const record: DecisionLedgerEntry = {
      id: `decision-${this.nextId++}`,
      createdAt: createdAt.toISOString(),
      kind: entry.kind,
      context: cloneContext(entry.context),
      decision: cloneJson(entry.decision)
    };
    this.entries.push(record);
    return cloneJson(record);
  }

  all(): DecisionLedgerEntry[] {
    return this.entries.map((entry) => cloneJson(entry));
  }

  clear(): void {
    this.entries.length = 0;
    this.nextId = 1;
  }
}

export class WorkingMemory {
  private readonly facts = new FactStore();

  observe(observation: Observation): void {
    this.facts.observe(observation);
  }

  observeEntity(entityId: string, observation: Observation): void {
    this.facts.setEntity(entityId, observation);
  }

  snapshot(): PredicateContext {
    return this.facts.context();
  }

  entitySnapshot(entityId: string): PredicateContext {
    return this.facts.entityContext(entityId);
  }

  entityIds(): string[] {
    return this.facts.entityIds();
  }
}

export type HolographicMemoryTrace = {
  dimension: number;
  bindings: number;
  vector: number[];
};

export class HolographicMemory {
  private trace: Tensor;
  private bindingCount = 0;

  constructor(readonly dimension: number) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new Error(`HolographicMemory dimension must be a positive integer, received ${dimension}.`);
    }
    this.trace = zeros([dimension]);
  }

  bind(role: Tensor, value: Tensor): Tensor {
    this.assertVector(role, "role");
    this.assertVector(value, "value");
    const bound = bind(role, value);
    this.trace = add(this.trace, bound);
    this.bindingCount += 1;
    return bound;
  }

  recall(role: Tensor): Tensor {
    this.assertVector(role, "role");
    return unbind(this.trace, role);
  }

  similarity(left: Tensor, right: Tensor): number {
    this.assertVector(left, "left");
    this.assertVector(right, "right");
    return sum(mul(left, right)).item();
  }

  snapshot(): HolographicMemoryTrace {
    return {
      dimension: this.dimension,
      bindings: this.bindingCount,
      vector: this.trace.toArray()
    };
  }

  clear(): void {
    this.trace = zeros([this.dimension]);
    this.bindingCount = 0;
  }

  private assertVector(value: Tensor, label: string): void {
    if (value.ndim !== 1 || value.size !== this.dimension) {
      throw new Error(`HolographicMemory ${label} must have shape [${this.dimension}], received [${value.shape.join(", ")}].`);
    }
  }
}

export function vectorSymbol(values: readonly number[]): Tensor {
  return tensor(values, { shape: [values.length] });
}

export class RuleAgent {
  readonly memory = new WorkingMemory();
  readonly ledger = new DecisionLedger();

  constructor(
    private readonly program: RuleProgram,
    private readonly engine: FuzzyRuleEngine,
    private readonly threshold = 0.5
  ) {}

  observe(observation: Observation): void {
    this.memory.observe(observation);
  }

  decide(): AgentDecision {
    return this.decideFromContext(this.memory.snapshot());
  }

  decideTrace(): SerializedAgentDecision {
    return this.serializeDecision(this.decide());
  }

  decideEntityTrace(entityId: string): SerializedEntityDecision {
    return {
      entityId,
      ...this.serializeDecision(this.decideFromContext(this.memory.entitySnapshot(entityId)))
    };
  }

  decideEntitiesTrace(options: EntityDecisionOptions | readonly string[] = {}): SerializedEntityDecision[] {
    const normalized = normalizeEntityDecisionOptions(options);
    const entityIds = normalized.entityIds ?? this.memory.entityIds();
    let decisions = entityIds
      .map((entityId) => this.decideEntityTrace(entityId))
      .sort(compareEntityDecisions);
    if (normalized.minScore !== undefined) {
      const minScore = normalized.minScore;
      decisions = decisions.filter((decision) => decision.score >= minScore);
    }
    if (normalized.acceptedOnly) decisions = decisions.filter((decision) => decision.accepted);
    if (normalized.topK !== undefined) decisions = decisions.slice(0, normalized.topK);
    return decisions;
  }

  recordDecision(createdAt?: Date): DecisionLedgerEntry {
    return this.ledger.append({
      kind: "agent",
      context: this.memory.snapshot(),
      decision: this.decideTrace()
    }, createdAt);
  }

  recordEntityDecision(entityId: string, createdAt?: Date): DecisionLedgerEntry {
    return this.ledger.append({
      kind: "entity",
      context: this.memory.entitySnapshot(entityId),
      decision: this.decideEntityTrace(entityId)
    }, createdAt);
  }

  recordEntityDecisions(options: EntityDecisionOptions | readonly string[] = {}, createdAt?: Date): DecisionLedgerEntry[] {
    const decisions = this.decideEntitiesTrace(options);
    return decisions.map((decision) => this.ledger.append({
      kind: "entity",
      context: this.memory.entitySnapshot(decision.entityId),
      decision
    }, createdAt));
  }

  private decideFromContext(context: PredicateContext): AgentDecision {
    const results = this.engine.evaluateProgramGrouped(this.program, context);
    const best = selectBestResult(results);
    const action = best && best.score.item() >= this.threshold ? best.head : "no_action";
    return { action, results };
  }

  private serializeDecision(decision: AgentDecision): SerializedAgentDecision {
    const best = selectBestResult(decision.results);
    const score = best?.score.item() ?? 0;
    const accepted = Boolean(best && score >= this.threshold);
    return {
      schemaVersion: AGENT_DECISION_SCHEMA_VERSION,
      action: accepted && best ? best.head : "no_action",
      selectedHead: best?.head ?? null,
      score,
      threshold: this.threshold,
      accepted,
      trace: best ? decisionTrace(best) : null,
      results: decision.results.map((result) => decisionTrace(result))
    };
  }
}

export function isSerializedAgentDecision(value: unknown): value is SerializedAgentDecision {
  if (!isRecord(value)) return false;
  return (
    value.schemaVersion === AGENT_DECISION_SCHEMA_VERSION &&
    typeof value.action === "string" &&
    isNullableString(value.selectedHead) &&
    isFiniteNumber(value.score) &&
    isFiniteNumber(value.threshold) &&
    typeof value.accepted === "boolean" &&
    (value.trace === null || isRecord(value.trace)) &&
    Array.isArray(value.results) &&
    value.results.every(isRecord)
  );
}

export function isSerializedEntityDecision(value: unknown): value is SerializedEntityDecision {
  if (!isRecord(value)) return false;
  const entityId = value.entityId;
  return isSerializedAgentDecision(value) && typeof entityId === "string";
}

function selectBestResult(results: readonly AggregatedRuleResult[]): AggregatedRuleResult | null {
  return results.reduce<AggregatedRuleResult | null>((winner, result) => {
    if (!winner || result.score.item() > winner.score.item()) return result;
    return winner;
  }, null);
}

function compareEntityDecisions(left: SerializedEntityDecision, right: SerializedEntityDecision): number {
  const scoreOrder = right.score - left.score;
  if (scoreOrder !== 0) return scoreOrder;
  return left.entityId.localeCompare(right.entityId);
}

function normalizeEntityDecisionOptions(options: EntityDecisionOptions | readonly string[]): EntityDecisionOptions {
  return isEntityIdList(options) ? { entityIds: options } : options;
}

function isEntityIdList(options: EntityDecisionOptions | readonly string[]): options is readonly string[] {
  return Array.isArray(options);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function cloneContext(context: PredicateContext): PredicateContext {
  return cloneJson(context) as PredicateContext;
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
