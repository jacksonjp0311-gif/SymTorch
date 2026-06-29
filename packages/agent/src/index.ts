import { decisionTrace, FactStore, type AggregatedRuleResult, type FuzzyRuleEngine, type PredicateContext, type RuleProgram, type SerializedAggregatedRuleExplanation } from "@symtorch/logic";

export type Observation = Record<string, unknown>;

export type AgentDecision = {
  action: string;
  results: readonly AggregatedRuleResult[];
};

export type SerializedAgentDecision = {
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

export class RuleAgent {
  readonly memory = new WorkingMemory();

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
