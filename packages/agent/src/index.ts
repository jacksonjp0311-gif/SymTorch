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
    const results = this.engine.evaluateProgramGrouped(this.program, this.memory.snapshot());
    const best = selectBestResult(results);
    const action = best && best.score.item() >= this.threshold ? best.head : "no_action";
    return { action, results };
  }

  decideTrace(): SerializedAgentDecision {
    const decision = this.decide();
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
