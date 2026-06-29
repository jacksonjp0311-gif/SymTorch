import type { FuzzyRuleEngine, PredicateContext, RuleProgram, RuleResult } from "@symtorch/logic";

export type Observation = Record<string, unknown>;

export type AgentDecision = {
  action: string;
  results: readonly RuleResult[];
};

export class WorkingMemory {
  private readonly state = new Map<string, unknown>();

  observe(observation: Observation): void {
    for (const [key, value] of Object.entries(observation)) this.state.set(key, value);
  }

  snapshot(): PredicateContext {
    return Object.fromEntries(this.state.entries());
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
    const results = this.engine.evaluateProgram(this.program, this.memory.snapshot());
    const best = results.reduce<RuleResult | null>((winner, result) => {
      if (!winner || result.score.item() > winner.score.item()) return result;
      return winner;
    }, null);
    const action = best && best.score.item() >= this.threshold ? best.explanation.head : "no_action";
    return { action, results };
  }
}

