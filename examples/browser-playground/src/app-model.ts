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
