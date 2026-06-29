import { tensor } from "@symtorch/core";
import { decisionCard, FuzzyRuleEngine, PredicateRegistry, RuleProgram, RuleTrainer, ThresholdPredicate } from "@symtorch/logic";

const program = new RuleProgram(`
  escalate(X) :- high_risk(X), not approved(X).
`);

const highRisk = new ThresholdPredicate("high_risk", "risk", 0.9, 10);
const registry = new PredicateRegistry()
  .register(highRisk)
  .fixed("approved", (_call, context) => tensor(typeof context.approved === "number" ? context.approved : 0));

const engine = new FuzzyRuleEngine(registry);
const trainer = new RuleTrainer(engine, program.rules[0]!, registry, { learningRate: 0.2 });

const cases = [
  { risk: 0.15, approved: 0.05, label: 0 },
  { risk: 0.35, approved: 0.2, label: 0 },
  { risk: 0.72, approved: 0.05, label: 1 },
  { risk: 0.9, approved: 0.1, label: 1 },
  { risk: 0.88, approved: 0.95, label: 0 }
];

function score(row: { risk: number; approved: number }): number {
  return engine.evaluate(program.rules[0]!, row).score.item();
}

console.log("before", {
  threshold: Number(highRisk.threshold.item().toFixed(3)),
  lowRisk: Number(score({ risk: 0.25, approved: 0.05 }).toFixed(3)),
  highRisk: Number(score({ risk: 0.8, approved: 0.05 }).toFixed(3))
});

const training = trainer.fit(cases, { epochs: 100 });

const explanation = trainer.predict({ risk: 0.82, approved: 0.08 }).explanation;

console.log("after", {
  threshold: Number(highRisk.threshold.item().toFixed(3)),
  finalLoss: Number(training.finalLoss.toFixed(4)),
  lowRisk: Number(score({ risk: 0.25, approved: 0.05 }).toFixed(3)),
  highRisk: Number(score({ risk: 0.8, approved: 0.05 }).toFixed(3))
});
console.log(decisionCard({ score: trainer.predict({ risk: 0.82, approved: 0.08 }).score, explanation }));
console.log(JSON.stringify(explanation, null, 2));
