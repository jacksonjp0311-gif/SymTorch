import { tensor } from "@symtorch/core";
import {
  decisionTrace,
  FuzzyRuleEngine,
  PredicateRegistry,
  RuleProgram,
  RuleTrainer,
  ThresholdPredicate
} from "@symtorch/logic";

const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
const highRisk = new ThresholdPredicate("high_risk", "risk", 0.9, 10);
const registry = new PredicateRegistry()
  .register(highRisk)
  .fixed("approved", (_call, context) => tensor(typeof context.approved === "number" ? context.approved : 0));
const engine = new FuzzyRuleEngine(registry);
const trainer = new RuleTrainer(engine, program.rules[0]!, registry, { learningRate: 0.2 });

const lowRisk = { risk: 0.25, approved: 0.05 };
const highRiskCase = { risk: 0.82, approved: 0.08 };
const thresholdBefore = highRisk.threshold.item();
const lowBefore = trainer.predict(lowRisk).score.item();
const highBefore = trainer.predict(highRiskCase).score.item();

const result = trainer.fit([
  { risk: 0.2, approved: 0.05, label: 0 },
  { risk: 0.35, approved: 0.15, label: 0 },
  { risk: 0.75, approved: 0.05, label: 1 },
  { risk: 0.9, approved: 0.1, label: 1 },
  { risk: 0.88, approved: 0.95, label: 0 }
], { epochs: 100 });

const prediction = trainer.predict(highRiskCase);
const lowAfter = trainer.predict(lowRisk).score.item();
const highAfter = prediction.score.item();

console.log("SymTorch Rule Training Demo");
console.log(`rule: ${program.rules[0]!.source}`);
console.log(`threshold before: ${format(thresholdBefore)}`);
console.log(`low risk score before: ${format(lowBefore)}`);
console.log(`high risk score before: ${format(highBefore)}`);
console.log(`threshold after: ${format(highRisk.threshold.item())}`);
console.log(`final loss: ${format(result.finalLoss)}`);
console.log(`low risk score after: ${format(lowAfter)}`);
console.log(`high risk score after: ${format(highAfter)}`);
console.log("trace:");
console.log(JSON.stringify(decisionTrace(prediction), null, 2));

if (result.finalLoss >= result.history[0]!.loss) throw new Error("Expected training loss to decrease.");
if (highRisk.threshold.item() >= thresholdBefore) throw new Error("Expected threshold to move downward.");
if (highAfter <= highBefore) throw new Error("Expected high-risk score to improve.");

function format(value: number): string {
  return value.toFixed(4);
}
