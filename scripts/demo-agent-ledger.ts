import { RuleAgent } from "@symtorch/agent";
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

const program = new RuleProgram(`
  escalate(X) :- high_risk(X), not approved(X).
  defer(X) :- approved(X).
`);
const registry = new PredicateRegistry()
  .register(new FactPredicate("high_risk"))
  .register(new FactPredicate("approved"));
const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

agent.memory.observeEntity("case-low", { high_risk: 0.2, approved: 0.1 });
agent.memory.observeEntity("case-approved", { high_risk: 0.9, approved: 0.95 });
agent.memory.observeEntity("case-hot", { high_risk: 0.9, approved: 0.1 });
agent.memory.observeEntity("case-borderline", { high_risk: 0.55, approved: 0.2 });

const ranked = agent.decideEntitiesTrace();
const accepted = agent.decideEntitiesTrace({ acceptedOnly: true, topK: 2 });
const entries = agent.recordEntityDecisions({ acceptedOnly: true, topK: 2 }, new Date("2026-06-29T00:00:00.000Z"));

console.log("SymTorch Agent Ledger Demo");
console.log("ranked decisions:");
console.log(JSON.stringify(ranked.map(({ entityId, action, score, accepted }) => ({
  entityId,
  action,
  score: Number(score.toFixed(4)),
  accepted
})), null, 2));
console.log("accepted top-2:");
console.log(JSON.stringify(accepted.map(({ entityId, action, score }) => ({
  entityId,
  action,
  score: Number(score.toFixed(4))
})), null, 2));
console.log("ledger replay:");
console.log(JSON.stringify(agent.ledger.all(), null, 2));

if (accepted.length !== 2) throw new Error("Expected two accepted decisions.");
if (entries.length !== 2) throw new Error("Expected two ledger entries.");
if (agent.ledger.all()[0]?.id !== "decision-1") throw new Error("Expected deterministic ledger ids.");
