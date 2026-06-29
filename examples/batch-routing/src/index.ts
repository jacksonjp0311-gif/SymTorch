import { decisionCard, FactPredicate, FactStore, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

const program = new RuleProgram(`
  escalate(X) :- high_risk(X), not approved(X).
  escalate(X) :- customer_vip(X).
  defer(X) :- approved(X).
`);

const facts = new FactStore()
  .setEntity("case-low", { high_risk: 0.2, approved: 0.1, customer_vip: 0.1 })
  .setEntity("case-approved", { high_risk: 0.9, approved: 0.95, customer_vip: 0.1 })
  .setEntity("case-hot", { high_risk: 0.9, approved: 0.1, customer_vip: 0.2 })
  .setEntity("case-vip", { high_risk: 0.3, approved: 0.1, customer_vip: 0.8 });

const registry = new PredicateRegistry()
  .register(new FactPredicate("high_risk"))
  .register(new FactPredicate("approved"))
  .register(new FactPredicate("customer_vip"));

const engine = new FuzzyRuleEngine(registry);
const ranked = engine.rankEntitiesByHead(program, facts, "escalate(X)");

console.log("escalation ranking");
for (const item of ranked) {
  console.log(`${item.entityId}: ${item.result.score.item().toFixed(4)}`);
}

console.log("\ntop decision");
console.log(decisionCard(ranked[0]!.result));
