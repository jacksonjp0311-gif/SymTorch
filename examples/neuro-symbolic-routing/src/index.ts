import { RuleAgent } from "@symtorch/agent";
import { tensor } from "@symtorch/core";
import { FuzzyRuleEngine, RuleProgram } from "@symtorch/logic";

const program = new RuleProgram(`
  escalate(X) :- high_risk(X), not approved(X).
`);

const engine = new FuzzyRuleEngine((call, context) => {
  const scores = context.scores as Record<string, number>;
  return tensor(scores[call.name] ?? 0);
});

const agent = new RuleAgent(program, engine, 0.5);
agent.observe({
  caseId: "case-1042",
  scores: {
    high_risk: 0.91,
    approved: 0.18
  }
});

const decision = agent.decide();
console.log(decision.action);
console.log(JSON.stringify(decision.results[0]?.explanation, null, 2));

