import { RuleAgent } from "@symtorch/agent";
import {
  FactPredicate,
  FuzzyRuleEngine,
  PredicateRegistry,
  RuleProgram,
  RuleTrainer,
  ThresholdPredicate,
  validateProgram
} from "@symtorch/logic";
import { tensor } from "@symtorch/core";

type CaseFacts = {
  entityId: string;
  high_risk: number;
  approved: number;
};

const defaultRule = `escalate(X) :- high_risk(X), not approved(X).
defer(X) :- approved(X).`;

const cases: CaseFacts[] = [
  { entityId: "case-low", high_risk: 0.2, approved: 0.1 },
  { entityId: "case-hot", high_risk: 0.9, approved: 0.1 },
  { entityId: "case-approved", high_risk: 0.9, approved: 0.95 },
  { entityId: "case-borderline", high_risk: 0.55, approved: 0.2 }
];

const registry = new PredicateRegistry()
  .register(new FactPredicate("high_risk"))
  .register(new FactPredicate("approved"));

let trainedThreshold = 0.9;
let trainingSummary = "Not trained yet.";

const ruleSource = mustElement<HTMLTextAreaElement>("ruleSource");
const diagnostics = mustElement<HTMLElement>("diagnostics");
const facts = mustElement<HTMLElement>("facts");
const decisionList = mustElement<HTMLElement>("decisionList");
const traceOutput = mustElement<HTMLElement>("traceOutput");
const trainingStats = mustElement<HTMLElement>("trainingStats");
const evaluate = mustElement<HTMLButtonElement>("evaluate");
const record = mustElement<HTMLButtonElement>("record");
const resetRule = mustElement<HTMLButtonElement>("resetRule");
const train = mustElement<HTMLButtonElement>("train");

ruleSource.value = defaultRule;
renderFacts();
evaluatePolicy();

evaluate.addEventListener("click", evaluatePolicy);
record.addEventListener("click", recordLedger);
train.addEventListener("click", trainHighRisk);
resetRule.addEventListener("click", () => {
  ruleSource.value = defaultRule;
  evaluatePolicy();
});

function evaluatePolicy(): RuleAgent | null {
  const validation = validateProgram(ruleSource.value, { registry });
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    decisionList.innerHTML = "";
    traceOutput.textContent = JSON.stringify(validation.diagnostics, null, 2);
    return null;
  }

  diagnostics.textContent = "Rule validation: PASS";
  const agent = buildAgent(new RuleProgram(ruleSource.value));
  const decisions = agent.decideEntitiesTrace();
  decisionList.innerHTML = decisions.map(renderDecision).join("");
  traceOutput.textContent = JSON.stringify(decisions[0] ?? null, null, 2);
  renderTrainingStats();
  return agent;
}

function recordLedger(): void {
  const agent = evaluatePolicy();
  if (!agent) return;
  agent.recordEntityDecisions({ acceptedOnly: true, topK: 2 }, new Date("2026-06-29T00:00:00.000Z"));
  traceOutput.textContent = JSON.stringify(agent.ledger.all(), null, 2);
}

function trainHighRisk(): void {
  const validation = validateProgram(ruleSource.value, { registry });
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    return;
  }

  const program = new RuleProgram(ruleSource.value);
  const highRisk = new ThresholdPredicate("high_risk", "risk", trainedThreshold, 10);
  const trainableRegistry = new PredicateRegistry()
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
  trainedThreshold = highRisk.threshold.item();
  trainingSummary = [
    `threshold: ${beforeThreshold.toFixed(4)} -> ${trainedThreshold.toFixed(4)}`,
    `score: ${beforeScore.toFixed(4)} -> ${prediction.score.item().toFixed(4)}`,
    `loss: ${result.finalLoss.toFixed(4)}`
  ].join("\n");
  trainingStats.textContent = trainingSummary;
  traceOutput.textContent = JSON.stringify(prediction.explanation, null, 2);
}

function buildAgent(program: RuleProgram): RuleAgent {
  const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
  for (const item of cases) {
    agent.memory.observeEntity(item.entityId, {
      high_risk: item.high_risk,
      approved: item.approved
    });
  }
  return agent;
}

function renderTrainingStats(): void {
  trainingStats.textContent = trainingSummary;
}

function renderFacts(): void {
  facts.innerHTML = cases.map((item) => `
    <article class="case-row">
      <strong>${item.entityId}</strong>
      <label>risk <input data-entity="${item.entityId}" data-key="high_risk" type="range" min="0" max="1" step="0.01" value="${item.high_risk}" /></label>
      <label>approved <input data-entity="${item.entityId}" data-key="approved" type="range" min="0" max="1" step="0.01" value="${item.approved}" /></label>
      <span>${item.high_risk.toFixed(2)} / ${item.approved.toFixed(2)}</span>
    </article>
  `).join("");

  facts.querySelectorAll<HTMLInputElement>("input").forEach((input) => {
    input.addEventListener("input", () => {
      const item = cases.find((candidate) => candidate.entityId === input.dataset.entity);
      const key = input.dataset.key as "high_risk" | "approved" | undefined;
      if (!item || !key) return;
      item[key] = Number(input.value);
      renderFacts();
      evaluatePolicy();
    });
  });
}

function renderDecision(decision: ReturnType<RuleAgent["decideEntitiesTrace"]>[number]): string {
  return `
    <article class="decision ${decision.accepted ? "accepted" : ""}">
      <strong>${decision.entityId}</strong>
      <span>${decision.action}</span>
      <span>${decision.score.toFixed(4)}</span>
    </article>
  `;
}

function mustElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element #${id}`);
  return element as T;
}
