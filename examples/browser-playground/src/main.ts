import { RuleAgent } from "@symtorch/agent";
import { RuleProgram } from "@symtorch/logic";
import { buildAgent, createFactRegistry, defaultCases, defaultRule, trainHighRiskRule, validateRuleSource } from "./app-model";

const cases = defaultCases();
const registry = createFactRegistry();

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
  const validation = validateRuleSource(ruleSource.value, registry);
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    decisionList.innerHTML = "";
    traceOutput.textContent = JSON.stringify(validation.diagnostics, null, 2);
    return null;
  }

  diagnostics.textContent = "Rule validation: PASS";
  const agent = buildAgent(new RuleProgram(ruleSource.value), cases, registry);
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
  const validation = validateRuleSource(ruleSource.value, registry);
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    return;
  }

  const result = trainHighRiskRule(ruleSource.value, trainedThreshold);
  trainedThreshold = result.afterThreshold;
  trainingSummary = [
    `threshold: ${result.beforeThreshold.toFixed(4)} -> ${result.afterThreshold.toFixed(4)}`,
    `score: ${result.beforeScore.toFixed(4)} -> ${result.afterScore.toFixed(4)}`,
    `loss: ${result.finalLoss.toFixed(4)}`
  ].join("\n");
  trainingStats.textContent = trainingSummary;
  traceOutput.textContent = JSON.stringify(result.explanationJson, null, 2);
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
