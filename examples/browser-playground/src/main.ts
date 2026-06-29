import { RuleAgent } from "@symtorch/agent";
import { RuleProgram } from "@symtorch/logic";
import {
  buildAgent,
  createFactRegistry,
  createPlaygroundState,
  defaultScenario,
  exportPlaygroundState,
  parsePlaygroundState,
  playgroundScenarios,
  scenarioById,
  trainHighRiskRule,
  validateRuleSource
} from "./app-model";

const stateKey = "symtorch.browser-playground.state.v1";
const initialState = loadState();
const initialScenario = initialState ? scenarioById(initialState.scenarioId) ?? defaultScenario() : defaultScenario();
let scenarioId = initialState?.scenarioId ?? initialScenario.id;
const cases = initialState?.cases ?? initialScenario.cases;
const trainingExamples = initialState?.trainingExamples ?? initialScenario.trainingExamples;
const registry = createFactRegistry();

let trainedThreshold = initialState?.trainedThreshold ?? initialScenario.trainedThreshold;
let trainingSummary = "Not trained yet.";

const scenarioTitle = mustElement<HTMLElement>("scenarioTitle");
const scenarioDescription = mustElement<HTMLElement>("scenarioDescription");
const scenarioSelect = mustElement<HTMLSelectElement>("scenarioSelect");
const ruleSource = mustElement<HTMLTextAreaElement>("ruleSource");
const diagnostics = mustElement<HTMLElement>("diagnostics");
const facts = mustElement<HTMLElement>("facts");
const decisionList = mustElement<HTMLElement>("decisionList");
const traceOutput = mustElement<HTMLElement>("traceOutput");
const trainingExamplesView = mustElement<HTMLElement>("trainingExamples");
const trainingStats = mustElement<HTMLElement>("trainingStats");
const evaluate = mustElement<HTMLButtonElement>("evaluate");
const record = mustElement<HTMLButtonElement>("record");
const resetRule = mustElement<HTMLButtonElement>("resetRule");
const train = mustElement<HTMLButtonElement>("train");
const exportState = mustElement<HTMLButtonElement>("exportState");
const importState = mustElement<HTMLButtonElement>("importState");
const stateBuffer = mustElement<HTMLTextAreaElement>("stateBuffer");
const stateStatus = mustElement<HTMLElement>("stateStatus");

scenarioSelect.innerHTML = playgroundScenarios
  .map((scenario) => `<option value="${scenario.id}">${scenario.title}</option>`)
  .join("");
scenarioSelect.value = scenarioId;
ruleSource.value = initialState?.ruleSource ?? initialScenario.ruleSource;
renderScenarioHeader();
renderFacts();
renderTrainingExamples();
evaluatePolicy();

scenarioSelect.addEventListener("change", loadSelectedScenario);
evaluate.addEventListener("click", evaluatePolicy);
record.addEventListener("click", recordLedger);
train.addEventListener("click", trainHighRisk);
exportState.addEventListener("click", exportCurrentState);
importState.addEventListener("click", importBufferedState);
ruleSource.addEventListener("input", persistState);
resetRule.addEventListener("click", () => {
  const scenario = scenarioById(scenarioId) ?? defaultScenario();
  ruleSource.value = scenario.ruleSource;
  cases.splice(0, cases.length, ...scenario.cases.map((item) => ({ ...item })));
  trainingExamples.splice(0, trainingExamples.length, ...scenario.trainingExamples.map((item) => ({ ...item })));
  trainedThreshold = scenario.trainedThreshold;
  trainingSummary = "Not trained yet.";
  persistState();
  renderScenarioHeader();
  renderFacts();
  renderTrainingExamples();
  evaluatePolicy();
});

function loadSelectedScenario(): void {
  const scenario = scenarioById(scenarioSelect.value) ?? defaultScenario();
  scenarioId = scenario.id;
  ruleSource.value = scenario.ruleSource;
  cases.splice(0, cases.length, ...scenario.cases.map((item) => ({ ...item })));
  trainingExamples.splice(0, trainingExamples.length, ...scenario.trainingExamples.map((item) => ({ ...item })));
  trainedThreshold = scenario.trainedThreshold;
  trainingSummary = "Not trained yet.";
  renderScenarioHeader();
  renderFacts();
  renderTrainingExamples();
  persistState();
  evaluatePolicy();
}

function evaluatePolicy(): RuleAgent | null {
  persistState();
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

  const result = trainHighRiskRule(ruleSource.value, trainedThreshold, trainingExamples);
  trainedThreshold = result.afterThreshold;
  persistState();
  trainingSummary = [
    `threshold: ${result.beforeThreshold.toFixed(4)} -> ${result.afterThreshold.toFixed(4)}`,
    `score: ${result.beforeScore.toFixed(4)} -> ${result.afterScore.toFixed(4)}`,
    `loss: ${result.finalLoss.toFixed(4)}`
  ].join("\n");
  trainingStats.textContent = trainingSummary;
  traceOutput.textContent = JSON.stringify(result.explanationJson, null, 2);
}

function exportCurrentState(): void {
  stateBuffer.value = exportPlaygroundState(scenarioId, ruleSource.value, cases, trainedThreshold, trainingExamples);
  stateStatus.textContent = "Exported current playground state.";
}

function importBufferedState(): void {
  const imported = parsePlaygroundState(stateBuffer.value);
  if (!imported) {
    stateStatus.textContent = "Import failed: expected symtorch.playground.v1 JSON.";
    return;
  }

  ruleSource.value = imported.ruleSource;
  scenarioId = imported.scenarioId;
  scenarioSelect.value = scenarioId;
  cases.splice(0, cases.length, ...imported.cases);
  trainingExamples.splice(0, trainingExamples.length, ...imported.trainingExamples);
  trainedThreshold = imported.trainedThreshold;
  trainingSummary = "Imported state.";
  persistState();
  renderScenarioHeader();
  renderFacts();
  renderTrainingExamples();
  evaluatePolicy();
  stateStatus.textContent = "Imported playground state.";
}

function renderTrainingStats(): void {
  trainingStats.textContent = trainingSummary;
}

function renderScenarioHeader(): void {
  const scenario = scenarioById(scenarioId) ?? defaultScenario();
  scenarioTitle.textContent = scenario.title;
  scenarioDescription.textContent = scenario.description;
}

function renderTrainingExamples(): void {
  trainingExamplesView.innerHTML = trainingExamples.map((item, index) => `
    <article class="training-row">
      <strong>ex ${index + 1}</strong>
      <label>risk <input data-index="${index}" data-key="risk" type="range" min="0" max="1" step="0.01" value="${item.risk}" /></label>
      <label>approved <input data-index="${index}" data-key="approved" type="range" min="0" max="1" step="0.01" value="${item.approved}" /></label>
      <label>label <input data-index="${index}" data-key="label" type="range" min="0" max="1" step="1" value="${item.label}" /></label>
      <span>${item.risk.toFixed(2)} / ${item.approved.toFixed(2)} -> ${item.label.toFixed(0)}</span>
    </article>
  `).join("");

  trainingExamplesView.querySelectorAll<HTMLInputElement>("input").forEach((input) => {
    input.addEventListener("input", () => {
      const index = Number(input.dataset.index);
      const key = input.dataset.key as "risk" | "approved" | "label" | undefined;
      const item = trainingExamples[index];
      if (!item || !key) return;
      item[key] = Number(input.value);
      renderTrainingExamples();
    });
  });
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
      persistState();
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

function loadState(): ReturnType<typeof parsePlaygroundState> {
  try {
    return parsePlaygroundState(localStorage.getItem(stateKey));
  } catch {
    return null;
  }
}

function persistState(): void {
  try {
    localStorage.setItem(stateKey, JSON.stringify(createPlaygroundState(scenarioId, ruleSource.value, cases, trainedThreshold, trainingExamples)));
  } catch {
    // Persistence is best-effort; evaluation should still work if storage is unavailable.
  }
}
