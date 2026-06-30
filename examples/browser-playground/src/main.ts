import {
  serializeDecisionLedger,
  verifyDecisionLedgerReplay,
  type DecisionLedgerEntry,
  type RuleAgent,
  type SerializedAgentDecision,
  type SerializedEntityDecision
} from "@symtorch/agent";
import type { SerializedPolicyBundle } from "@symtorch/logic";
import {
  buildPolicyBundleAgent,
  createFactRegistry,
  createPlaygroundPolicyBundle,
  createPolicyHealth,
  createPolicyLibrary,
  createPlaygroundState,
  createTrainingRun,
  defaultScenario,
  exportPolicyBundleLibrary,
  exportPlaygroundPolicyBundle,
  exportPlaygroundScenario,
  exportPlaygroundState,
  parsePolicyBundleLibrary,
  parsePlaygroundPolicyBundle,
  parsePlaygroundState,
  parsePlaygroundScenario,
  type PolicyBundleLibrary,
  playgroundScenarios,
  scenarioById,
  scenarioIdFromPolicyBundle,
  savePolicyBundleToLibrary,
  summarizeTrainingRun,
  thresholdFromPolicyBundle,
  type TrainingRun,
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
let lastTrainingRun: TrainingRun | null = initialState?.lastTrainingRun ?? null;
let trainingSummary = summarizeTrainingRun(lastTrainingRun);
let policyLibrary: PolicyBundleLibrary = initialState?.policyLibrary ?? createPolicyLibrary();

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
const trainingHistory = mustElement<HTMLElement>("trainingHistory");
const policyHealth = mustElement<HTMLElement>("policyHealth");
const policyLibrarySelect = mustElement<HTMLSelectElement>("policyLibrarySelect");
const policyLibraryStatus = mustElement<HTMLElement>("policyLibraryStatus");
const evaluate = mustElement<HTMLButtonElement>("evaluate");
const record = mustElement<HTMLButtonElement>("record");
const resetRule = mustElement<HTMLButtonElement>("resetRule");
const train = mustElement<HTMLButtonElement>("train");
const saveBundle = mustElement<HTMLButtonElement>("saveBundle");
const loadBundle = mustElement<HTMLButtonElement>("loadBundle");
const exportLibrary = mustElement<HTMLButtonElement>("exportLibrary");
const exportState = mustElement<HTMLButtonElement>("exportState");
const exportScenario = mustElement<HTMLButtonElement>("exportScenario");
const exportBundle = mustElement<HTMLButtonElement>("exportBundle");
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
renderPolicyLibrary();
evaluatePolicy();

scenarioSelect.addEventListener("change", loadSelectedScenario);
evaluate.addEventListener("click", evaluatePolicy);
record.addEventListener("click", recordLedger);
train.addEventListener("click", trainHighRisk);
saveBundle.addEventListener("click", saveCurrentPolicyBundle);
loadBundle.addEventListener("click", loadSelectedPolicyBundle);
exportLibrary.addEventListener("click", exportCurrentPolicyLibrary);
exportState.addEventListener("click", exportCurrentState);
exportScenario.addEventListener("click", exportCurrentScenario);
exportBundle.addEventListener("click", exportCurrentPolicyBundle);
importState.addEventListener("click", importBufferedState);
ruleSource.addEventListener("input", persistState);
resetRule.addEventListener("click", () => {
  const scenario = scenarioById(scenarioId) ?? defaultScenario();
  ruleSource.value = scenario.ruleSource;
  cases.splice(0, cases.length, ...scenario.cases.map((item) => ({ ...item })));
  trainingExamples.splice(0, trainingExamples.length, ...scenario.trainingExamples.map((item) => ({ ...item })));
  trainedThreshold = scenario.trainedThreshold;
  lastTrainingRun = null;
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
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
  lastTrainingRun = null;
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
  renderScenarioHeader();
  renderFacts();
  renderTrainingExamples();
  persistState();
  evaluatePolicy();
}

type EvaluationResult = {
  bundle: SerializedPolicyBundle;
  agent: ReturnType<typeof buildPolicyBundleAgent>;
  decisions: ReturnType<ReturnType<typeof buildPolicyBundleAgent>["decideEntitiesTrace"]>;
};

function evaluatePolicy(replayOk: boolean | null = null): EvaluationResult | null {
  persistState();
  const validation = validateRuleSource(ruleSource.value, registry);
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    decisionList.innerHTML = "";
    traceOutput.textContent = JSON.stringify(validation.diagnostics, null, 2);
    policyHealth.innerHTML = "";
    return null;
  }

  diagnostics.textContent = "Rule validation: PASS";
  const bundle = createPlaygroundPolicyBundle(scenarioId, ruleSource.value, trainedThreshold);
  const agent = buildPolicyBundleAgent(bundle, cases);
  const decisions = agent.decideEntitiesTrace();
  decisionList.innerHTML = decisions.map(renderDecision).join("");
  traceOutput.textContent = JSON.stringify(decisions[0] ?? null, null, 2);
  renderPolicyHealth(createPolicyHealth(bundle, decisions[0] ?? null, replayOk));
  renderTrainingStats();
  return { bundle, agent, decisions };
}

function recordLedger(): void {
  const result = evaluatePolicy();
  if (!result) return;
  const { bundle, agent } = result;
  agent.recordEntityDecisions({ acceptedOnly: true, topK: 2 }, new Date("2026-06-29T00:00:00.000Z"));
  const snapshot = serializeDecisionLedger(agent.ledger);
  const replayReport = verifyDecisionLedgerReplay(snapshot, (entry) => replayDecision(bundle, entry), { atol: 1e-6 });
  traceOutput.textContent = JSON.stringify({ ledger: snapshot, replay: replayReport }, null, 2);
  renderPolicyHealth(createPolicyHealth(bundle, result.decisions[0] ?? null, replayReport.ok));
}

function trainHighRisk(): void {
  const validation = validateRuleSource(ruleSource.value, registry);
  if (!validation.ok) {
    diagnostics.textContent = validation.diagnostics.map((item) => item.message).join("\n");
    return;
  }

  const result = trainHighRiskRule(ruleSource.value, trainedThreshold, trainingExamples);
  trainedThreshold = result.afterThreshold;
  lastTrainingRun = createTrainingRun(scenarioId, result);
  persistState();
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
  trainingStats.textContent = trainingSummary;
  renderTrainingHistory();
  traceOutput.textContent = JSON.stringify(result.explanationJson, null, 2);
}

function exportCurrentState(): void {
  stateBuffer.value = exportPlaygroundState(scenarioId, ruleSource.value, cases, trainedThreshold, trainingExamples, lastTrainingRun, policyLibrary);
  stateStatus.textContent = "Exported current playground state.";
}

function exportCurrentScenario(): void {
  const scenario = scenarioById(scenarioId) ?? defaultScenario();
  stateBuffer.value = exportPlaygroundScenario({
    ...scenario,
    ruleSource: ruleSource.value,
    cases: cases.map((item) => ({ ...item })),
    trainingExamples: trainingExamples.map((item) => ({ ...item })),
    trainedThreshold
  });
  stateStatus.textContent = "Exported scenario contract.";
}

function exportCurrentPolicyBundle(): void {
  stateBuffer.value = exportPlaygroundPolicyBundle(scenarioId, ruleSource.value, trainedThreshold);
  stateStatus.textContent = "Exported policy bundle.";
  evaluatePolicy();
}

function saveCurrentPolicyBundle(): void {
  const result = evaluatePolicy();
  if (!result) return;
  policyLibrary = savePolicyBundleToLibrary(policyLibrary, result.bundle);
  persistState();
  renderPolicyLibrary(result.bundle.hash);
  policyLibraryStatus.textContent = `Saved ${result.bundle.name} (${result.bundle.hash}).`;
}

function loadSelectedPolicyBundle(): void {
  const selected = policyLibrary.bundles.find((item) => item.id === policyLibrarySelect.value);
  if (!selected) {
    policyLibraryStatus.textContent = "Select a saved policy bundle.";
    return;
  }
  loadPolicyBundleIntoWorkbench(selected.bundle);
  policyLibraryStatus.textContent = `Loaded ${selected.bundle.name} (${selected.bundle.hash}).`;
}

function exportCurrentPolicyLibrary(): void {
  stateBuffer.value = exportPolicyBundleLibrary(policyLibrary);
  stateStatus.textContent = "Exported policy bundle library.";
}

function importBufferedState(): void {
  const library = parsePolicyBundleLibrary(stateBuffer.value);
  if (library) {
    policyLibrary = library;
    persistState();
    renderPolicyLibrary();
    stateStatus.textContent = "Imported policy bundle library.";
    return;
  }

  const bundle = parsePlaygroundPolicyBundle(stateBuffer.value);
  if (bundle.ok) {
    policyLibrary = savePolicyBundleToLibrary(policyLibrary, bundle.bundle);
    loadPolicyBundleIntoWorkbench(bundle.bundle);
    renderPolicyLibrary(bundle.bundle.hash);
    stateStatus.textContent = "Imported policy bundle.";
    return;
  }

  const imported = parsePlaygroundState(stateBuffer.value);
  if (imported) {
    loadImportedState(imported);
    stateStatus.textContent = "Imported playground state.";
    return;
  }

  const scenario = parsePlaygroundScenario(stateBuffer.value);
  if (scenario.ok) {
    loadScenario(scenario.scenario);
    stateStatus.textContent = "Imported scenario contract.";
    return;
  }

  const bundleDiagnostics = bundle.diagnostics.map((item) => `${item.path} ${item.message}`).join(" ");
  const scenarioDiagnostics = scenario.diagnostics.map((item) => `${item.path} ${item.message}`).join(" ");
  stateStatus.textContent = `Import failed: ${scenarioDiagnostics || bundleDiagnostics}`;
}

function loadImportedState(imported: NonNullable<ReturnType<typeof parsePlaygroundState>>): void {
  scenarioId = imported.scenarioId;
  scenarioSelect.value = scenarioId;
  ruleSource.value = imported.ruleSource;
  cases.splice(0, cases.length, ...imported.cases);
  trainingExamples.splice(0, trainingExamples.length, ...imported.trainingExamples);
  trainedThreshold = imported.trainedThreshold;
  lastTrainingRun = imported.lastTrainingRun;
  policyLibrary = imported.policyLibrary;
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
  refreshAfterLoad();
}

function loadScenario(scenario: ReturnType<typeof defaultScenario>): void {
  scenarioId = scenario.id;
  scenarioSelect.value = scenarioId;
  ruleSource.value = scenario.ruleSource;
  cases.splice(0, cases.length, ...scenario.cases.map((item) => ({ ...item })));
  trainingExamples.splice(0, trainingExamples.length, ...scenario.trainingExamples.map((item) => ({ ...item })));
  trainedThreshold = scenario.trainedThreshold;
  lastTrainingRun = null;
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
  refreshAfterLoad();
}

function loadPolicyBundleIntoWorkbench(bundle: SerializedPolicyBundle): void {
  const importedScenarioId = scenarioIdFromPolicyBundle(bundle);
  scenarioId = importedScenarioId && scenarioById(importedScenarioId) ? importedScenarioId : scenarioId;
  scenarioSelect.value = scenarioId;
  ruleSource.value = bundle.rules;
  trainedThreshold = thresholdFromPolicyBundle(bundle) ?? trainedThreshold;
  lastTrainingRun = null;
  trainingSummary = summarizeTrainingRun(lastTrainingRun);
  refreshAfterLoad();
}

function refreshAfterLoad(): void {
  persistState();
  renderScenarioHeader();
  renderFacts();
  renderTrainingExamples();
  renderPolicyLibrary();
  evaluatePolicy();
}

function renderPolicyHealth(health: ReturnType<typeof createPolicyHealth>): void {
  policyHealth.innerHTML = [
    renderHealthItem("Schema", health.schemaVersion),
    renderHealthItem("Hash", health.hash),
    renderHealthItem("Verified", health.hashVerified ? "PASS" : "FAIL"),
    renderHealthItem("Rules", String(health.ruleCount)),
    renderHealthItem("Predicates", String(health.predicateCount)),
    renderHealthItem("Decision", health.lastDecisionStatus),
    renderHealthItem("Replay", health.replayStatus)
  ].join("");
}

function renderPolicyLibrary(selectedHash: string | null = null): void {
  if (policyLibrary.bundles.length === 0) {
    policyLibrarySelect.innerHTML = `<option value="">No saved bundles</option>`;
    policyLibrarySelect.value = "";
    policyLibraryStatus.textContent = "No saved policy bundles.";
    return;
  }
  policyLibrarySelect.innerHTML = policyLibrary.bundles.map((item) => `
    <option value="${item.id}">
      ${item.bundle.name} ${item.bundle.version} ${item.bundle.hash}
    </option>
  `).join("");
  const selected = selectedHash
    ? policyLibrary.bundles.find((item) => item.bundle.hash === selectedHash)
    : policyLibrary.bundles[0];
  policyLibrarySelect.value = selected?.id ?? policyLibrary.bundles[0]!.id;
  policyLibraryStatus.textContent = `${policyLibrary.bundles.length} saved policy bundle${policyLibrary.bundles.length === 1 ? "" : "s"}.`;
}

function renderHealthItem(label: string, value: string): string {
  return `<div class="health-item"><span>${label}</span><strong>${value}</strong></div>`;
}

function replayDecision(bundle: SerializedPolicyBundle, entry: DecisionLedgerEntry): SerializedAgentDecision | SerializedEntityDecision {
  const replayAgent = buildPolicyBundleAgent(bundle, []);
  if (entry.kind === "entity" && "entityId" in entry.decision) {
    replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
    return replayAgent.decideEntityTrace(entry.decision.entityId);
  }
  replayAgent.observe(entry.context);
  return replayAgent.decideTrace();
}

function renderTrainingStats(): void {
  trainingStats.textContent = trainingSummary;
  renderTrainingHistory();
}

function renderTrainingHistory(): void {
  if (!lastTrainingRun) {
    trainingHistory.innerHTML = "";
    return;
  }
  const sampled = sampleHistory(lastTrainingRun.history, 12);
  trainingHistory.innerHTML = sampled.map((item) => `
    <span title="epoch ${item.epoch}: ${item.loss.toFixed(5)}">${item.loss.toFixed(3)}</span>
  `).join("");
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
    localStorage.setItem(stateKey, JSON.stringify(createPlaygroundState(scenarioId, ruleSource.value, cases, trainedThreshold, trainingExamples, lastTrainingRun, policyLibrary)));
  } catch {
    // Persistence is best-effort; evaluation should still work if storage is unavailable.
  }
}

function sampleHistory(history: readonly { epoch: number; loss: number }[], maxItems: number): readonly { epoch: number; loss: number }[] {
  if (history.length <= maxItems) return history;
  const stride = (history.length - 1) / (maxItems - 1);
  return Array.from({ length: maxItems }, (_value, index) => history[Math.round(index * stride)]!);
}
