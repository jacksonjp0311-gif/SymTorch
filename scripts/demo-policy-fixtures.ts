import { readdirSync, readFileSync } from "node:fs";
import { createPolicyAgent, serializeDecisionLedger, verifyDecisionLedgerReplay, type DecisionLedgerEntry } from "@symtorch/agent";
import { isSerializedPolicyBundle, loadPolicyBundle, verifyPolicyBundleHash, type SerializedPolicyBundle } from "@symtorch/logic";

type FixtureCase = {
  entityId: string;
  high_risk: number;
  approved: number;
};

type ExpectedDecision = {
  rank: number;
  entityId: string;
  action: string;
  accepted: boolean;
  scoreMin: number;
  scoreMax: number;
};

type ExpectedDecisionFile = {
  schemaVersion: "symtorch.policyExpectations.v1";
  fixtures: Record<string, ExpectedDecision[]>;
};

const fixtureCases: Record<string, FixtureCase[]> = {
  "case-escalation": [
    { entityId: "case-hot", high_risk: 0.9, approved: 0.1 },
    { entityId: "case-approved", high_risk: 0.9, approved: 0.95 }
  ],
  "fraud-review": [
    { entityId: "txn-hot", high_risk: 0.94, approved: 0.05 },
    { entityId: "txn-approved", high_risk: 0.8, approved: 0.92 }
  ],
  "support-routing": [
    { entityId: "ticket-outage", high_risk: 0.93, approved: 0.08 },
    { entityId: "ticket-resolved", high_risk: 0.72, approved: 0.9 }
  ]
};

const policiesDir = new URL("../examples/policies/", import.meta.url);
const files = readdirSync(policiesDir)
  .filter((file) => file.endsWith(".policy.json"))
  .sort();
const expectations = readExpectations();

console.log("SymTorch Policy Fixture Suite");
console.log(`fixtures: ${files.length}`);

for (const file of files) {
  const bundle = readBundle(file);
  const loaded = loadPolicyBundle(bundle);
  const cases = casesFor(bundle);
  const agent = createPolicyAgent(bundle, {
    threshold: 0.5,
    limits: {
      maxRuleSourceLength: 1_000,
      maxEntitiesPerBatch: 20,
      maxReplayEntries: 20
    }
  });

  for (const item of cases) {
    agent.memory.observeEntity(item.entityId, {
      high_risk: item.high_risk,
      approved: item.approved
    });
  }

  const decisions = agent.decideEntitiesTrace();
  const expected = expectations.fixtures[file] ?? [];
  assertExpectedDecisions(file, decisions, expected);
  const entries = agent.recordEntityDecisions({ acceptedOnly: true, topK: cases.length }, new Date("2026-06-30T00:00:00.000Z"));
  const snapshot = serializeDecisionLedger(agent.ledger);
  const replayReport = verifyDecisionLedgerReplay(snapshot, (entry: DecisionLedgerEntry) => replayDecision(bundle, entry), { atol: 1e-6 });

  console.log(`${file}: PASS`);
  console.log(JSON.stringify({
    name: bundle.name,
    hash: bundle.hash,
    rules: loaded.program.rules.length,
    predicates: bundle.predicates.length,
    decisions: decisions.map((decision) => ({
      entityId: decision.entityId,
      action: decision.action,
      accepted: decision.accepted,
      score: Number(decision.score.toFixed(4))
    })),
    expectations: expected.length,
    ledgerEntries: entries.length,
    replay: replayReport.ok
  }, null, 2));

  if (!verifyPolicyBundleHash(bundle) || loaded.program.rules.length === 0 || decisions.length !== cases.length || expected.length === 0 || entries.length === 0 || !replayReport.ok) {
    throw new Error(`Policy fixture failed: ${file}`);
  }
}

function readBundle(file: string): SerializedPolicyBundle {
  const value = JSON.parse(readFileSync(new URL(file, policiesDir), "utf8")) as unknown;
  if (!isSerializedPolicyBundle(value)) {
    throw new Error(`Invalid policy fixture: ${file}`);
  }
  return value;
}

function readExpectations(): ExpectedDecisionFile {
  const value = JSON.parse(readFileSync(new URL("expected-decisions.json", policiesDir), "utf8")) as ExpectedDecisionFile;
  if (value.schemaVersion !== "symtorch.policyExpectations.v1") {
    throw new Error("Invalid expected decision fixture schema.");
  }
  return value;
}

function assertExpectedDecisions(
  file: string,
  decisions: ReturnType<ReturnType<typeof createPolicyAgent>["decideEntitiesTrace"]>,
  expected: readonly ExpectedDecision[]
): void {
  if (expected.length === 0) {
    throw new Error(`No expected decisions registered for ${file}.`);
  }
  for (const item of expected) {
    const decision = decisions[item.rank];
    if (!decision) throw new Error(`Missing rank ${item.rank} decision for ${file}.`);
    if (
      decision.entityId !== item.entityId ||
      decision.action !== item.action ||
      decision.accepted !== item.accepted ||
      decision.score < item.scoreMin ||
      decision.score > item.scoreMax
    ) {
      throw new Error(`Expected decision drift in ${file} at rank ${item.rank}.`);
    }
  }
}

function casesFor(bundle: SerializedPolicyBundle): FixtureCase[] {
  const scenarioId = bundle.metadata.scenarioId;
  if (typeof scenarioId !== "string" || !fixtureCases[scenarioId]) {
    throw new Error(`No fixture cases registered for policy ${bundle.name}.`);
  }
  return fixtureCases[scenarioId];
}

function replayDecision(bundle: SerializedPolicyBundle, entry: DecisionLedgerEntry) {
  const replayAgent = createPolicyAgent(bundle, { threshold: entry.decision.threshold });
  if (entry.kind === "entity" && "entityId" in entry.decision) {
    replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
    return replayAgent.decideEntityTrace(entry.decision.entityId);
  }
  replayAgent.observe(entry.context);
  return replayAgent.decideTrace();
}
