import {
  createPolicyAgent,
  serializeDecisionLedger,
  verifyDecisionLedgerReplay,
  type DecisionLedgerEntry,
  type SerializedAgentDecision,
  type SerializedEntityDecision
} from "@symtorch/agent";
import {
  createPolicyBundle,
  loadPolicyBundle,
  verifyPolicyBundleHash
} from "@symtorch/logic";

const bundle = createPolicyBundle({
  name: "Escalation Policy",
  version: "2026.06.30",
  rules: "escalate(X) :- high_risk(X), not approved(X).",
  predicates: [
    { kind: "threshold", name: "high_risk", valueKey: "risk", threshold: 0.7, slope: 10 },
    { kind: "fact", name: "approved" }
  ],
  metadata: {
    owner: "risk",
    purpose: "demo"
  }
});

const loaded = loadPolicyBundle(bundle);
const agent = createPolicyAgent(bundle, {
  threshold: 0.5,
  limits: {
    maxRuleSourceLength: 500,
    maxEntitiesPerBatch: 10,
    maxReplayEntries: 10
  }
});

agent.memory.observeEntity("case-hot", { risk: 0.9, approved: 0.1 });
const decision = agent.decideEntityTrace("case-hot");
const entry = agent.recordEntityDecision("case-hot", new Date("2026-06-30T00:00:00.000Z"));
const snapshot = serializeDecisionLedger(agent.ledger);

const replay = (record: DecisionLedgerEntry): SerializedAgentDecision | SerializedEntityDecision => {
  const replayAgent = createPolicyAgent(bundle, { threshold: record.decision.threshold });
  if (record.kind === "entity" && "entityId" in record.decision) {
    replayAgent.memory.observeEntity(record.decision.entityId, record.context);
    return replayAgent.decideEntityTrace(record.decision.entityId);
  }
  replayAgent.observe(record.context);
  return replayAgent.decideTrace();
};
const replayReport = verifyDecisionLedgerReplay(snapshot, replay, {
  limits: { maxReplayEntries: 10 },
  atol: 1e-6
});

console.log("SymTorch Policy Bundle Demo");
console.log(`bundle: ${bundle.name} ${bundle.version}`);
console.log(`schema: ${bundle.schemaVersion}`);
console.log(`hash: ${bundle.hash}`);
console.log(`hash verified: ${verifyPolicyBundleHash(bundle)}`);
console.log(`loaded rules: ${loaded.program.rules.length}`);
console.log("decision:");
console.log(JSON.stringify({
  entityId: decision.entityId,
  action: decision.action,
  score: decision.score,
  accepted: decision.accepted,
  selectedHead: decision.selectedHead
}, null, 2));
console.log("trace:");
console.log(JSON.stringify(decision.trace, null, 2));
console.log("ledger entry:");
console.log(JSON.stringify({
  id: entry.id,
  kind: entry.kind,
  action: entry.decision.action
}, null, 2));
console.log("replay:");
console.log(JSON.stringify(replayReport, null, 2));

if (!verifyPolicyBundleHash(bundle) || !decision.accepted || !replayReport.ok) {
  throw new Error("Policy bundle demo failed.");
}
