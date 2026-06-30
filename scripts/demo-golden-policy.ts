import { readFileSync } from "node:fs";
import { createPolicyAgent, serializeDecisionLedger, verifyDecisionLedgerReplay, type DecisionLedgerEntry } from "@symtorch/agent";
import { isSerializedPolicyBundle, loadPolicyBundle, verifyPolicyBundleHash } from "@symtorch/logic";

const bundlePath = new URL("../examples/policies/escalation.policy.json", import.meta.url);
const bundle = JSON.parse(readFileSync(bundlePath, "utf8")) as unknown;

if (!isSerializedPolicyBundle(bundle)) {
  throw new Error("Golden policy bundle failed schema or hash verification.");
}

const loaded = loadPolicyBundle(bundle);
const agent = createPolicyAgent(bundle, {
  threshold: 0.5,
  limits: {
    maxRuleSourceLength: 500,
    maxEntitiesPerBatch: 10,
    maxReplayEntries: 10
  }
});

agent.memory.observeEntity("case-hot", { high_risk: 0.9, approved: 0.1 });
agent.memory.observeEntity("case-approved", { high_risk: 0.9, approved: 0.95 });

const decisions = agent.decideEntitiesTrace();
const entries = agent.recordEntityDecisions({ acceptedOnly: true, topK: 2 }, new Date("2026-06-30T00:00:00.000Z"));
const snapshot = serializeDecisionLedger(agent.ledger);
const replayReport = verifyDecisionLedgerReplay(snapshot, (entry: DecisionLedgerEntry) => {
  const replayAgent = createPolicyAgent(bundle, { threshold: entry.decision.threshold });
  if (entry.kind === "entity" && "entityId" in entry.decision) {
    replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
    return replayAgent.decideEntityTrace(entry.decision.entityId);
  }
  replayAgent.observe(entry.context);
  return replayAgent.decideTrace();
});

console.log("SymTorch Golden Policy Demo");
console.log(`policy: ${bundle.name} ${bundle.version}`);
console.log(`schema: ${bundle.schemaVersion}`);
console.log(`hash: ${bundle.hash}`);
console.log(`hash verified: ${verifyPolicyBundleHash(bundle)}`);
console.log(`loaded rules: ${loaded.program.rules.length}`);
console.log(`loaded predicates: ${bundle.predicates.length}`);
console.log("top decisions:");
console.log(JSON.stringify(decisions.map((decision) => ({
  entityId: decision.entityId,
  action: decision.action,
  score: decision.score,
  accepted: decision.accepted
})), null, 2));
console.log("ledger:");
console.log(JSON.stringify(entries.map((entry) => ({
  id: entry.id,
  kind: entry.kind,
  action: entry.decision.action
})), null, 2));
console.log("replay:");
console.log(JSON.stringify(replayReport, null, 2));

if (!verifyPolicyBundleHash(bundle) || entries.length === 0 || !replayReport.ok) {
  throw new Error("Golden policy demo failed.");
}
