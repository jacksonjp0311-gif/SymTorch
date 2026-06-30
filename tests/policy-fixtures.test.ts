import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createPolicyAgent, serializeDecisionLedger, verifyDecisionLedgerReplay, type DecisionLedgerEntry } from "@symtorch/agent";
import { isSerializedPolicyBundle, loadPolicyBundle, verifyPolicyBundleHash, type SerializedPolicyBundle } from "@symtorch/logic";
import {
  createPolicyLibrary,
  migratePolicyBundleLibrary,
  savePolicyBundleToLibrary
} from "../examples/browser-playground/src/app-model";

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
const policyFiles = readdirSync(policiesDir)
  .filter((file) => file.endsWith(".policy.json"))
  .sort();
const expectations = readExpectations();

describe("policy fixtures", () => {
  it("ships multiple checked-in policies", () => {
    expect(policyFiles).toEqual([
      "escalation.policy.json",
      "fraud-review.policy.json",
      "support-routing.policy.json"
    ]);
  });

  it("ships expected decision coverage for every checked-in policy", () => {
    expect(Object.keys(expectations.fixtures).sort()).toEqual(policyFiles);
    for (const file of policyFiles) {
      expect(expectations.fixtures[file]?.length).toBeGreaterThan(0);
    }
  });

  it.each(policyFiles)("verifies, executes, records, and replays %s", (file) => {
    const bundle = readBundle(file);
    const loaded = loadPolicyBundle(bundle);
    const cases = casesFor(bundle);
    const agent = createPolicyAgent(bundle, { threshold: 0.5 });

    for (const item of cases) {
      agent.memory.observeEntity(item.entityId, {
        high_risk: item.high_risk,
        approved: item.approved
      });
    }

    const decisions = agent.decideEntitiesTrace();
    const expected = expectations.fixtures[file] ?? [];
    const entries = agent.recordEntityDecisions({ acceptedOnly: true, topK: cases.length }, new Date("2026-06-30T00:00:00.000Z"));
    const snapshot = serializeDecisionLedger(agent.ledger);
    const replay = verifyDecisionLedgerReplay(snapshot, (entry: DecisionLedgerEntry) => replayDecision(bundle, entry), { atol: 1e-6 });

    expect(verifyPolicyBundleHash(bundle)).toBe(true);
    expect(loaded.program.rules.length).toBeGreaterThan(0);
    expect(bundle.predicates.length).toBeGreaterThan(0);
    expect(decisions).toHaveLength(cases.length);
    expectExpectedDecisions(decisions, expected);
    expect(entries.length).toBeGreaterThan(0);
    expect(replay.ok).toBe(true);
  });

  it("loads the fixture corpus into a migratable workbench policy library", () => {
    const library = policyFiles.reduce(
      (acc, file, index) => savePolicyBundleToLibrary(acc, readBundle(file), `2026-06-30T00:00:0${index}.000Z`),
      createPolicyLibrary()
    );
    const migration = migratePolicyBundleLibrary(library);

    expect(library.bundles).toHaveLength(policyFiles.length);
    expect(migration.ok).toBe(true);
    expect(migration.ok && migration.value).toEqual(library);
  });
});

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

function expectExpectedDecisions(
  decisions: ReturnType<ReturnType<typeof createPolicyAgent>["decideEntitiesTrace"]>,
  expected: readonly ExpectedDecision[]
): void {
  expect(expected.length).toBeGreaterThan(0);
  for (const item of expected) {
    const decision = decisions[item.rank];
    expect(decision?.entityId).toBe(item.entityId);
    expect(decision?.action).toBe(item.action);
    expect(decision?.accepted).toBe(item.accepted);
    expect(decision?.score).toBeGreaterThanOrEqual(item.scoreMin);
    expect(decision?.score).toBeLessThanOrEqual(item.scoreMax);
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
