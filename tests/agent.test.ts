import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  AGENT_DECISION_SCHEMA_VERSION,
  DECISION_LEDGER_SCHEMA_VERSION,
  DecisionLedger,
  HolographicMemory,
  isSerializedAgentDecision,
  isSerializedDecisionLedger,
  isSerializedEntityDecision,
  loadDecisionLedger,
  RuleAgent,
  serializeDecisionLedger,
  vectorSymbol,
  verifyDecisionLedgerReplay,
  type DecisionLedgerEntry,
  type SerializedAgentDecision,
  type SerializedEntityDecision
} from "@symtorch/agent";
import { FileDecisionLedgerSink } from "@symtorch/agent/node";
import { EXPLANATION_SCHEMA_VERSION, FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

describe("@symtorch/agent", () => {
  it("uses fact-store working memory for rule decisions", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X), not approved(X).");
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.9, approved: 0.2 });

    const decision = agent.decide();
    expect(decision.action).toBe("escalate(X)");
    expect(decision.results[0]?.explanation.rules[0]?.predicates[0]?.detail?.key).toBe("high_risk");
  });

  it("selects actions after aggregating same-head rules", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X).
      escalate(X) :- customer_vip(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("customer_vip"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.6);

    agent.observe({ high_risk: 0.4, customer_vip: 0.5 });

    const decision = agent.decide();
    expect(decision.action).toBe("escalate(X)");
    expect(decision.results[0]?.score.item()).toBeCloseTo(0.7, 5);
    expect(decision.results[0]?.explanation.ruleCount).toBe(2);
  });

  it("returns a stable serialized decision contract", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      defer(X) :- approved(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.9, approved: 0.2 });

    const decision = agent.decideTrace();
    const roundTrip = JSON.parse(JSON.stringify(decision));

    expect(decision).toMatchObject({
      schemaVersion: AGENT_DECISION_SCHEMA_VERSION,
      action: "escalate(X)",
      selectedHead: "escalate(X)",
      threshold: 0.5,
      accepted: true
    });
    expect(isSerializedAgentDecision(decision)).toBe(true);
    expect(decision.score).toBeCloseTo(0.72, 5);
    expect(decision.trace?.schemaVersion).toBe(EXPLANATION_SCHEMA_VERSION);
    expect(decision.trace?.type).toBe("aggregate");
    expect(decision.trace?.rules[0]?.predicates[0]?.detail).toEqual({ key: "high_risk" });
    expect(decision.results.map((result) => result.head)).toEqual(["escalate(X)", "defer(X)"]);
    expect(roundTrip).toEqual(decision);
  });

  it("keeps selected trace while returning no_action below threshold", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.8);

    agent.observe({ high_risk: 0.4 });

    const decision = agent.decideTrace();

    expect(decision.action).toBe("no_action");
    expect(decision.selectedHead).toBe("escalate(X)");
    expect(decision.accepted).toBe(false);
    expect(decision.score).toBeCloseTo(0.4, 5);
    expect(decision.trace?.head).toBe("escalate(X)");
  });

  it("supports entity-scoped working memory snapshots", () => {
    const agent = new RuleAgent(new RuleProgram("noop(X)."), new FuzzyRuleEngine(() => { throw new Error("unused"); }));
    agent.memory.observeEntity("case-1", { risk: 0.8, approved: 0.1 });
    agent.memory.observeEntity("case-2", { risk: 0.2, approved: 0.9 });

    expect(agent.memory.entitySnapshot("case-1")).toMatchObject({ entity: "case-1", risk: 0.8, approved: 0.1 });
    expect(agent.memory.entitySnapshot("case-2")).toMatchObject({ entity: "case-2", risk: 0.2, approved: 0.9 });
    expect(agent.memory.entityIds()).toEqual(["case-1", "case-2"]);
  });

  it("returns ranked serialized decisions for entity batches", () => {
    const program = new RuleProgram(`
      escalate(X) :- high_risk(X), not approved(X).
      defer(X) :- approved(X).
    `);
    const registry = new PredicateRegistry()
      .register(new FactPredicate("high_risk"))
      .register(new FactPredicate("approved"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-low", { high_risk: 0.2, approved: 0.1 });
    agent.memory.observeEntity("case-approved", { high_risk: 0.9, approved: 0.95 });
    agent.memory.observeEntity("case-hot", { high_risk: 0.9, approved: 0.1 });

    const decisions = agent.decideEntitiesTrace();
    const roundTrip = JSON.parse(JSON.stringify(decisions));

    expect(decisions.map((decision) => decision.entityId)).toEqual(["case-approved", "case-hot", "case-low"]);
    expect(decisions[0]).toMatchObject({
      schemaVersion: AGENT_DECISION_SCHEMA_VERSION,
      entityId: "case-approved",
      action: "defer(X)",
      selectedHead: "defer(X)",
      accepted: true
    });
    expect(decisions[1]).toMatchObject({
      entityId: "case-hot",
      action: "escalate(X)",
      selectedHead: "escalate(X)",
      accepted: true
    });
    expect(decisions[1]?.trace?.schemaVersion).toBe(EXPLANATION_SCHEMA_VERSION);
    expect(decisions[1]?.results.map((result) => result.head)).toEqual(["escalate(X)", "defer(X)"]);
    expect(decisions.every(isSerializedEntityDecision)).toBe(true);
    expect(roundTrip).toEqual(decisions);
  });

  it("rejects invalid serialized decision contracts", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.8 });
    const decision = agent.decideTrace();
    const entityDecision = { entityId: "case-hot", ...decision };

    expect(isSerializedAgentDecision(decision)).toBe(true);
    expect(isSerializedEntityDecision(entityDecision)).toBe(true);
    expect(isSerializedAgentDecision({ ...decision, schemaVersion: "symtorch.agentDecision.v0" })).toBe(false);
    expect(isSerializedAgentDecision({ ...decision, score: Number.NaN })).toBe(false);
    expect(isSerializedEntityDecision(decision)).toBe(false);
  });

  it("supports explicit entity batches and preserves below-threshold traces", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.8);

    agent.memory.observeEntity("case-a", { high_risk: 0.4 });
    agent.memory.observeEntity("case-b", { high_risk: 0.6 });
    agent.memory.observeEntity("case-c", { high_risk: 0.1 });

    const decisions = agent.decideEntitiesTrace(["case-a", "case-c"]);

    expect(decisions.map((decision) => decision.entityId)).toEqual(["case-a", "case-c"]);
    expect(decisions[0]).toMatchObject({
      action: "no_action",
      selectedHead: "escalate(X)",
      accepted: false
    });
    expect(decisions[0]?.score).toBeCloseTo(0.4, 5);
    expect(decisions[0]?.trace?.head).toBe("escalate(X)");
  });

  it("filters entity decisions with deterministic ordering options", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-c", { high_risk: 0.7 });
    agent.memory.observeEntity("case-a", { high_risk: 0.7 });
    agent.memory.observeEntity("case-b", { high_risk: 0.3 });
    agent.memory.observeEntity("case-d", { high_risk: 0.9 });

    const ranked = agent.decideEntitiesTrace();
    const filtered = agent.decideEntitiesTrace({ minScore: 0.5, acceptedOnly: true, topK: 2 });
    const explicit = agent.decideEntitiesTrace({ entityIds: ["case-b", "case-a"], minScore: 0.4 });

    expect(ranked.map((decision) => decision.entityId)).toEqual(["case-d", "case-a", "case-c", "case-b"]);
    expect(filtered.map((decision) => decision.entityId)).toEqual(["case-d", "case-a"]);
    expect(filtered.every((decision) => decision.accepted)).toBe(true);
    expect(explicit.map((decision) => decision.entityId)).toEqual(["case-a"]);
  });

  it("records single decisions in an append-only ledger", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.observe({ high_risk: 0.8 });
    const first = agent.recordDecision(new Date("2026-06-29T12:00:00.000Z"));
    agent.observe({ high_risk: 0.1 });
    const second = agent.recordDecision(new Date("2026-06-29T12:01:00.000Z"));

    expect(first).toMatchObject({
      id: "decision-1",
      createdAt: "2026-06-29T12:00:00.000Z",
      kind: "agent",
      context: { high_risk: 0.8 }
    });
    expect(first.decision.action).toBe("escalate(X)");
    expect(second.id).toBe("decision-2");
    expect(second.decision.action).toBe("no_action");
    expect(agent.ledger.all().map((entry) => entry.id)).toEqual(["decision-1", "decision-2"]);
    expect(agent.ledger.all()[0]?.context).toEqual({ high_risk: 0.8 });
  });

  it("records ranked entity decisions with replayable snapshots", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-low", { high_risk: 0.2 });
    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });

    const entries = agent.recordEntityDecisions({ acceptedOnly: true }, new Date("2026-06-29T13:00:00.000Z"));
    agent.memory.observeEntity("case-hot", { high_risk: 0.1 });
    const replay = agent.ledger.all();

    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      id: "decision-1",
      createdAt: "2026-06-29T13:00:00.000Z",
      kind: "entity",
      context: { entity: "case-hot", high_risk: 0.9 }
    });
    expect(entries[0]?.decision).toMatchObject({
      entityId: "case-hot",
      action: "escalate(X)",
      accepted: true
    });
    expect(replay[0]?.context).toEqual({ entity: "case-hot", high_risk: 0.9 });
    expect(JSON.parse(JSON.stringify(replay))).toEqual(replay);
  });

  it("emits decision and ledger observer events", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const events: unknown[] = [];
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5, {
      observer: {
        onDecision: (event) => events.push(event),
        onLedgerAppend: (event) => events.push(event)
      }
    });

    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });
    agent.recordEntityDecision("case-hot", new Date("2026-06-29T15:00:00.000Z"));

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      kind: "agent.decision",
      entityId: "case-hot",
      action: "escalate(X)",
      selectedHead: "escalate(X)",
      accepted: true,
      resultCount: 1
    });
    expect(events[1]).toMatchObject({
      kind: "ledger.append",
      entryId: "decision-1",
      decisionKind: "entity",
      action: "escalate(X)",
      accepted: true,
      contextKeys: ["entity", "high_risk"]
    });
  });

  it("serializes and restores versioned decision ledger snapshots", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });
    agent.recordEntityDecision("case-hot", new Date("2026-06-29T14:00:00.000Z"));

    const snapshot = serializeDecisionLedger(agent.ledger);
    const restored = loadDecisionLedger(new DecisionLedger(), JSON.parse(JSON.stringify(snapshot)));

    expect(snapshot.schemaVersion).toBe(DECISION_LEDGER_SCHEMA_VERSION);
    expect(isSerializedDecisionLedger(snapshot)).toBe(true);
    expect(restored.snapshot()).toEqual(snapshot);

    restored.append({
      kind: "agent",
      context: { high_risk: 0.2 },
      decision: {
        schemaVersion: AGENT_DECISION_SCHEMA_VERSION,
        action: "no_action",
        selectedHead: "escalate(X)",
        score: 0.2,
        threshold: 0.5,
        accepted: false,
        trace: null,
        results: []
      }
    }, new Date("2026-06-29T14:01:00.000Z"));

    expect(restored.all().map((entry) => entry.id)).toEqual(["decision-1", "decision-2"]);
  });

  it("verifies decision ledger replay against current policy behavior", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });
    agent.recordEntityDecision("case-hot", new Date("2026-06-29T14:00:00.000Z"));
    const snapshot = serializeDecisionLedger(agent.ledger);
    const replay = (entry: DecisionLedgerEntry): SerializedAgentDecision | SerializedEntityDecision => {
      const replayAgent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
      if (entry.kind === "entity" && "entityId" in entry.decision) {
        replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
        return replayAgent.decideEntityTrace(entry.decision.entityId);
      }
      replayAgent.observe(entry.context);
      return replayAgent.decideTrace();
    };

    const report = verifyDecisionLedgerReplay(snapshot, replay);
    const changed = JSON.parse(JSON.stringify(snapshot));
    changed.entries[0].decision.score = 0.1;
    const mismatch = verifyDecisionLedgerReplay(changed, replay);

    expect(report).toEqual({ ok: true, checked: 1, mismatches: [] });
    expect(mismatch.ok).toBe(false);
    expect(mismatch.checked).toBe(1);
    expect(mismatch.mismatches[0]).toMatchObject({
      entryId: "decision-1",
      reason: "decision mismatch"
    });
  });

  it("verifies decision replay with configured tolerance for float drift", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });
    agent.recordEntityDecision("case-hot", new Date("2026-06-29T14:00:00.000Z"));
    const snapshot = serializeDecisionLedger(agent.ledger);
    const replay = (entry: DecisionLedgerEntry): SerializedAgentDecision | SerializedEntityDecision => {
      const replayAgent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
      if (entry.kind === "entity" && "entityId" in entry.decision) {
        replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
        return replayAgent.decideEntityTrace(entry.decision.entityId);
      }
      replayAgent.observe(entry.context);
      return replayAgent.decideTrace();
    };

    const exactReport = verifyDecisionLedgerReplay(snapshot, replay);
    expect(exactReport.ok).toBe(true);

    const drifted = JSON.parse(JSON.stringify(snapshot));
    drifted.entries[0].decision.score = 0.905;

    const strictReport = verifyDecisionLedgerReplay(drifted, replay);
    expect(strictReport.ok).toBe(false);

    const tolerantReport = verifyDecisionLedgerReplay(drifted, replay, { atol: 0.01 });
    expect(tolerantReport.ok).toBe(true);
    expect(tolerantReport.checked).toBe(1);
    expect(tolerantReport.mismatches).toHaveLength(0);

    const actionDrifted = JSON.parse(JSON.stringify(snapshot));
    actionDrifted.entries[0].decision.action = "wrong_action";
    const actionReport = verifyDecisionLedgerReplay(actionDrifted, replay, { atol: 1.0 });
    expect(actionReport.ok).toBe(false);
    expect(actionReport.mismatches[0]?.reason).toBe("decision mismatch");
  });

  it("emits replay observer summaries", () => {
    const program = new RuleProgram("escalate(X) :- high_risk(X).");
    const registry = new PredicateRegistry().register(new FactPredicate("high_risk"));
    const agent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);

    agent.memory.observeEntity("case-hot", { high_risk: 0.9 });
    agent.recordEntityDecision("case-hot", new Date("2026-06-29T14:00:00.000Z"));
    const snapshot = serializeDecisionLedger(agent.ledger);
    const replay = (entry: DecisionLedgerEntry): SerializedAgentDecision | SerializedEntityDecision => {
      const replayAgent = new RuleAgent(program, new FuzzyRuleEngine(registry), 0.5);
      if (entry.kind === "entity" && "entityId" in entry.decision) {
        replayAgent.memory.observeEntity(entry.decision.entityId, entry.context);
        return replayAgent.decideEntityTrace(entry.decision.entityId);
      }
      replayAgent.observe(entry.context);
      return replayAgent.decideTrace();
    };
    const events: unknown[] = [];

    const report = verifyDecisionLedgerReplay(snapshot, replay, {
      observer: {
        onReplay: (event) => events.push(event)
      }
    });

    expect(report.ok).toBe(true);
    expect(events).toEqual([{
      kind: "ledger.replay",
      ok: true,
      checked: 1,
      mismatchCount: 0
    }]);
  });

  it("persists decision ledger snapshots through the node file sink", async () => {
    const dir = await mkdtemp(join(tmpdir(), "symtorch-ledger-"));
    try {
      const sink = new FileDecisionLedgerSink(join(dir, "ledger.json"));
      const ledger = new DecisionLedger();
      ledger.append({
        kind: "agent",
        context: { high_risk: 0.8 },
        decision: {
          schemaVersion: AGENT_DECISION_SCHEMA_VERSION,
          action: "escalate(X)",
          selectedHead: "escalate(X)",
          score: 0.8,
          threshold: 0.5,
          accepted: true,
          trace: null,
          results: []
        }
      }, new Date("2026-06-29T14:02:00.000Z"));

      await sink.write(ledger.snapshot());
      const restored = await sink.read();

      expect(restored).toEqual(ledger.snapshot());
      expect(isSerializedDecisionLedger(restored)).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("rejects invalid decision ledger snapshots", () => {
    const ledger = new DecisionLedger();

    expect(isSerializedDecisionLedger({ schemaVersion: "old", entries: [] })).toBe(false);
    expect(isSerializedDecisionLedger({
      schemaVersion: DECISION_LEDGER_SCHEMA_VERSION,
      entries: [{ id: "decision-1", createdAt: "now", kind: "entity", context: {}, decision: { action: "bad" } }]
    })).toBe(false);
    expect(() => ledger.load({ schemaVersion: "old", entries: [] } as never)).toThrow(DECISION_LEDGER_SCHEMA_VERSION);
  });

  it("binds and recalls vector-symbolic memory traces", () => {
    const memory = new HolographicMemory(5);
    const riskRole = vectorSymbol([1, 0, 0, 0, 0]);
    const approvedRole = vectorSymbol([0, 1, 0, 0, 0]);
    const highRisk = vectorSymbol([0.8, 0.1, 0.1, 0, 0]);
    const approved = vectorSymbol([0.1, 0.9, 0, 0, 0]);

    memory.bind(riskRole, highRisk);
    memory.bind(approvedRole, approved);

    const recalledRisk = memory.recall(riskRole);
    const recalledApproved = memory.recall(approvedRole);
    expect(memory.snapshot()).toMatchObject({ dimension: 5, bindings: 2 });
    expect(memory.similarity(recalledRisk, highRisk)).toBeGreaterThan(memory.similarity(recalledRisk, approved));
    expect(memory.similarity(recalledApproved, approved)).toBeGreaterThan(memory.similarity(recalledApproved, highRisk));
  });

  it("clears holographic memory traces", () => {
    const memory = new HolographicMemory(3);
    memory.bind(vectorSymbol([1, 0, 0]), vectorSymbol([0.2, 0.4, 0.6]));
    expect(memory.snapshot().bindings).toBe(1);

    memory.clear();

    expect(memory.snapshot()).toEqual({
      dimension: 3,
      bindings: 0,
      vector: [0, 0, 0]
    });
  });
});
