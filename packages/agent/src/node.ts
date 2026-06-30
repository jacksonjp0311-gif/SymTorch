import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  DECISION_LEDGER_SCHEMA_VERSION,
  isSerializedAgentDecision,
  isSerializedDecisionLedger,
  isSerializedEntityDecision,
  type DecisionLedgerAppendSink,
  type DecisionLedgerEntry,
  type DecisionLedgerSink,
  type SerializedDecisionLedger
} from "./index.js";

export class FileDecisionLedgerSink implements DecisionLedgerSink {
  constructor(readonly path: string) {}

  async write(snapshot: SerializedDecisionLedger): Promise<void> {
    if (!isSerializedDecisionLedger(snapshot)) {
      throw new Error(`Expected ${DECISION_LEDGER_SCHEMA_VERSION} snapshot.`);
    }
    await mkdir(dirname(this.path), { recursive: true });
    await writeFile(this.path, `${JSON.stringify(snapshot, null, 2)}\n`, "utf8");
  }

  async read(): Promise<SerializedDecisionLedger> {
    const parsed = JSON.parse(await readFile(this.path, "utf8")) as unknown;
    if (!isSerializedDecisionLedger(parsed)) {
      throw new Error(`Expected ${DECISION_LEDGER_SCHEMA_VERSION} snapshot.`);
    }
    return parsed;
  }
}

export class AppendFileDecisionLedgerSink implements DecisionLedgerAppendSink {
  constructor(readonly path: string) {}

  async append(entry: DecisionLedgerEntry): Promise<void> {
    if (!isDecisionLedgerEntry(entry)) {
      throw new Error(`Expected ${DECISION_LEDGER_SCHEMA_VERSION} ledger entry.`);
    }
    await mkdir(dirname(this.path), { recursive: true });
    await appendFile(this.path, `${JSON.stringify(entry)}\n`, "utf8");
  }

  async write(snapshot: SerializedDecisionLedger): Promise<void> {
    if (!isSerializedDecisionLedger(snapshot)) {
      throw new Error(`Expected ${DECISION_LEDGER_SCHEMA_VERSION} snapshot.`);
    }
    await mkdir(dirname(this.path), { recursive: true });
    await writeFile(this.path, snapshot.entries.map((entry) => JSON.stringify(entry)).join("\n") + "\n", "utf8");
  }

  async read(): Promise<SerializedDecisionLedger> {
    const text = await readFile(this.path, "utf8");
    const entries = text
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as unknown);
    if (!entries.every(isDecisionLedgerEntry)) {
      throw new Error(`Expected ${DECISION_LEDGER_SCHEMA_VERSION} ledger entries.`);
    }
    return {
      schemaVersion: DECISION_LEDGER_SCHEMA_VERSION,
      entries
    };
  }
}

function isDecisionLedgerEntry(value: unknown): value is DecisionLedgerEntry {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  const kind = record.kind;
  const decision = record.decision;
  return (
    typeof record.id === "string" &&
    typeof record.createdAt === "string" &&
    (kind === "agent" || kind === "entity") &&
    typeof record.context === "object" &&
    record.context !== null &&
    (kind === "entity" ? isSerializedEntityDecision(decision) : isSerializedAgentDecision(decision))
  );
}
