import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  DECISION_LEDGER_SCHEMA_VERSION,
  isSerializedDecisionLedger,
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
