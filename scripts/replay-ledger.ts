import { RuleAgent, verifyDecisionLedgerReplay, type DecisionLedgerEntry, type SerializedAgentDecision, type SerializedEntityDecision } from "@symtorch/agent";
import { FileDecisionLedgerSink } from "@symtorch/agent/node";
import { FactPredicate, FuzzyRuleEngine, PredicateRegistry, RuleProgram } from "@symtorch/logic";

type ReplayArgs = {
  ledger: string;
  program: string;
  predicates: string[];
  threshold: number;
  atol: number;
  rtol: number;
  json: boolean;
};

const args = parseArgs(process.argv.slice(2));
const program = new RuleProgram(args.program);
const registry = args.predicates.reduce(
  (current, name) => current.register(new FactPredicate(name)),
  new PredicateRegistry()
);
const engine = new FuzzyRuleEngine(registry);
const sink = new FileDecisionLedgerSink(args.ledger);
const snapshot = await sink.read();

const report = verifyDecisionLedgerReplay(
  snapshot,
  (entry) => replayDecision(entry, program, engine, args.threshold),
  { atol: args.atol, rtol: args.rtol }
);

if (args.json) {
  console.log(JSON.stringify(report, null, 2));
} else if (report.ok) {
  console.log(`SymTorch ledger replay: PASS (${report.checked} decisions checked)`);
} else {
  console.error(`SymTorch ledger replay: FAIL (${report.mismatches.length}/${report.checked} mismatched)`);
  for (const mismatch of report.mismatches) {
    console.error(`- ${mismatch.entryId}: ${mismatch.reason}`);
    console.error(`  expected action=${mismatch.expected.action} score=${mismatch.expected.score}`);
    console.error(`  actual   action=${mismatch.actual.action} score=${mismatch.actual.score}`);
  }
}

if (!report.ok) process.exitCode = 1;

function replayDecision(
  entry: DecisionLedgerEntry,
  program: RuleProgram,
  engine: FuzzyRuleEngine,
  threshold: number
): SerializedAgentDecision | SerializedEntityDecision {
  const agent = new RuleAgent(program, engine, threshold);
  if (entry.kind === "entity" && "entityId" in entry.decision) {
    agent.memory.observeEntity(entry.decision.entityId, entry.context);
    return agent.decideEntityTrace(entry.decision.entityId);
  }
  agent.observe(entry.context);
  return agent.decideTrace();
}

function parseArgs(argv: string[]): ReplayArgs {
  const ledger = readOption(argv, "--ledger");
  const program = readOption(argv, "--program");
  const predicateList = readOption(argv, "--predicates");
  const thresholdText = readOption(argv, "--threshold") ?? "0.5";
  const atolText = readOption(argv, "--atol") ?? "0";
  const rtolText = readOption(argv, "--rtol") ?? "0";
  if (!ledger || !program || !predicateList) {
    printUsage();
    process.exit(2);
  }
  const threshold = Number(thresholdText);
  if (!Number.isFinite(threshold)) {
    throw new Error(`Expected --threshold to be finite, received ${thresholdText}.`);
  }
  const atol = Number(atolText);
  const rtol = Number(rtolText);
  if (!Number.isFinite(atol) || atol < 0) throw new Error(`Expected --atol to be a non-negative finite number, received ${atolText}.`);
  if (!Number.isFinite(rtol) || rtol < 0) throw new Error(`Expected --rtol to be a non-negative finite number, received ${rtolText}.`);
  const predicates = predicateList.split(",").map((item) => item.trim()).filter(Boolean);
  if (predicates.length === 0) throw new Error("Expected --predicates to include at least one predicate name.");
  return {
    ledger,
    program,
    predicates,
    threshold,
    atol,
    rtol,
    json: argv.includes("--json")
  };
}

function readOption(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1) return undefined;
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Expected a value after ${name}.`);
  return value;
}

function printUsage(): void {
  console.error(`Usage:
  pnpm ledger:replay -- --ledger ./ledger.json --program "escalate(X) :- high_risk(X), not approved(X)." --predicates high_risk,approved [--threshold 0.5] [--atol 0] [--rtol 0] [--json]
`);
}
