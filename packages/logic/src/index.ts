import { mul, sub, Tensor, tensor } from "@symtorch/core";

export type Term = {
  kind: "variable" | "constant";
  name: string;
};

export type PredicateCall = {
  name: string;
  terms: readonly Term[];
  negated: boolean;
};

export type RuleAst = {
  head: PredicateCall;
  body: readonly PredicateCall[];
  source: string;
};

export type PredicateContext = Record<string, unknown>;
export type PredicateResolver = (call: PredicateCall, context: PredicateContext) => Tensor;

export type RuleExplanation = {
  rule: string;
  head: string;
  score: number;
  predicates: PredicateTrace[];
};

export type PredicateTrace = {
  name: string;
  negated: boolean;
  value: number;
  contribution: number;
};

export type RuleResult = {
  score: Tensor;
  explanation: RuleExplanation;
};

export class RuleProgram {
  readonly rules: readonly RuleAst[];

  constructor(source: string) {
    this.rules = parseProgram(source);
  }
}

export class FuzzyRuleEngine {
  constructor(private readonly resolver: PredicateResolver) {}

  evaluate(rule: RuleAst, context: PredicateContext = {}): RuleResult {
    let score = tensor(1);
    const traces: PredicateTrace[] = [];
    for (const call of rule.body) {
      const raw = this.resolver(call, context);
      const value = call.negated ? sub(1, raw) : raw;
      score = mul(score, value);
      traces.push({
        name: formatPredicate(call),
        negated: call.negated,
        value: raw.item(),
        contribution: value.item()
      });
    }
    return {
      score,
      explanation: {
        rule: rule.source,
        head: formatPredicate(rule.head),
        score: score.item(),
        predicates: traces
      }
    };
  }

  evaluateProgram(program: RuleProgram, context: PredicateContext = {}): RuleResult[] {
    return program.rules.map((rule) => this.evaluate(rule, context));
  }
}

export function parseProgram(source: string): RuleAst[] {
  return source
    .split(".")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => parseRule(`${chunk}.`));
}

export function parseRule(source: string): RuleAst {
  const normalized = source.trim().replace(/\.$/, "");
  const [headText, bodyText] = normalized.split(":-").map((part) => part.trim());
  if (!headText) throw new Error(`Rule is missing a head: ${source}`);
  const head = parsePredicate(headText);
  const body = bodyText ? splitTopLevel(bodyText).map(parsePredicate) : [];
  return { head, body, source: source.trim() };
}

export function productAnd(values: readonly Tensor[]): Tensor {
  return values.reduce((acc, value) => mul(acc, value), tensor(1));
}

export function probabilisticOr(a: Tensor, b: Tensor): Tensor {
  return sub(sub(a, mul(a, b)), sub(tensor(0), b));
}

export function fuzzyNot(value: Tensor): Tensor {
  return sub(1, value);
}

export function formatPredicate(call: PredicateCall): string {
  const prefix = call.negated ? "not " : "";
  return `${prefix}${call.name}(${call.terms.map((term) => term.name).join(", ")})`;
}

function parsePredicate(text: string): PredicateCall {
  const trimmed = text.trim();
  const negated = trimmed.startsWith("not ");
  const body = negated ? trimmed.slice(4).trim() : trimmed;
  const match = /^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$/.exec(body);
  if (!match) throw new Error(`Invalid predicate call: ${text}`);
  const name = match[1];
  if (!name) throw new Error(`Invalid predicate name: ${text}`);
  const terms = (match[2] ?? "")
    .split(",")
    .map((term) => term.trim())
    .filter(Boolean)
    .map(parseTerm);
  return { name, terms, negated };
}

function parseTerm(text: string): Term {
  const name = text.trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error(`Invalid term: ${text}`);
  return {
    kind: /^[A-Z_]/.test(name) ? "variable" : "constant",
    name
  };
}

function splitTopLevel(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === "(") depth += 1;
    if (char === ")") depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(text.slice(start).trim());
  return parts.filter(Boolean);
}
