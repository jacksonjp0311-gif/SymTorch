# Scenario Contract

`symtorch.scenario.v1` is the browser workbench contract for portable agent-policy scenarios.

It is intentionally small:

```json
{
  "schemaVersion": "symtorch.scenario.v1",
  "id": "case-escalation",
  "title": "Case Escalation",
  "description": "Escalate risky unapproved cases while deferring approved ones.",
  "ruleSource": "escalate(X) :- high_risk(X), not approved(X).\ndefer(X) :- approved(X).",
  "cases": [
    { "entityId": "case-hot", "high_risk": 0.9, "approved": 0.1 }
  ],
  "trainingExamples": [
    { "risk": 0.75, "approved": 0.05, "label": 1 }
  ],
  "trainedThreshold": 0.9
}
```

Validation checks:

- schema version
- non-empty scenario identity fields
- valid rule syntax and registered predicate bindings
- case rows with `entityId`, `high_risk`, and `approved`
- training rows with `risk`, `approved`, and `label`
- finite threshold values

The contract is not a security boundary. It is an authoring and interchange format for demos, tests, and future agent-policy tooling.

Training results are stored separately in playground state as `symtorch.trainingRun.v1` records. Scenario contracts describe the starting policy and data; training-run records describe what happened after local fitting.
