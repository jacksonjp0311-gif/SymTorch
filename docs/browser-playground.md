# Browser Playground Prep

The browser playground is a minimal Vite example for the public agent-policy demo.

Run it from the repository root:

```powershell
pnpm playground:browser
```

Run the focused playground regression tests:

```powershell
pnpm playground:test
```

Run the production preview smoke test:

```powershell
pnpm playground:smoke
```

Run the browser interaction E2E test:

```powershell
pnpm playground:e2e
```

The first screen is the actual tool, not a landing page. It supports:

- switching between bundled case escalation, fraud review, and support routing scenarios
- editing a readable rule
- validating rule syntax and predicate bindings
- changing entity-scoped facts with sliders
- training a `ThresholdPredicate` for `high_risk(X)`
- editing the small supervised training set used by the threshold demo
- viewing threshold, score, and loss movement after training
- ranking serialized entity decisions
- recording accepted top-K decisions into the in-memory ledger
- inspecting JSON-safe traces and ledger replay data
- preserving rule edits, facts, and trained threshold across refreshes
- importing and exporting versioned playground state JSON
- importing and exporting standalone `symtorch.scenario.v1` scenario contracts

Current scope:

- Uses CPU-backed SymTorch packages.
- Ships a small scenario catalog for agent-policy demos.
- Uses fixed `FactPredicate` inputs for entity decision ranking.
- Uses a trainable `ThresholdPredicate` in the training panel.
- Uses a small editable in-memory training dataset.
- Persists playground state in browser local storage.
- Imports and exports rules, facts, training examples, and threshold state with the `symtorch.playground.v1` state schema.
- Imports and exports portable scenario definitions with the `symtorch.scenario.v1` contract.
- Does not persist ledger entries outside memory.
- CI verifies the production build and a Vite preview smoke test.
- CI verifies browser interactions with Playwright.

Next browser step:

- Add policy replay against saved ledgers.
