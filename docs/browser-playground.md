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

The first screen is the actual tool, not a landing page. It supports:

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

Current scope:

- Uses CPU-backed SymTorch packages.
- Uses fixed `FactPredicate` inputs for entity decision ranking.
- Uses a trainable `ThresholdPredicate` in the training panel.
- Uses a small editable in-memory training dataset.
- Persists playground state in browser local storage.
- Imports and exports rules, facts, training examples, and threshold state with the `symtorch.playground.v1` schema.
- Does not persist ledger entries outside memory.

Next browser step:

- Add end-to-end browser interaction tests once the UI surface stabilizes.
