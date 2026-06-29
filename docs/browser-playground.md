# Browser Playground Prep

The browser playground is a minimal Vite example for the public agent-policy demo.

Run it from the repository root:

```powershell
pnpm playground:browser
```

The first screen is the actual tool, not a landing page. It supports:

- editing a readable rule
- validating rule syntax and predicate bindings
- changing entity-scoped facts with sliders
- ranking serialized entity decisions
- recording accepted top-K decisions into the in-memory ledger
- inspecting JSON-safe traces and ledger replay data

Current scope:

- Uses CPU-backed SymTorch packages.
- Uses fixed `FactPredicate` inputs.
- Does not yet train predicates in the browser.
- Does not persist ledger entries outside memory.

Next browser step:

- Add an in-browser training panel for `ThresholdPredicate`.
- Show before/after threshold and score movement.
- Keep the explanation trace visible during training.
