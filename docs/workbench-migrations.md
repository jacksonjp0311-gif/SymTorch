# Workbench Migrations

SymTorch `0.25.0` adds explicit migration helpers for browser workbench artifacts.

The goal is modest but important: saved local artifacts should either upgrade into the current contract or fail with structured diagnostics. The workbench should not silently accept malformed state.

## Current Contracts

- `symtorch.playground.v1` for browser playground state.
- `symtorch.policyLibrary.v1` for saved local policy bundle libraries.
- `symtorch.policyBundle.v1` for executable policy bundles.

## Migration Helpers

The browser playground model exports:

- `migratePlaygroundState(value)`
- `migratePolicyBundleLibrary(value)`

Both return:

```ts
type MigrationResult<T> =
  | { ok: true; migrated: boolean; value: T; diagnostics: [] }
  | { ok: false; migrated: false; value: null; diagnostics: ScenarioValidationDiagnostic[] };
```

## Supported Upgrade Paths

- `symtorch.playground.v1` state without `policyLibrary` is upgraded with an empty `symtorch.policyLibrary.v1`.
- `symtorch.playground.v1` state without `trainingExamples` is upgraded with default training examples.
- A bare array of saved bundles is upgraded into `symtorch.policyLibrary.v1`.

## Non-Claims

This is not a remote migration runner, database migration system, or compatibility promise for arbitrary future schemas. It is a local-first guardrail for persisted workbench artifacts.
