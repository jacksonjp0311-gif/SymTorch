# Holographic Memory

SymTorch `0.4.0` adds a small vector-symbolic memory primitive for agent experiments.

At the tensor level:

- `circularConvolve(a, b)` binds two rank-1 vectors.
- `circularCorrelate(trace, role)` approximately unbinds a role from a trace.
- `bind` and `unbind` are aliases for those operations.

At the agent level:

- `HolographicMemory` superposes bound role/value vectors into one trace.
- `recall(role)` returns an approximate value vector.
- `snapshot()` returns JSON-safe trace metadata and vector contents.

This is not cleanup memory, database storage, or a guarantee of exact symbolic retrieval. It is a differentiable substrate for compact role/value binding experiments.

Run:

```powershell
pnpm demo:memory
```
