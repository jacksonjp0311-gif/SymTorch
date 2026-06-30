# SymTorch Publish Readiness Report (v0.30.0)

This report focuses on turning the existing **production contract corpus alpha** into a **shippable** public release.

## Evidence

- `@symtorch/*` packages are **not** on npm yet (HTTP 404):
  - `@symtorch/core` → 404
  - `@symtorch/logic` → 404
  - `@symtorch/nn` → 404
  - `@symtorch/agent` → 404
  - `@symtorch/webgpu` → 404

- Root `package.json` has `"private": true` (fine for monorepos).
- Sub-packages currently have minimal `package.json` files (no `files`, no `publishConfig`, minimal metadata).

---

## Release blockers (must address before publish)

### 1) Scoped package access must be public

Scoped packages default to restricted unless explicitly published public.

**Recommendation (preferred):** add to each `packages/*/package.json`:

```json
{
  "publishConfig": { "access": "public" }
}
```

Alternative: always publish with `--access public`.

---

### 2) Ensure tarballs only include intended artifacts

Right now, without a `files` whitelist (or `.npmignore`), you risk shipping extra files.

**Recommendation:** in each `packages/*/package.json` add:

```json
{
  "files": ["dist"],
  "sideEffects": false
}
```

Notes:
- `sideEffects:false` improves tree-shaking for ESM libs.
- You can also include `"files": ["dist", "README.md", "LICENSE"]` if you want those to ship.

---

### 3) Published package metadata (professional polish)

Add these fields to each `packages/*/package.json` (recommended for credibility):

- `description`
- `license` (MIT)
- `repository`
- `homepage`
- `bugs`
- `keywords`

This makes npm pages look legitimate and improves discoverability.

---

## Root monorepo settings (recommended)

### Keep root `private: true`

This is normal for monorepos and prevents accidental publication of `symtorch-monorepo`.

### Add a root publish helper script (optional)

Proposed addition (requires approval to modify root `package.json`):

```json
{
  "scripts": {
    "publish:dry": "pnpm -r --filter \"./packages/*\" publish --dry-run --access public",
    "publish:real": "pnpm -r --filter \"./packages/*\" publish --access public"
  }
}
```

### Add a release automation workflow (optional)

Add `.github/workflows/release.yml` that triggers on tags `v*`:
- generates GitHub Release notes
- optionally builds and uploads artifacts

This is not required for first ship.

---

## Workspace dependency rewriting

Your packages use `"workspace:*"` for internal deps.

**Expected behavior:** `pnpm publish` rewrites these to the actual version (`0.30.0`).

**Dry-run requirement:** verify in the `--dry-run` output that:
- `@symtorch/logic` depends on `@symtorch/core@0.30.0` and `@symtorch/nn@0.30.0`
- `@symtorch/agent` depends on `@symtorch/core@0.30.0` and `@symtorch/logic@0.30.0`

If it does not rewrite correctly, we should switch to explicit versions in sub-packages.

---

## Proposed patch snippets (NOT APPLIED)

You asked for “concrete diffs” but we cannot modify existing files without explicit approval.

Here is the minimal patch you’ll likely want to apply to **each** `packages/*/package.json`:

```diff
 {
   "name": "@symtorch/core",
   "version": "0.30.0",
   "type": "module",
   "main": "./dist/index.js",
   "types": "./dist/index.d.ts",
+  "license": "MIT",
+  "publishConfig": { "access": "public" },
+  "files": ["dist"],
+  "sideEffects": false,
   "exports": {
     ".": {
       "types": "./dist/index.d.ts",
       "import": "./dist/index.js"
     }
   }
 }
```

You’d replicate the same structure for `logic`, `nn`, `agent`, `webgpu` (with appropriate `description` + `keywords`).

---

## Approval checklist (explicit gates)

When you’re ready, I’ll ask you to approve each of these **separately**:

1) Modify `packages/*/package.json` to add `publishConfig`, `files`, metadata.
2) (Optional) Add root scripts (`publish:dry`, `publish:real`).
3) Create git tag `v0.30.0` and push it.
4) Publish to npm.
5) Create GitHub Release.

---

## Suggested order of operations

1) Implement `files` + `publishConfig` + metadata
2) Run `npm pack` inspection per package
3) Run `pnpm -r publish --dry-run --access public`
4) Tag `v0.30.0`
5) Publish
6) Release notes on GitHub
