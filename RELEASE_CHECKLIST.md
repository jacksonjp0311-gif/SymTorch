# SymTorch Release Checklist (v0.30.0 → first public ship)

This repo is **production-shaped** (contracts + gates) but not yet **production-shipped** (npm + tags + GitHub Releases).

This checklist is optimized for the *first* public release of the scoped packages:

- `@symtorch/core`
- `@symtorch/nn`
- `@symtorch/logic`
- `@symtorch/agent`
- `@symtorch/webgpu`

> Safety rule: do **dry runs** first. Do **not** publish or tag until the final “Authorize” checkpoints.

---

## 0) Preconditions (Accounts / Access)

- You have an npm account you control.
- You are logged in locally:

```powershell
npm whoami
```

- You have publish rights for the `@symtorch` scope (npm org or personal scope).

If this is the first time using this scope:
- create an npm org named `symtorch` (recommended) OR publish under your personal scope
- confirm that scoped packages can be published as public (`--access public`)

---

## 1) Repo preflight (must be clean)

From the repo root:

```powershell
git status --short
```

Expected: clean working tree.

Confirm current version:

```powershell
node -p "require('./package.json').version"
```

Expected: `0.30.0`

---

## 2) Run the full validation gates (must pass)

These are the release-manifest gates for `0.30.0`:

```powershell
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm playground:test
pnpm build
pnpm playground:build
pnpm exec tsx scripts/smoke-browser-playground.ts
pnpm playground:e2e
pnpm demo:policy-fixtures
pnpm demo:all
```

If any fail: stop here, fix, bump patch, repeat.

---

## 3) Verify build artifacts exist (dist + types)

For each package, confirm `dist/` exists and contains `index.js` + `index.d.ts`.

```powershell
ls packages/core/dist
ls packages/logic/dist
ls packages/nn/dist
ls packages/agent/dist
ls packages/webgpu/dist
```

---

## 4) Publish-readiness checks (pack first)

Before publishing, do an `npm pack` inspection per package (this prevents accidentally shipping `src/`, tests, or large files).

Example (repeat per package):

```powershell
cd packages/core
npm pack --silent
tar -tf *.tgz | more
cd ../..
```

What you want inside each tarball:
- `dist/**`
- `package.json`
- `README.md` (optional but recommended)
- `LICENSE` (recommended)

What you do **not** want:
- `src/**`
- `tests/**`
- `node_modules/**`
- `test-results/**`

If the tarball is too big or contains unwanted content, implement one of:
- per-package `"files": ["dist"]`
- or a root `.npmignore` / per-package `.npmignore`

(See `PUBLISH_READINESS_REPORT.md` for concrete suggestions.)

---

## 5) Dry-run publish (no network mutation)

pnpm supports recursive publish workflows; start with a dry-run.

From repo root:

```powershell
pnpm -r --filter "./packages/*" publish --dry-run
```

If pnpm prompts about access for scoped packages, expect you’ll need:

```powershell
pnpm -r --filter "./packages/*" publish --dry-run --access public
```

Confirm:
- pnpm replaces `workspace:*` with real versions
- each package publishes with version `0.30.0`
- no unexpected files are included

---

## 6) AUTHORIZE: Create the git tag

Only after gates + pack + dry-run are green.

```powershell
git tag -a v0.30.0 -m "SymTorch v0.30.0 — Production Contract Corpus Alpha"
git push origin v0.30.0
```

---

## 7) AUTHORIZE: Publish to npm (real)

```powershell
pnpm -r --filter "./packages/*" publish --access public
```

If this is the first publish, you may be asked for OTP (2FA). Complete it.

Post-publish verification:

```powershell
npm view @symtorch/core@0.30.0 version
npm view @symtorch/logic@0.30.0 version
npm view @symtorch/agent@0.30.0 version
```

---

## 8) Create GitHub Release (release notes)

- Go to GitHub → Releases → “Draft a new release”
- Tag: `v0.30.0`
- Title: `v0.30.0 — Production Contract Corpus Alpha`
- Paste release notes from `CHANGELOG.md` `v0.30.0` section
- Mark as **pre-release** (recommended for first ship)

---

## 9) Post-release smoke checks (consumer install)

In a fresh folder:

```powershell
mkdir symtorch-consumer-smoke
cd symtorch-consumer-smoke
pnpm init
pnpm add @symtorch/core @symtorch/logic @symtorch/agent
node -e "import('@symtorch/core').then(m=>console.log('core ok', Object.keys(m).length))"
```

Optional: run the 30-second demo code snippet from README using installed packages.

---

## 10) Rollback plan (if needed)

- npm: you generally **cannot** re-publish the same version. If something is wrong, cut a patch:
  - `0.30.1` with fixes and re-run the checklist.
- GitHub: tags can be deleted, but avoid doing so once published.

---

## Suggested release posture

For first public ship, recommend:
- Tag + GitHub Release: `v0.30.0` as **pre-release**
- npm: publish `0.30.0` as-is, then ship `0.30.1` quickly if any consumer smoke reveals packaging issues.
