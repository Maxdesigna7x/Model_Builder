# Agent instructions

## Preserve application behavior and user data

- Keep both the local-browser mode and the Tauri desktop app working. UI lives in `frontend/`, the Python engine in `backend/modelbuilder/`, and desktop integration in `src-tauri/`.
- Never commit or bundle projects, datasets, checkpoints, credentials, caches, or developer environments. Distributed user data must stay outside the install directory.
- Use `.venv` for development; do not modify shared Conda/system environments.
- Desktop setup supports CPU/NVIDIA and component progress. Do not run complete runtime/CUDA downloads just to check that startup begins; use brief relevant checks unless more is explicitly needed/requested.

## Build and publish desktop installers

The authoritative procedure is [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Keep it and this file updated when commands change.

Push the intended commit to the requested remote branch before starting GitHub Actions:

```bash
gh workflow run desktop.yml --ref main
```

Use the appropriate remote ref instead of `main` when needed. `.github/workflows/desktop.yml` also responds to `desktop-v*` tags. It builds Windows x64/Linux x86_64 packages, uploads artifacts and signatures, creates `latest.json`, and publishes a new release only after both jobs succeed. Each run gets its own tag and installer version; retain previous releases.

**Do not wait for installer builds or poll their status unless the user explicitly asks you to wait or verify results.** Start the workflow, report the request and Actions link, and do not claim successful compilation/publication without evidence.

For local builds, use Windows for Windows packages and Linux for Linux packages, with Tauri's native build requirements installed:

```bash
npm ci
python -m pip install uv==0.9.30
python scripts/prepare-desktop-runtime.py
npm run tauri -- build
```

Use `python3` on Linux if needed. Output is in `src-tauri/target/release/bundle/{deb,appimage,nsis,msi}/` as applicable. If asked to launch a local build without waiting, run it in the background with a log file.

These commands are repeatable, not guaranteed byte-for-byte reproducible; some Python dependencies use version ranges.

## Updates and compatibility

- Update signatures use the public key in `tauri.conf.json` and the private GitHub secret `TAURI_SIGNING_PRIVATE_KEY`. Never regenerate the key per build or commit/log private key material. Preserve a secure backup. Older apps without the updater need one manual install.
- Environment compatibility is based on dependency requirements, not app metadata/version. Prepare changed environments separately, verify before activating, and retain previous environments/projects. Do not add destructive data migrations to the updater.
- On Linux, `--graphics=compatible` adjusts interface rendering; `--diagnose` prints basic environment information. Do not claim the original blank-window problem is fixed without verifying the affected machine.
- Remaining startup work is in [docs/LINUX_STARTUP_PLAN.md](docs/LINUX_STARTUP_PLAN.md); current limits are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
