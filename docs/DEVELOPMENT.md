# Development and publishing

The downloaded app prepares its own runtime. This guide is for working on source code or producing installers.

## Requirements and source setup

Use Node.js 22.12+, npm, and Python 3.12 with pip/venv. Desktop development also needs stable Rust/Cargo and native Tauri build dependencies. Keep development packages in `.venv`; avoid modifying a shared Conda/system environment.

```bash
git clone https://github.com/Maxdesigna7x/Model_Builder.git
cd Model_Builder
```

### Linux browser mode

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm ci
bash run-web.sh
```

### Windows browser mode

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
npm ci
$env:MODELBUILDER_PYTHON = (Resolve-Path '.\.venv\Scripts\python.exe').Path
npm run dev
```

Open `http://localhost:1420`. Set `MODELBUILDER_PYTHON` to use a different interpreter. CUDA development needs the appropriate PyTorch build and driver; `constraints-linux-py312-cu130.txt` records a specific Linux/Python 3.12/CUDA environment and is not a cross-platform lock.

### Desktop development

On Ubuntu/Debian, `bash scripts/install-linux.sh` installs native dependencies and prepares the development environment; then use `bash run.sh`. The installer requires Python 3.11/3.12, Node/npm, sudo, and an apt-based system. Review the script before use: it installs packages and may install Rust. `run.sh` can terminate a process occupying its configured development port.

On Windows, install Rust, Node, Python, WebView2, and Visual Studio C++ Build Tools/Windows SDK. Run `scripts/install-windows.ps1`, then `run-windows.ps1` in PowerShell. Inspect installer output for failures.

## Code map

| Path | Responsibility |
| --- | --- |
| `frontend/` | React/TypeScript UI, graph editing, data views, charts |
| `frontend/src/bridge.ts` | Tauri IPC or browser-mode backend transport |
| `backend/modelbuilder/` | Data adapters, graph/model execution, training, curation, persistence |
| `src-tauri/` | Desktop shell, private runtime, updates, native file dialogs |
| `tests/` and `scripts/test-*` | Backend, graph, and UI checks |
| `workspace_data/` | Source-mode projects and artifacts; ignored by Git |

Distributed apps use their OS user data directory instead of the source workspace. Keep it separate from install files. Never package datasets, checkpoints, local credentials, caches, or developer environments.

### Runtime contracts

`bridge.ts` sends the same backend actions through Tauri IPC or the local Vite proxy. The desktop shell invokes the Python engine with a JSON request and reads its JSON response; training streams events through a separate process. Keep both transports aligned when changing actions or error responses.

The backend catalog defines valid task/architecture combinations. Model graphs, prepared-data manifests, and run/checkpoint metadata are persisted locally. Changing those formats requires compatibility handling and regression tests; preserving an environment alone is insufficient. Dataset-curation jobs have their own persisted state and should not be treated as model-training runs.

Pretrained integrations use curated pinned revisions. Do not enable untrusted remote model code as an import shortcut. Keep authenticated downloads and model caches outside version control.

## Checks

```bash
.venv/bin/python -m pytest -q
npm run build
npm run test:builder-graph
npm run test:desktop-setup
cargo test --manifest-path src-tauri/Cargo.toml --lib setup_tests
```

On Windows, use `.\.venv\Scripts\python.exe`. UI checks require Chrome/Chromium (`CHROME_PATH` can select it) and often use simulated backend responses. `package.json` lists additional targeted graph/UI checks. Run checks relevant to the change; a successful build is not a graphics compatibility test.

`python scripts/test-desktop-runtime.py` creates a clean runtime and downloads packages. The optional `--backend nvidia` requires a compatible NVIDIA machine and can download several GB. Do not run complete runtime downloads merely to check that startup begins.

To generate separate demo screenshots without overwriting the README images:

```bash
CAPTURE_DIR=capturas/readme-demo npm run capturas:readme
```

## Build installers locally

Build Windows packages on Windows and Linux packages on Linux, with the native requirements installed:

```bash
npm ci
python -m pip install uv==0.9.30
python scripts/prepare-desktop-runtime.py
npm run tauri -- build
```

Use `python3` on Linux if needed. Outputs are under `src-tauri/target/release/bundle/`: `deb/`, `appimage/`, `nsis/`, and `msi/` as applicable. `npm run build` alone only builds the frontend.

Updater artifacts require the existing signing key in `TAURI_SIGNING_PRIVATE_KEY`. Keep that key private and backed up; do not generate a replacement per release. Its public counterpart is in `tauri.conf.json`. This cryptographic update signature is separate from OS publisher/code signing.

## Build and publish through GitHub

Commit and push the intended source revision, then run:

```bash
gh workflow run desktop.yml --ref main
```

The workflow builds both systems, uploads installers/signatures, generates `latest.json`, and publishes a new release only after both platform jobs succeed. Each run has its own tag; installer patch versions follow the run number. It retains previous releases.

The repository secret `TAURI_SIGNING_PRIVATE_KEY` supplies the signing key. `scripts/create-update-manifest.py` maps AppImage, NSIS, and MSI packages to their respective updater targets and rejects missing signatures or mismatched versions.

**Start the workflow and report that it was requested. Do not wait or repeatedly poll unless the user explicitly requests it.** Do not report an installer or publication as successful before verification.

The process is repeatable, not guaranteed byte-for-byte reproducible: some Python dependencies use version ranges. The updater is disabled in development builds. Test a real signed upgrade before claiming full end-to-end update validation.
