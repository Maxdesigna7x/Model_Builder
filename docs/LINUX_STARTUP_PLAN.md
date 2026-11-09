# Linux startup implementation plan

Status: partially implemented. `--graphics=compatible`, `--diagnose`, and the base HTML loading message exist. The work below remains necessary; do not report the original Zorin failure as resolved without testing it.

## Target

Show the first setup screen on a clean supported Linux x86_64 system without Python, Node, Rust, or Internet. Download the heavy engine only after device selection. Initially target Zorin/Ubuntu 22.04-based systems or later and record exactly which releases were tested. Preserve Windows and browser mode.

## Package audit

Add `scripts/audit-linux-package.sh` and run it against the real release packages in a clean target system.

- Extract DEB and AppImage into temporary directories; also consider AppImage without FUSE.
- Inspect main executables, uv, WebKit helpers, dynamic loads, ABI requirements, and DEB `Depends`. Checking only the main binary is insufficient.
- Report required library, bundled/system origin, missing dependencies, and corresponding system package.
- Keep the oldest supported build baseline. For DEB, declare system dependencies; for AppImage, use supported Tauri/linuxdeploy bundling for redistributable libraries.
- Do not copy developer GPU drivers, kernel, or glibc. Do not fetch arbitrary native libraries or change global loader paths at first launch.
- If bundling cannot produce a coherent base, document the blocker before changing architecture or hand-copying libraries.

## Early diagnostics

Add a `src-tauri/src/startup.rs` module and invoke it before WebView creation.

- Write bounded, rotated startup logs in user data, excluding credentials and dataset contents.
- Record app version, graphics mode, session type, process/WebView/frontend milestones, and errors/panics.
- Expand `--diagnose` to describe the environment without downloading the engine.
- A dynamic-loader failure can occur before Rust runs. Cover it through package auditing and a launcher report; do not promise React can recover from it.

## Graphics recovery

Extend the existing flags in `src-tauri/src/main.rs`.

- Keep normal rendering by default. Apply WebKit settings before threads/WebView creation, respecting existing user variables.
- The implemented compatible flag sets `WEBKIT_DISABLE_DMABUF_RENDERER=1`. Evaluate `WEBKIT_DISABLE_COMPOSITING_MODE=1` only as a tested second recovery level.
- Add a desktop-menu action to open compatible mode without needing the main window.
- Avoid blanket NVIDIA workarounds without confirming the affected configuration. Do not infer that missing `nvidia-smi` means no NVIDIA GPU.
- Limit any recovery restart attempts. Never alter CUDA selection, drivers, project data, or the private runtime during graphics recovery.

## Frontend failure handling

Extend `frontend/index.html`, `main.tsx`, the desktop IPC layer, and capability permissions only as required.

- Keep a local loading fallback; add a React error boundary and useful resource/JavaScript error reporting.
- Send explicit frontend-ready/setup-visible/app-visible acknowledgements to Rust.
- Use a bounded startup timer, separate from engine download time. If readiness never arrives, record diagnostics and attempt a native warning with compatible-mode/log-location options.
- Verify a native dialog route independently of WebKit; do not assume it can open if its own toolkit libraries are missing.
- JavaScript readiness does not prove that pixels render correctly. Retain external recovery access and real visual checks.

## Brief verification and release gate

Add a package startup smoke test to the Linux build job before artifact upload/publication.

- Use the actual release DEB/AppImage, temporary user data, and a clean target environment without developer Python.
- With network disabled, verify that setup appears, remains responsive, and does not start engine downloads itself.
- Test normal/compatible modes and AppImage/FUSE alternatives. Capture readiness events, process failures, and a screenshot.
- Use Xvfb as an additional CI check, not a substitute for NVIDIA/Wayland testing.
- Add frontend tests for load failure, missing readiness, and recovery. Keep Windows/browser regression checks.
- Stop after setup appears or preparation begins; do not wait for complete CUDA downloads by default.
- Verify the affected Zorin machine if available; otherwise leave that result explicitly pending.

Acceptance: first-screen startup passes on documented clean systems; dependencies are accounted for; failures produce bounded diagnostics and externally accessible recovery; projects and CUDA choice remain intact; failing startup checks prevent automatic release publication.

## References

- [Tauri Linux graphics issues](https://v2.tauri.app/develop/debug/linux-graphics/)
- [Tauri AppImage packaging](https://v2.tauri.app/distribute/appimage/)
- [Tauri Debian packages](https://v2.tauri.app/distribute/debian/)
