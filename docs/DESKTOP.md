# Desktop setup and updates

## Install

Download an executable or package from [GitHub Releases](https://github.com/Maxdesigna7x/Model_Builder/releases/latest).

- **Windows x64:** run the `-setup.exe` installer or install the `.msi` package.
- **Linux x86_64:** install the `.deb` through a compatible distribution's package manager, or make the `.AppImage` executable and run it. AppImage may require FUSE support or extraction on some systems.

The app prepares its own Python environment. You do not need Python, Node, or Rust developer tools. DEB dependencies are managed by the system package manager; AppImage still requires a compatible Linux system and graphics stack.

## First launch

Choose CPU or NVIDIA. ModelBuilder downloads Python 3.12 and the required packages into your user data directory. NVIDIA uses the CUDA 13.0 variant of PyTorch and requires compatible hardware/drivers. It does not install or replace your graphics driver.

Allow several GB of free space and Internet access. The progress display counts prepared components, including packages reused from cache. The current component uses an indeterminate indicator because a reliable per-file download percentage is not available.

If setup fails, check the error, connection, disk space, and driver compatibility. Retry or select CPU. Datasets and pretrained weights can require additional downloads later.

## Updates

Current updater-enabled builds check for a new release when the main app opens. Use the **Actualizaciones** button to check manually. Release notes describe which capabilities are in a particular download.

- **Windows and AppImage:** save your work, finish tasks, then choose install/restart. The app verifies the update signature before installation.
- **DEB:** use the download link and install the new package through the system package manager.
- **Older apps without the updater:** install an updater-enabled build manually once.

Installation is blocked while an app-managed training process is active. Save unsaved edits and finish other tasks yourself before updating. If a check fails offline, the current app remains usable.

### What is preserved

App files are separate from the user data directory. Projects, datasets, checkpoints, downloaded Python, and compatible environments survive normal app updates. Do not remove that directory when upgrading or uninstalling if you want to retain your data.

Changing app metadata/version does not invalidate the environment. The app compares runtime dependency requirements; when they change, it prepares a separate environment and activates it after verification, retaining the previous one. Cached Python/packages are reused where available. Real dependency changes or a cleared cache can require downloads.

Preserving files is not a guarantee that every future PyTorch version can load every old checkpoint. Changes to project/checkpoint formats need explicit migrations and compatibility testing.

## Linux: blank window

Close the app and try:

```bash
modelbuilder --graphics=compatible
```

For AppImage, replace `modelbuilder` with the executable's path. This changes interface rendering without disabling CUDA training. Existing user-provided WebKit settings are respected.

For basic environment information without downloading the engine:

```bash
modelbuilder --diagnose
```

These are initial diagnostic tools, not a universal fix. If the window remains blank, launch it from a terminal and retain the error output along with OS/version, GPU/driver, and X11/Wayland session details. A complete dependency audit and startup recovery remain [pending](LIMITATIONS.md#linux-startup-work).
