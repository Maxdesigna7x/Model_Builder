# Known limitations and remaining work

This page distinguishes implemented features from coverage that is still incomplete.

## Desktop and updates

- Linux graphics compatibility depends on the system's libraries, GPU/driver, and display session. The compatible-rendering flag is implemented; the original failure on another Zorin machine has not been confirmed fixed.
- Signed-update code and feed generation have local checks. A complete upgrade between two published versions still needs end-to-end verification. DEB updates use the package manager/download route.
- New dependency sets can require large downloads and extra disk space while the previous environment is retained. Old environments are not automatically deleted.
- Project/checkpoint format changes will need migrations. Keeping a runtime does not guarantee arbitrary future checkpoint compatibility.

## Models and data

- VAE and encoder–decoder Transformer families are not in the current catalog.
- Pretrained backbones are supported through curated contracts; their internals are not fully editable graph sublayers. Arbitrary external YOLO checkpoints are not automatically compatible with the training builder.
- Pretrained segmentation support requires matching processors, output taxonomy, and task contracts; an unrelated segmentation model is not interchangeable with U-Net.
- Download cancellation/offline recovery varies by tool. Dataset licenses/access conditions still apply.
- Some image adapters resize before the editable pipeline, so later enlargement cannot recover original detail.
- Dataset-curation model coverage and desktop QA are incomplete. Existing YOLO/embedding checks do not establish accuracy across all supported checkpoints/domains.
- Additional video, job-recovery, multi-page selection, EXIF, and concurrent-editing scenarios need broader QA. Dataset manifests are rewritten as JSON; very large inventories may need incremental storage.

## Linux startup work

The remaining implementation plan is [here](LINUX_STARTUP_PLAN.md). Priorities:

1. Audit DEB/AppImage native dependencies and baseline ABI in a clean supported system.
2. Add bounded early startup logs and recovery accessible outside the WebView.
3. Detect frontend startup failure without confusing long engine downloads with graphics failure.
4. Add brief package-level startup checks before automatic release publishing.
5. Verify on the affected machine; CI/Xvfb alone cannot establish hardware-driver compatibility.
