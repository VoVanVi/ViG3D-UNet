# Changelog

## Unreleased
- Add a config-driven training skeleton and dry-run pipeline for BraTS experiments.
- Log run metadata, save configs, metrics, and checkpoints for reproducibility.
- Add a minimal BraTS NPY dataset loader and example config for data sanity checks.
- Add a BraTS NIfTI dataset loader and example config that uses split files.
- Add Dice + Cross Entropy loss and per-class Dice metrics for multiclass training.
- Support `.nii` extensions and absolute case paths in the BraTS NIfTI loader.
- Normalize split entries that include the dataset root folder name and silence git hash warnings.
- Map BraTS label 4 -> 3 by default to support 4-class training targets.
- Add a ViG3D-only A0 model factory path and config-driven ablation entrypoint.
- Add a ViG3D encoder + UNet decoder (A1) with a config-driven model switch.
- Add a CNN+ViG3D concat fusion (A2) with a config-driven model switch.
- Add a CNN+ViG3D channel attention fusion (A3) with a config-driven model switch.
- Add the full A4 model with channel attention and offset (paperclip) decoder.
- Add a sweep runner and ablation study guide for aggregating runs.
- Add runtime device/gpu config support and move batches to the selected device.
- Add ViG3D stem options for downsampled voxel nodes (Option A) or patch-token nodes (Option B).
- Downsample labels to match patch-token outputs during training.
- Add early stopping and evaluation script with Dice/HD95 metrics.
- Add BraTS region metrics (WT/TC/ET) with mean Dice/HD95 reporting.
- Add visualization helper and fix ViG fusion shape alignment.
- Add configurable Dice/CE weights, validation WT/TC/ET logging, and val-loss early stopping defaults.
