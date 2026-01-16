# Changelog

## Unreleased
- Add a config-driven training skeleton and dry-run pipeline for BraTS experiments.
- Log run metadata, save configs, metrics, and checkpoints for reproducibility.
- Add a minimal BraTS NPY dataset loader and example config for data sanity checks.
- Add a BraTS NIfTI dataset loader and example config that uses split files.
- Add Dice + Cross Entropy loss and per-class Dice metrics for multiclass training.
- Support `.nii` extensions and absolute case paths in the BraTS NIfTI loader.
