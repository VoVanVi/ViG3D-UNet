# ViG3D-UNet

This repository provides the offical Pytorch implementation of ViG3D-UNet in the following Paper:

**ViG3D-UNet: Volumetric Vascular Connectivity-Aware Segmentation via 3D Vision Graph Representation**<br/>
Bowen Liu, Chunlei Meng, Wei Lin, Hongda Zhang, Ziqing Zhou, Zhongxue Gan, Chun Ouyang<br/>
Fudan University<br/>
IEEE Journal of Biomedical and Health Informatics (https://ieeexplore.ieee.org/document/11155174)<br/>

## News

2025.09.10 Early Access on IEEE Eplore.

2025.09.04 🌟 Accepted by IEEE JBHI.

2025.04.12 Initial Submission on IEEE JBHI.

## Checklist
- ✅ Model Source code release.
- [ ] Inference code is coming SOON 
- [ ] Release ViG3D-UNet_Large and ViG3D-UNet_Large_medium checkpoints.
- [ ] Training  code.

## How to use ViG3D-UNet(To be done...)
### 1. Requirements

### 2. Installation

### 3. Inference

### 4.Training

### 5.Evaluation


## Contributions
**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**

Some insights about 3D CNN module is originated from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and the 3D Vision GNN(ViG3D) module is originated from [Vision GNN](https://github.com/huawei-noah/Efficient-AI-Backbones).

## BraTS Experiments (WIP)
The BraTS workflow is under active development with config-driven training runs. Use the dry-run config to verify the pipeline and output layout:

```bash
python -m src.train --config configs/brats/dry_run.yaml
```

The dry run writes a run folder under `runs/<exp_name>/<timestamp>/` with a config copy, environment info, and logs. It will log a single batch shape and exit.

### Minimal BraTS NPY layout (Step 2)
Expect preprocessed NPY files with multi-modal inputs stacked as `(C, D, H, W)` and labels as `(D, H, W)`. The loader matches image/label filenames.

```
data/brats_npy/
  train/
    images/
      case_001.npy
    labels/
      case_001.npy
  val/
    images/
      case_002.npy
    labels/
      case_002.npy
```

Run the data sanity check (dry-run) with:

```bash
python -m src.train --config configs/brats/data_example.yaml
```

### BraTS NIfTI layout with split files
For native BraTS NIfTI folders (e.g., `BraTS20_Training_001`), the dataset loader expects per-case files like:

- `BraTS20_Training_001_t1.nii.gz`
- `BraTS20_Training_001_t1ce.nii.gz`
- `BraTS20_Training_001_t2.nii.gz`
- `BraTS20_Training_001_flair.nii.gz`
- `BraTS20_Training_001_seg.nii.gz`

Split files should list case directory names (one per line), e.g. `BraTS20_Training_001`. Absolute case paths are also supported, and entries that include the dataset root folder name (e.g. `TextBraTSData/BraTS20_Training_001`) are handled. Both `.nii.gz` and `.nii` extensions are supported by default. Run a dry-run sanity check with:

```bash
python -m src.train --config configs/brats/brats_nifti_example.yaml
```

BraTS labels are commonly encoded as 0/1/2/4. The loader maps label 4 -> 3 by default so you can train with `num_classes: 4` without out-of-bounds errors. This can be overridden with `data.label_mapping` in the config.

### Step 3: Loss + metrics (baseline)
The training loop now uses multiclass Dice + Cross Entropy loss and logs Dice per class plus mean Dice. Dry-run logs the computed loss and mean Dice for a single batch.

### Step 4: A0 ViG3D-only backbone (graph-only)
Run the A0 ViG3D-only ablation (graph backbone + 1x1x1 segmentation head) with:

```bash
python -m src.train --config configs/brats/a0_vig3d_only.yaml
```

Expected output: a run directory under `runs/brats_a0_vig3d_only/<timestamp>/` containing `config.yaml`, `env.txt`, `train_log.txt`, and (if not dry-run) checkpoints + metrics.

## Citation
If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐.
