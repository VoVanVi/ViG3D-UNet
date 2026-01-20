import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.brats_dataset import BratsNiftiDataset, BratsNpyDataset
from src.models.factory import ModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation outputs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="runs/vis.png", help="Output image path")
    return parser.parse_args()


def load_config(path: Path):
    return yaml.safe_load(path.read_text())


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    runtime_cfg = config.get("runtime", {})
    device = torch.device(runtime_cfg.get("device", "cpu"))

    model = ModelFactory.create(model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    if data_cfg.get("type") == "brats_nifti":
        data_root = Path(data_cfg.get("data_root", "data/brats_nifti"))
        val_split = Path(data_cfg.get("val_split", data_root / "val.txt"))
        modalities = data_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"])
        label_suffix = data_cfg.get("label_suffix", "seg")
        extensions = data_cfg.get("extensions", [".nii.gz", ".nii"])
        label_mapping = data_cfg.get("label_mapping", {4: 3})
        dataset = BratsNiftiDataset(
            data_root, val_split, modalities, label_suffix, extensions, label_mapping
        )
    else:
        data_root = Path(data_cfg.get("data_root", "data/brats"))
        val_images = Path(data_cfg.get("val_images", data_root / "val" / "images"))
        val_labels = Path(data_cfg.get("val_labels", data_root / "val" / "labels"))
        dataset = BratsNpyDataset(val_images, val_labels)

    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    y_np = y.numpy()
    slice_idx = y_np.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(y_np[slice_idx], cmap="gray")
    axes[0].set_title("Ground Truth")
    axes[1].imshow(preds[slice_idx], cmap="gray")
    axes[1].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
