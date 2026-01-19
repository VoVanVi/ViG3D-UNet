import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.brats_dataset import BratsNiftiDataset, BratsNpyDataset
from src.data.dummy_dataset import RandomVolumeDataset
from src.models.factory import ModelFactory
from src.utils.metrics import brats_region_metrics, dice_per_class, hd95_per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="runs/eval_metrics.json", help="Output JSON path")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config: Dict[str, Any]) -> DataLoader:
    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("type", "dummy")
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    in_shape = tuple(data_cfg.get("shape", [4, 32, 32, 32]))
    num_classes = int(config.get("model", {}).get("num_classes", 4))

    if dataset_type == "brats_npy":
        data_root = Path(data_cfg.get("data_root", "data/brats"))
        val_images = Path(data_cfg.get("val_images", data_root / "val" / "images"))
        val_labels = Path(data_cfg.get("val_labels", data_root / "val" / "labels"))
        dataset = BratsNpyDataset(val_images, val_labels)
    elif dataset_type == "brats_nifti":
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
        dataset = RandomVolumeDataset(int(data_cfg.get("val_size", 1)), in_shape, num_classes)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    device = torch.device(runtime_cfg.get("device", "cpu"))

    if model_cfg.get("type") == "dummy" or "type" not in model_cfg:
        raise ValueError("Evaluation requires a non-dummy model config")

    model = ModelFactory.create(model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    loader = build_dataset(config)
    num_classes = int(model_cfg.get("num_classes", 4))
    dice_scores = []
    hd95_scores = []
    region_metrics = {"WT": {"dice": [], "hd95": []}, "TC": {"dice": [], "hd95": []}, "ET": {"dice": [], "hd95": []}}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if logits.shape[2:] != y.shape[1:]:
                y = torch.nn.functional.interpolate(
                    y.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest"
                ).squeeze(1).long()
            dice_scores.append(dice_per_class(logits, y, num_classes))
            hd95_scores.append(hd95_per_class(logits, y, num_classes))
            region = brats_region_metrics(logits, y)
            for key in region_metrics:
                region_metrics[key]["dice"].append(region[key]["dice"])
                region_metrics[key]["hd95"].append(region[key]["hd95"])

    dice_tensor = torch.stack(dice_scores).mean(dim=0)
    hd95_tensor = torch.stack(hd95_scores).mean(dim=0)
    result = {
        "dice_mean": dice_tensor.mean().item(),
        "dice_per_class": [v.item() for v in dice_tensor],
        "hd95_mean": hd95_tensor.mean().item(),
        "hd95_per_class": [v.item() for v in hd95_tensor],
        "brats_regions": {
            region: {
                "dice": float(np.mean(values["dice"])) if values["dice"] else 0.0,
                "hd95": float(np.mean(values["hd95"])) if values["hd95"] else 0.0,
            }
            for region, values in region_metrics.items()
        },
        "brats_region_mean": {
            "dice": float(
                np.mean([np.mean(values["dice"]) for values in region_metrics.values() if values["dice"]])
            ),
            "hd95": float(
                np.mean([np.mean(values["hd95"]) for values in region_metrics.values() if values["hd95"]])
            ),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
