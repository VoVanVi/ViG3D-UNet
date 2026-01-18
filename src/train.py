import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dummy_dataset import RandomVolumeDataset
from src.models.factory import ModelFactory
from src.utils.checkpoints import save_checkpoint
from src.utils.logger import log_environment, setup_logging
from src.utils.metrics import dice_loss, dice_metrics_as_dict, dice_per_class
from src.utils.seed import get_seed, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViG3D-UNet BraTS training (WIP)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry_run", action="store_true", help="Run a single batch and exit")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dummy_model(in_channels: int, out_channels: int, feature_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_channels, feature_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(feature_channels, out_channels, kernel_size=1),
    )


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def write_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_classes: int,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y) + dice_loss(preds, y, num_classes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            loss = criterion(preds, y) + dice_loss(preds, y, num_classes)
            total_loss += loss.item()
            dice_scores.append(dice_per_class(preds, y, num_classes))
    mean_loss = total_loss / max(1, len(loader))
    dice_tensor = torch.stack(dice_scores).mean(dim=0) if dice_scores else torch.zeros(num_classes)
    metrics = {"val_loss": mean_loss, "val_dice_mean": dice_tensor.mean().item()}
    for idx, value in enumerate(dice_metrics_as_dict(dice_tensor)):
        metrics[f"val_dice_class_{idx}"] = value
    return metrics


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    seed = get_seed(config.get("seed"))
    set_seed(seed)

    exp_name = config.get("exp_name", "brats_dummy")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config.get("output_dir", "runs"))
    run_dir = output_root / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir)
    logger.info("Starting run %s", run_dir)
    logger.info("Using seed %d", seed)

    config_copy_path = run_dir / "config.yaml"
    config_copy_path.write_text(yaml.safe_dump(config, sort_keys=False))

    env_info = {
        "git_commit": get_git_commit(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    log_environment(logger, run_dir / "env.txt", json.dumps(env_info, indent=2))

    data_cfg = config.get("data", {})
    dataset_type = data_cfg.get("type", "dummy")
    train_size = int(data_cfg.get("train_size", 4))
    val_size = int(data_cfg.get("val_size", 2))
    batch_size = int(data_cfg.get("batch_size", 2))
    num_workers = int(data_cfg.get("num_workers", 0))
    in_shape = tuple(data_cfg.get("shape", [4, 32, 32, 32]))

    model_cfg = config.get("model", {})
    in_channels = int(model_cfg.get("in_channels", in_shape[0]))
    num_classes = int(model_cfg.get("num_classes", model_cfg.get("out_channels", 3)))
    out_channels = int(model_cfg.get("out_channels", num_classes))
    feature_channels = int(model_cfg.get("feature_channels", 8))

    if dataset_type == "brats_npy":
        from src.data.brats_dataset import BratsNpyDataset

        data_root = Path(data_cfg.get("data_root", "data/brats"))
        train_images = Path(data_cfg.get("train_images", data_root / "train" / "images"))
        train_labels = Path(data_cfg.get("train_labels", data_root / "train" / "labels"))
        val_images = Path(data_cfg.get("val_images", data_root / "val" / "images"))
        val_labels = Path(data_cfg.get("val_labels", data_root / "val" / "labels"))
        train_dataset = BratsNpyDataset(train_images, train_labels)
        val_dataset = BratsNpyDataset(val_images, val_labels)
    elif dataset_type == "brats_nifti":
        from src.data.brats_dataset import BratsNiftiDataset

        data_root = Path(data_cfg.get("data_root", "data/brats_nifti"))
        train_split = Path(data_cfg.get("train_split", data_root / "train.txt"))
        val_split = Path(data_cfg.get("val_split", data_root / "val.txt"))
        modalities = data_cfg.get("modalities", ["t1", "t1ce", "t2", "flair"])
        label_suffix = data_cfg.get("label_suffix", "seg")
        extensions = data_cfg.get("extensions", [".nii.gz", ".nii"])
        label_mapping = data_cfg.get("label_mapping", {4: 3})
        train_dataset = BratsNiftiDataset(
            data_root, train_split, modalities, label_suffix, extensions, label_mapping
        )
        val_dataset = BratsNiftiDataset(
            data_root, val_split, modalities, label_suffix, extensions, label_mapping
        )
    else:
        train_dataset = RandomVolumeDataset(train_size, in_shape, num_classes)
        val_dataset = RandomVolumeDataset(val_size, in_shape, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    if model_cfg.get("type") == "dummy" or "type" not in model_cfg:
        model = build_dummy_model(in_channels, out_channels, feature_channels)
    else:
        model = ModelFactory.create(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("train", {}).get("lr", 1e-3)))
    criterion = nn.CrossEntropyLoss()

    if args.dry_run or config.get("train", {}).get("dry_run", False):
        x, y = next(iter(train_loader))
        preds = model(x)
        loss = criterion(preds, y) + dice_loss(preds, y, num_classes)
        dice_scores = dice_per_class(preds, y, num_classes)
        logger.info(
            "Dry run batch shapes: x=%s y=%s preds=%s", x.shape, y.shape, preds.shape
        )
        logger.info(
            "Dry run loss: %.4f mean_dice: %.4f", loss.item(), dice_scores.mean().item()
        )
        return

    epochs = int(config.get("train", {}).get("epochs", 1))
    metrics_rows: List[Dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, num_classes)
        val_metrics = validate(model, val_loader, criterion, num_classes)
        metrics_row = {"epoch": epoch, "train_loss": train_loss}
        metrics_row.update(val_metrics)
        metrics_rows.append(metrics_row)
        logger.info(
            "Epoch %d - train_loss: %.4f val_loss: %.4f val_dice_mean: %.4f",
            epoch,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_dice_mean"],
        )
        save_checkpoint(
            {"epoch": epoch, "state_dict": model.state_dict(), "val_loss": val_metrics["val_loss"]},
            run_dir / "checkpoints",
            "last.pt",
        )
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                },
                run_dir / "checkpoints",
                "best.pt",
            )

    write_metrics(run_dir / "metrics.csv", metrics_rows)


if __name__ == "__main__":
    main()
