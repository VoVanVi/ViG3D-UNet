from typing import Dict, List

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum().float() + target_mask.sum().float()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    return torch.stack(dice_scores)


def dice_metrics_as_dict(dice_scores: torch.Tensor) -> List[float]:
    return [score.item() for score in dice_scores]


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def _surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if pred.sum() == 0 and target.sum() == 0:
        return np.array([0.0])
    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)
    target_dt = distance_transform_edt(~target_surface)
    return target_dt[pred_surface]


def hd95_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    hd95_scores = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets_np == cls
        distances = _surface_distances(pred_mask, target_mask)
        hd95_scores.append(np.percentile(distances, 95))
    return torch.tensor(hd95_scores, dtype=torch.float32)


def brats_region_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "WT": labels > 0,
        "TC": np.logical_or(labels == 1, labels == 3),
        "ET": labels == 3,
    }


def brats_region_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    metrics: Dict[str, Dict[str, float]] = {}
    for region in ["WT", "TC", "ET"]:
        pred_mask = brats_region_masks(preds)[region]
        target_mask = brats_region_masks(targets_np)[region]
        intersection = (pred_mask & target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        distances = _surface_distances(pred_mask, target_mask)
        hd95 = float(np.percentile(distances, 95))
        metrics[region] = {"dice": float(dice), "hd95": float(hd95)}
    return metrics
