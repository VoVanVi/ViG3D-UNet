from typing import List

import torch


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


def dice_metrics_as_dict(dice_scores: torch.Tensor) -> List[float]:
    return [score.item() for score in dice_scores]
