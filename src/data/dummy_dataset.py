from typing import Tuple

import torch
from torch.utils.data import Dataset


class RandomVolumeDataset(Dataset):
    def __init__(
        self,
        size: int,
        in_shape: Tuple[int, int, int, int],
        num_classes: int,
    ) -> None:
        self.size = size
        self.in_shape = in_shape
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        channels, depth, height, width = self.in_shape
        x = torch.randn(channels, depth, height, width)
        y = torch.randint(0, self.num_classes, (depth, height, width))
        return x, y
