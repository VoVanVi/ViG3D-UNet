from typing import Tuple

import torch
from torch.utils.data import Dataset


class RandomVolumeDataset(Dataset):
    def __init__(
        self,
        size: int,
        in_shape: Tuple[int, int, int, int],
        out_channels: int,
    ) -> None:
        self.size = size
        self.in_shape = in_shape
        self.out_channels = out_channels

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        channels, depth, height, width = self.in_shape
        x = torch.randn(channels, depth, height, width)
        y = torch.randn(self.out_channels, depth, height, width)
        return x, y
