from typing import List, Tuple

import torch
import torch.nn as nn

from model_vig.gcn_lib import Grapher3d


class ViG3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        k: int = 9,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [Grapher3d(hidden_channels, kernel_size=k, conv="mr") for _ in range(num_blocks)]
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = self.act(block(x))
        return x
