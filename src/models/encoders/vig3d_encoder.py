from typing import List

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


class ViG3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        blocks_per_stage: List[int],
        k: int = 9,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, (in_ch, num_blocks) in enumerate(zip(channels, blocks_per_stage)):
            blocks = nn.Sequential(
                *[Grapher3d(in_ch, kernel_size=k, conv="mr") for _ in range(num_blocks)]
            )
            self.stages.append(blocks)
            if idx < len(channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv3d(in_ch, channels[idx + 1], kernel_size=2, stride=2),
                        nn.BatchNorm3d(channels[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        self.act = nn.ReLU(inplace=True)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = self.act(stage(x))
            features.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return features
