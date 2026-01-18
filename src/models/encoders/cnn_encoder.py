from typing import List

import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder3d(nn.Module):
    def __init__(self, in_channels: int, channels: List[int]) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must be non-empty")
        self.stem = ConvBlock3d(in_channels, channels[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for idx, ch in enumerate(channels):
            if idx == 0:
                self.stages.append(nn.Identity())
            else:
                self.stages.append(ConvBlock3d(ch, ch))
            if idx < len(channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv3d(ch, channels[idx + 1], kernel_size=2, stride=2),
                        nn.BatchNorm3d(channels[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.stem(x)
        features.append(x)
        for idx, stage in enumerate(self.stages[1:], start=1):
            x = self.downsamples[idx - 1](x)
            x = stage(x)
            features.append(x)
        return features
