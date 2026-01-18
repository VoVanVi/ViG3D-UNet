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


class UNetDecoder3d(nn.Module):
    def __init__(self, encoder_channels: List[int], num_classes: int) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(encoder_channels))
        for idx in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[idx]
            skip_ch = reversed_channels[idx + 1]
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_ch, skip_ch, kernel_size=2, stride=2),
                    ConvBlock3d(skip_ch * 2, skip_ch),
                )
            )
        self.head = nn.Conv3d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for idx, block in enumerate(self.up_blocks):
            skip = features[-(idx + 2)]
            x = block[0](x)
            x = torch.cat([x, skip], dim=1)
            x = block[1](x)
        return self.head(x)
