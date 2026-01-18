from typing import List

import torch
import torch.nn as nn

from src.models.decoders.unet_decoder import ConvBlock3d


class OffsetDecoder3d(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int,
        texture_skip_levels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.texture_skip_levels = texture_skip_levels
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

    def forward(
        self,
        fused_features: List[torch.Tensor],
        texture_features: List[torch.Tensor],
    ) -> torch.Tensor:
        if len(fused_features) != len(texture_features):
            raise ValueError("fused_features and texture_features must have the same length")
        x = fused_features[-1]
        total_skips = len(self.up_blocks)
        for idx, block in enumerate(self.up_blocks):
            skip_index = -(idx + 2)
            use_texture = idx >= max(0, total_skips - self.texture_skip_levels)
            skip = texture_features[skip_index] if use_texture else fused_features[skip_index]
            x = block[0](x)
            x = torch.cat([x, skip], dim=1)
            x = block[1](x)
        return self.head(x)
