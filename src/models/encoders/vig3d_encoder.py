from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model_vig.gcn_lib import Grapher3d


def _to_3tuple(value: int) -> Tuple[int, int, int]:
    return (value, value, value)


def _build_stem(
    in_channels: int,
    out_channels: int,
    stem_stride: int,
    patch_size: Optional[int],
) -> nn.Module:
    if patch_size is not None:
        kernel = _to_3tuple(patch_size)
        stride = _to_3tuple(patch_size)
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stem_stride, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class ViG3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        k: int = 9,
        stem_stride: int = 1,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if stem_stride > 1 and patch_size is not None:
            raise ValueError("Use either stem_stride or patch_size, not both")
        self.stem = _build_stem(in_channels, hidden_channels, stem_stride, patch_size)
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
        stem_stride: int = 1,
        patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")
        if stem_stride > 1 and patch_size is not None:
            raise ValueError("Use either stem_stride or patch_size, not both")

        self.stem = _build_stem(in_channels, channels[0], stem_stride, patch_size)
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
