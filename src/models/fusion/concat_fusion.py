from typing import List

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int]) -> None:
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length")
        self.projections = nn.ModuleList(
            [nn.Conv3d(in_ch, out_ch, kernel_size=1) for in_ch, out_ch in zip(in_channels, out_channels)]
        )

    def forward(self, cnn_feats: List[torch.Tensor], vig_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(cnn_feats) != len(vig_feats):
            raise ValueError("cnn_feats and vig_feats must have the same length")
        fused = []
        for idx, (cnn_feat, vig_feat) in enumerate(zip(cnn_feats, vig_feats)):
            x = torch.cat([cnn_feat, vig_feat], dim=1)
            fused.append(self.projections[idx](x))
        return fused
