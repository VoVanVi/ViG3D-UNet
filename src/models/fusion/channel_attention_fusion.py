from typing import List

import torch
import torch.nn as nn


class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int], reduction: int = 8) -> None:
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length")
        self.projections = nn.ModuleList()
        self.attn_mlps = nn.ModuleList()
        self.pools = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            self.projections.append(nn.Conv3d(in_ch, out_ch, kernel_size=1))
            hidden = max(1, out_ch // reduction)
            self.attn_mlps.append(
                nn.Sequential(
                    nn.Linear(out_ch, hidden, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, out_ch, bias=False),
                    nn.Sigmoid(),
                )
            )
            self.pools.append(nn.AdaptiveAvgPool3d(1))

    def forward(self, cnn_feats: List[torch.Tensor], vig_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(cnn_feats) != len(vig_feats):
            raise ValueError("cnn_feats and vig_feats must have the same length")
        fused = []
        for idx, (cnn_feat, vig_feat) in enumerate(zip(cnn_feats, vig_feats)):
            x = torch.cat([cnn_feat, vig_feat], dim=1)
            x = self.projections[idx](x)
            pooled = self.pools[idx](x).flatten(1)
            attn = self.attn_mlps[idx](pooled).view(pooled.size(0), -1, 1, 1, 1)
            fused.append(x * attn)
        return fused
