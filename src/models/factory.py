from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.models.decoders.offset_decoder import OffsetDecoder3d
from src.models.decoders.unet_decoder import UNetDecoder3d
from src.models.encoders.cnn_encoder import CNNEncoder3d
from src.models.encoders.vig3d_encoder import ViG3DBackbone, ViG3DEncoder
from src.models.fusion.channel_attention_fusion import ChannelAttentionFusion
from src.models.fusion.concat_fusion import ConcatFusion
from src.models.heads.seg_head import SegmentationHead


def _get_vig_stem_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stem_stride": int(model_cfg.get("vig_stem_stride", 1)),
        "patch_size": model_cfg.get("vig_patch_size"),
    }


class ModelFactory:
    @staticmethod
    def create(model_cfg: Dict[str, Any]) -> nn.Module:
        model_type = model_cfg.get("type", "dummy")

        # Ablation switch happens here; choose model based on config only.
        if model_type == "a0_vig3d":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            hidden_channels = int(model_cfg.get("vig_hidden_channels", 32))
            num_blocks = int(model_cfg.get("vig_num_blocks", 2))
            k = int(model_cfg.get("vig_k", 9))
            stem_cfg = _get_vig_stem_cfg(model_cfg)
            backbone = ViG3DBackbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                k=k,
                stem_stride=stem_cfg["stem_stride"],
                patch_size=stem_cfg["patch_size"],
            )
            head = SegmentationHead(hidden_channels, num_classes)
            return nn.Sequential(backbone, head)

        if model_type == "a1_vig3d_unet":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            encoder_channels = model_cfg.get("vig_encoder_channels", [16, 32, 64])
            blocks_per_stage = model_cfg.get("vig_blocks_per_stage", [1, 1, 1])
            k = int(model_cfg.get("vig_k", 9))
            stem_cfg = _get_vig_stem_cfg(model_cfg)
            encoder = ViG3DEncoder(
                in_channels=in_channels,
                channels=[int(c) for c in encoder_channels],
                blocks_per_stage=[int(b) for b in blocks_per_stage],
                k=k,
                stem_stride=stem_cfg["stem_stride"],
                patch_size=stem_cfg["patch_size"],
            )
            decoder = UNetDecoder3d([int(c) for c in encoder_channels], num_classes)
            return nn.Sequential(encoder, decoder)

        if model_type == "a2_cnn_vig_concat":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            encoder_channels = model_cfg.get("encoder_channels", [16, 32, 64])
            vig_blocks_per_stage = model_cfg.get("vig_blocks_per_stage", [1, 1, 1])
            k = int(model_cfg.get("vig_k", 9))
            stem_cfg = _get_vig_stem_cfg(model_cfg)
            cnn_encoder = CNNEncoder3d(in_channels, [int(c) for c in encoder_channels])
            vig_encoder = ViG3DEncoder(
                in_channels=in_channels,
                channels=[int(c) for c in encoder_channels],
                blocks_per_stage=[int(b) for b in vig_blocks_per_stage],
                k=k,
                stem_stride=stem_cfg["stem_stride"],
                patch_size=stem_cfg["patch_size"],
            )
            fusion = ConcatFusion(
                [int(c) * 2 for c in encoder_channels], [int(c) for c in encoder_channels]
            )
            decoder = UNetDecoder3d([int(c) for c in encoder_channels], num_classes)

            class A2Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.cnn_encoder = cnn_encoder
                    self.vig_encoder = vig_encoder
                    self.fusion = fusion
                    self.decoder = decoder

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    cnn_feats = self.cnn_encoder(x)
                    vig_feats = self.vig_encoder(x)
                    fused = self.fusion(cnn_feats, vig_feats)
                    return self.decoder(fused)

            return A2Model()

        if model_type == "a3_cnn_vig_attn":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            encoder_channels = model_cfg.get("encoder_channels", [16, 32, 64])
            vig_blocks_per_stage = model_cfg.get("vig_blocks_per_stage", [1, 1, 1])
            k = int(model_cfg.get("vig_k", 9))
            reduction = int(model_cfg.get("attn_reduction", 8))
            stem_cfg = _get_vig_stem_cfg(model_cfg)
            cnn_encoder = CNNEncoder3d(in_channels, [int(c) for c in encoder_channels])
            vig_encoder = ViG3DEncoder(
                in_channels=in_channels,
                channels=[int(c) for c in encoder_channels],
                blocks_per_stage=[int(b) for b in vig_blocks_per_stage],
                k=k,
                stem_stride=stem_cfg["stem_stride"],
                patch_size=stem_cfg["patch_size"],
            )
            fusion = ChannelAttentionFusion(
                [int(c) * 2 for c in encoder_channels],
                [int(c) for c in encoder_channels],
                reduction=reduction,
            )
            decoder = UNetDecoder3d([int(c) for c in encoder_channels], num_classes)

            class A3Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.cnn_encoder = cnn_encoder
                    self.vig_encoder = vig_encoder
                    self.fusion = fusion
                    self.decoder = decoder

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    cnn_feats = self.cnn_encoder(x)
                    vig_feats = self.vig_encoder(x)
                    fused = self.fusion(cnn_feats, vig_feats)
                    return self.decoder(fused)

            return A3Model()

        if model_type == "a4_full_paperclip":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            encoder_channels = model_cfg.get("encoder_channels", [16, 32, 64])
            vig_blocks_per_stage = model_cfg.get("vig_blocks_per_stage", [1, 1, 1])
            k = int(model_cfg.get("vig_k", 9))
            reduction = int(model_cfg.get("attn_reduction", 8))
            texture_skip_levels = int(model_cfg.get("texture_skip_levels", 1))
            stem_cfg = _get_vig_stem_cfg(model_cfg)
            cnn_encoder = CNNEncoder3d(in_channels, [int(c) for c in encoder_channels])
            vig_encoder = ViG3DEncoder(
                in_channels=in_channels,
                channels=[int(c) for c in encoder_channels],
                blocks_per_stage=[int(b) for b in vig_blocks_per_stage],
                k=k,
                stem_stride=stem_cfg["stem_stride"],
                patch_size=stem_cfg["patch_size"],
            )
            fusion = ChannelAttentionFusion(
                [int(c) * 2 for c in encoder_channels],
                [int(c) for c in encoder_channels],
                reduction=reduction,
            )
            decoder = OffsetDecoder3d(
                [int(c) for c in encoder_channels],
                num_classes,
                texture_skip_levels=texture_skip_levels,
            )

            class A4Model(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.cnn_encoder = cnn_encoder
                    self.vig_encoder = vig_encoder
                    self.fusion = fusion
                    self.decoder = decoder

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    cnn_feats = self.cnn_encoder(x)
                    vig_feats = self.vig_encoder(x)
                    fused = self.fusion(cnn_feats, vig_feats)
                    return self.decoder(fused, cnn_feats)

            return A4Model()

        raise ValueError(f"Unknown model type: {model_type}")
