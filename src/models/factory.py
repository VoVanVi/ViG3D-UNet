from typing import Any, Dict, List

import torch.nn as nn

from src.models.decoders.unet_decoder import UNetDecoder3d
from src.models.encoders.vig3d_encoder import ViG3DBackbone, ViG3DEncoder
from src.models.heads.seg_head import SegmentationHead


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
            backbone = ViG3DBackbone(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                k=k,
            )
            head = SegmentationHead(hidden_channels, num_classes)
            return nn.Sequential(backbone, head)

        if model_type == "a1_vig3d_unet":
            in_channels = int(model_cfg.get("in_channels", 4))
            num_classes = int(model_cfg.get("num_classes", 4))
            encoder_channels = model_cfg.get("vig_encoder_channels", [16, 32, 64])
            blocks_per_stage = model_cfg.get("vig_blocks_per_stage", [1, 1, 1])
            k = int(model_cfg.get("vig_k", 9))
            encoder = ViG3DEncoder(
                in_channels=in_channels,
                channels=[int(c) for c in encoder_channels],
                blocks_per_stage=[int(b) for b in blocks_per_stage],
                k=k,
            )
            decoder = UNetDecoder3d([int(c) for c in encoder_channels], num_classes)
            return nn.Sequential(encoder, decoder)

        raise ValueError(f"Unknown model type: {model_type}")
