
from torch import nn
import argparse
from typing import Dict, Tuple, Optional

from .base_backbone import BaseEncoder
from .config.mobilevit_v3 import get_configuration
from .layers import ConvLayer, LinearLayer, GlobalPool, Identity
from .modules import InvertedResidual, MobileViTv3_Track_Block


class MobileViTv3_backbone(BaseEncoder):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:

        mobilevit_config = get_configuration(opts=opts)
        image_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]
        

        super().__init__(opts, *args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        #self.conv_1 = Patch_Embed_stage(in_chans=image_channels, embed_dim=out_channels, num_path=num_path, offset_clamp=(-3,3))
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=False,
        )
        self.model_conf_dict["layer4"] = {"in": in_channels, "out": out_channels}

        # check model
        # self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.mitv2.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between FFN layers. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.dropout",
            type=float,
            default=0.0,
            help="Dropout in attention layer. Defaults to 0.0",
        )
        group.add_argument(
            "--model.classification.mitv2.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        group.add_argument(
            "--model.classification.mitv2.attn-norm-layer",
            type=str,
            default="layer_norm_2d",
            help="Norm layer in attention block. Defaults to LayerNorm",
        )
        return parser

    def _make_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mobilevit_layer(
                opts=opts, input_channel=input_channel, cfg=cfg, dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts, input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
        opts, input_channel: int, cfg: Dict
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mobilevit_layer(
        self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False
    ) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        dropout = getattr(opts, "model.classification.mitv2.dropout", 0.0)

        block.append(
            MobileViTv3_Track_Block(
                opts=opts,
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=dropout,
                ffn_dropout=getattr(
                    opts, "model.classification.mitv2.ffn_dropout", 0.0
                ),
                attn_dropout=getattr(
                    opts, "model.classification.mitv2.attn_dropout", 0.0
                ),
                conv_ksize=3,
                attn_norm_layer=getattr(
                    opts, "model.classification.mitv2.attn_norm_layer", "layer_norm_2d"
                ),
                dilation=self.dilation,
            )
        )

        return nn.Sequential(*block), input_channel
