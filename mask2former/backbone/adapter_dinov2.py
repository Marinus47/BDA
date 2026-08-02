import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


from .backbones import *

from .adapter import adapter

def get_adapter_dino_args(cfg):


    backbone_conf = cfg.MODEL.ADAPTER_DINOV2
    vit_backbone = get_models(backbone_conf.NAME, backbone_conf.WEIGHTS)
    bg_configs = vit_backbone.configs_dict
    adapter_module_conf = cfg.MODEL.ADAPTER


    adapter_config = {
        "type": "Adapter",
        "dim": adapter_module_conf.DIM,
        "embed_dims": bg_configs['embed_dim'],
        "adapter_layer": adapter_module_conf.ADAPTER_LAYER,
        "fft_layer": adapter_module_conf.FFT_LAYER,
        "cutoff_ratio": adapter_module_conf.CUTOFF_RATIO,
        "scale": adapter_module_conf.SCALE,
        "with_token": adapter_module_conf.WITH_TOKEN,
        "token_dim": adapter_module_conf.TOKEN_DIM,
    }



    if backbone_conf.NAME == 'vitl':
        interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
    elif backbone_conf.NAME == 'vitb':
        interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
    else:
        raise NotImplementedError(f"Backbone {backbone_conf.NAME} not supported")

    return {
        "vit_module": vit_backbone,
        "adapter_config": adapter_config,
        "interaction_indexes": interaction_indexes,
        "freeze_backbone": backbone_conf.FREEZE,
        "finetune": backbone_conf.FINETUNE,
        "finetune_indexes": backbone_conf.FINETUNE_INDEXES,
        "with_cp": backbone_conf.WITH_CP,
        "patch_size": bg_configs['patch_size']
    }


@BACKBONE_REGISTRY.register()
class AdapterDinoVisionTransformer(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()


        args = get_adapter_dino_args(cfg)
        self.vit_module = args['vit_module']
        self.adapter_config = args['adapter_config']
        self.interaction_indexes = args['interaction_indexes']
        self.freeze_backbone = args['freeze_backbone']
        self.finetune = args['finetune']
        self.finetune_indexes = args['finetune_indexes']
        if "type" in self.adapter_config:
            self.adapter_config.pop("type")
        self.adapter = adapter(**self.adapter_config)


        if self.freeze_backbone:
            for p in self.vit_module.parameters():
                p.requires_grad_(False)
            if self.finetune:
                for idx in self.finetune_indexes:
                    start, end = self.interaction_indexes[idx]
                    for blk in self.vit_module.blocks[start: end + 1]:
                        for p in blk.parameters():
                            p.requires_grad_(True)

        self._out_features = ["res2", "res3", "res4", "res5"]
        dim = self.vit_module.embed_dim
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {k: dim for k in self._out_features}
        self.patch_size = args["patch_size"]


    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            self.vit_module.eval()
            if self.finetune:
                for idx in self.finetune_indexes:
                    start, end = self.interaction_indexes[idx]
                    for blk in self.vit_module.blocks[start: end + 1]:
                        blk.train()

    def forward(self, x):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        if self.freeze_backbone and not self.finetune:
            with torch.no_grad():
                x = self.vit_module.prepare_tokens_with_masks(x)
        else:
            x = self.vit_module.prepare_tokens_with_masks(x)
        raw_feat_list = []
        outputs = {}
        for i, blk in enumerate(self.vit_module.blocks):
            if self.freeze_backbone and not self.finetune:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)

            x = self.adapter(x, layer=i, batch_first=True, has_cls_token=True,H=H, W=W)

            for out_idx, (start, end) in enumerate(self.interaction_indexes):
                if i == end:
                    raw_feat_list.append(x)
                    feat = x[:, 1:, :]
                    feat = feat.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                    res_name = self._out_features[out_idx]
                    if res_name == "res2":
                        feat = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
                    elif res_name == "res3":
                        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
                    elif res_name == "res5":
                        feat = F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False)
                    outputs[res_name] = feat
        return outputs
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }