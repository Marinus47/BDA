import math
import weakref
from functools import partial
from typing import Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY, Backbone


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Output: [B, Hp, Wp, C]
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # [B, C, Hp, Wp]
        x = x.permute(0, 2, 3, 1)     # [B, Hp, Wp, C]
        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    """
    SAM original attention, input/output are [B, H, W, C]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        global_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None

            if global_attn:
                self.rel_pos_h = nn.Parameter(torch.zeros(4 * input_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(4 * input_size[1] - 1, head_dim))
            else:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if rel_pos_zero_init:
                nn.init.zeros_(self.rel_pos_h)
                nn.init.zeros_(self.rel_pos_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)

        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class Block(nn.Module):
    """
    SAM original transformer block, input/output are [B, H, W, C]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            global_attn=(window_size == 0),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer,
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SAMBlockSequenceAdapter(nn.Module):
    """
    把 SAM 的 BHWC block 包成 DINOv2 风格的 [B, 1+HW, C] block
    外部 adapter 可直接 enumerate(self.blocks)
    """

    def __init__(self, block: nn.Module, owner_ref):
        super().__init__()
        self.block = block
        self.owner_ref = owner_ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        owner = self.owner_ref()
        assert owner is not None, "Backbone reference is gone."

        H, W = owner._token_hw
        B, N, C = x.shape

        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        assert patch_tokens.shape[1] == H * W, (
            f"Patch token num {patch_tokens.shape[1]} != H*W {H*W}"
        )

        patch_tokens = patch_tokens.reshape(B, H, W, C)
        patch_tokens = self.block(patch_tokens)
        patch_tokens = patch_tokens.reshape(B, H * W, C)

        x = torch.cat([cls_token, patch_tokens], dim=1)
        return x


@BACKBONE_REGISTRY.register()
class SAMVisionTransformer(Backbone):
    """
    对外兼容 DINOv2 模板：
    - prepare_tokens_with_masks
    - forward_features
    - get_intermediate_layers
    - forward
    - self.blocks 可逐层遍历
    """

    def __init__(
        self,
        img_size: int = 1024,
        out_indices: Sequence[int] = (5, 11, 17, 23),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (5, 11, 17, 23),
        init_cfg=None,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.out_indices = list(out_indices)
        self.patch_size = patch_size

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.chunked_blocks = False

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    img_size // patch_size,
                    img_size // patch_size,
                    embed_dim,
                )
            )

        raw_blocks = []
        for i in range(depth):
            blk = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            raw_blocks.append(blk)

        owner_ref = weakref.ref(self)
        self.blocks = nn.ModuleList(
            [SAMBlockSequenceAdapter(blk, owner_ref) for blk in raw_blocks]
        )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self._token_hw = None

        self.configs_dict = {
            "img_size": img_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "out_indices": list(out_indices),
            "use_abs_pos": use_abs_pos,
            "use_rel_pos": use_rel_pos,
            "window_size": window_size,
            "global_attn_indexes": list(global_attn_indexes),
        }

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        dtype = x.dtype
        B, N, C = x.shape
        npatch = N - 1

        Hp = h // self.patch_size
        Wp = w // self.patch_size
        assert npatch == Hp * Wp, f"npatch={npatch}, but Hp*Wp={Hp*Wp}"

        cls_pos = torch.zeros(1, 1, C, device=x.device, dtype=torch.float32)

        if self.pos_embed is None:
            patch_pos = torch.zeros(1, Hp, Wp, C, device=x.device, dtype=torch.float32)
        else:
            pos_embed = self.pos_embed.float()
            H0, W0 = pos_embed.shape[1], pos_embed.shape[2]
            if H0 == Hp and W0 == Wp:
                patch_pos = pos_embed
            else:
                patch_pos = F.interpolate(
                    pos_embed.permute(0, 3, 1, 2),
                    size=(Hp, Wp),
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

        patch_pos = patch_pos.reshape(1, Hp * Wp, C)
        pos = torch.cat([cls_pos, patch_pos], dim=1).to(dtype)
        return pos

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # [B, Hp, Wp, C]
        Hp, Wp = x.shape[1], x.shape[2]
        self._token_hw = (Hp, Wp)

        if masks is not None:
            if masks.ndim == 2:
                masks = masks.view(B, Hp, Wp)
            elif masks.ndim == 3:
                pass
            else:
                raise ValueError(f"Unsupported masks shape: {masks.shape}")

            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).view(1, 1, 1, -1),
                x,
            )

        x = x.reshape(B, Hp * Wp, -1)
        cls_token = self.cls_token.expand(B, -1, -1).to(x.dtype)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, W, H)
        return x

    def forward_features_list(self, x_list, masks_list):
        outputs = []
        for x, masks in zip(x_list, masks_list):
            tokens = self.prepare_tokens_with_masks(x, masks)
            for blk in self.blocks:
                tokens = blk(tokens)

            x_norm = self.norm(tokens)
            outputs.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "x_prenorm": tokens,
                    "masks": masks,
                }
            )
        return outputs

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        x = self.prepare_tokens_with_masks(x, masks)

        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.out_indices:
                feat = (
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(B, -1, Hp, Wp)
                    .contiguous()
                )
                outs.append(feat)
        return outs

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)

        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]

        if reshape:
            B, _, H, W = x.shape
            Hp, Wp = H // self.patch_size, W // self.patch_size
            outputs = [
                out.reshape(B, Hp, Wp, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)

        if isinstance(ret[0], torch.Tensor):
            ret[0] = F.interpolate(
                ret[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[1] = F.interpolate(
                ret[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[3] = F.interpolate(
                ret[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
        else:
            ret[0][0] = F.interpolate(
                ret[0][0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[0][1] = F.interpolate(
                ret[0][1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[0][3] = F.interpolate(
                ret[0][3], scale_factor=0.5, mode="bilinear", align_corners=False
            )

        return ret


def _resize_abs_pos_embed_bhwc(pos_embed: torch.Tensor, target_hw: Tuple[int, int]):
    """
    pos_embed: [1, H, W, C]
    """
    if pos_embed.ndim != 4:
        return pos_embed

    target_h, target_w = target_hw
    if pos_embed.shape[1] == target_h and pos_embed.shape[2] == target_w:
        return pos_embed

    pos_embed = F.interpolate(
        pos_embed.permute(0, 3, 1, 2),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return pos_embed


def _resize_rel_pos(rel_pos: torch.Tensor, target_len: int):
    """
    rel_pos: [L, C]
    """
    if rel_pos.ndim != 2 or rel_pos.shape[0] == target_len:
        return rel_pos

    rel_pos = F.interpolate(
        rel_pos.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).transpose(0, 1)
    return rel_pos


def _remap_sam_state_dict_keys(state_dict):
    """
    官方 SAM checkpoint 常见情况：
    - image_encoder.xxx
    - module.image_encoder.xxx
    - backbone.xxx
    """
    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("image_encoder."):
            k = k[len("image_encoder."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]

        # 我们的 block 被包成了 blocks.i.block.xxx
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                k = f"blocks.{parts[1]}.block." + ".".join(parts[2:])

        new_state_dict[k] = v

    return new_state_dict


def load_sam_pretrained(model: nn.Module, weights: str):
    checkpoint = torch.load(weights, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    checkpoint = _remap_sam_state_dict_keys(checkpoint)
    model_state = model.state_dict()
    load_state = {}

    for k, v in checkpoint.items():
        if k not in model_state:
            continue

        target_shape = model_state[k].shape
        if v.shape != target_shape:
            if k == "pos_embed":
                v = _resize_abs_pos_embed_bhwc(v, target_shape[1:3])
            elif k.endswith("rel_pos_h") or k.endswith("rel_pos_w"):
                v = _resize_rel_pos(v, target_shape[0])
            elif k == "cls_token" and v.ndim == 2:
                v = v.unsqueeze(1)
            else:
                continue

            if v.shape != target_shape:
                continue

        load_state[k] = v

    msg = model.load_state_dict(load_state, strict=False)
    return msg


def get_models(model_name="vitl", weights=None):
    model_name = model_name.lower()

    model_cfgs = {
        "vitl": dict(
            img_size=1024,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=(5, 11, 17, 23),
            out_indices=(5, 11, 17, 23),
        ),
        "vitb": dict(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=(2, 5, 8, 11),
            out_indices=(2, 5, 8, 11),
        ),
        "vith": dict(
            img_size=1024,
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            window_size=14,
            global_attn_indexes=(7, 15, 23, 31),
            out_indices=(7, 15, 23, 31),
        ),
    }

    if model_name not in model_cfgs:
        raise NotImplementedError(f"SAM backbone {model_name} is not supported")

    model = SAMVisionTransformer(**model_cfgs[model_name])

    if weights is not None and weights != "":
        load_sam_pretrained(model, weights)

    return model