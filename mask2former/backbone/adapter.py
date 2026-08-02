
import math
import torch
import torch.nn as nn
from torch import Tensor


class adapter(nn.Module):
    def __init__(
            self,
            dim=64,
            embed_dims=1024,
            adapter_layer=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            fft_layer=[],
            with_token=False,
            token_dim=0,
            cutoff_ratio=0.3,
            scale=0.1,

    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dims
        self.fft_layer = fft_layer
        self.token_dim = token_dim
        self.adapter_layer = adapter_layer
        self.with_token = with_token
        self.cutoff_ratio = cutoff_ratio


        num_layers = max(adapter_layer) + 1 if adapter_layer else 24
        self.scale = nn.Parameter(torch.tensor([scale] * num_layers))


        if self.with_token:
            self.refine_token = nn.Parameter(torch.empty([num_layers, 1, self.token_dim]))

        if self.with_token:
            self.mlp_list1 = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.embed_dim + self.token_dim, self.dim), nn.ReLU(),
                               nn.Linear(self.dim, self.embed_dim)) for _ in range(num_layers)])
        else:
            self.mlp_list1 = nn.ModuleList(
                [nn.Sequential(nn.Linear(self.embed_dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.embed_dim)) for
                 _ in range(num_layers)])


        if len(self.fft_layer) > 0:
            self.freq_conv_low = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.dim, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim, self.embed_dim, kernel_size=1, bias=False)
                ) for _ in range(num_layers)
            ])

            self.freq_conv_mid = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.dim, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim, self.embed_dim, kernel_size=1, bias=False)
                ) for _ in range(num_layers)
            ])

            self.freq_conv_high = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.dim, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim, self.embed_dim, kernel_size=1, bias=False)
                ) for _ in range(num_layers)
            ])
            self.router = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(self.embed_dim, 3, kernel_size=1))
                                        for _ in range(num_layers)])
            self.fft_norm = nn.LayerNorm(self.embed_dim)

            self.mlp_list2 = nn.ModuleList(
                    [nn.Sequential(nn.Linear(self.embed_dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.embed_dim)) for
                     _ in range(num_layers)])

            for seq in self.freq_conv_low + self.freq_conv_mid + self.freq_conv_high+self.mlp_list2:
                nn.init.kaiming_uniform_(seq[0].weight, a=math.sqrt(5))
                nn.init.constant_(seq[2].weight, 0.0)



        else:
            for mlp in self.mlp_list1:
                nn.init.kaiming_uniform_(mlp[0].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mlp[2].weight, a=math.sqrt(5))

    def forward(
            self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True, H=None, W=None
    ) -> Tensor:
        if layer not in self.adapter_layer:
            return feats

        feats = feats.permute(1, 0, 2)

        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        if self.with_token:
            tokens = self.refine_token[layer].expand(feats.shape[0], feats.shape[1], -1)
            combined_feats = torch.cat([feats, tokens], dim=-1)
        else:
            combined_feats = feats

        if layer not in self.fft_layer:
            delta_feat = self.mlp_list1[layer](combined_feats)
            feats = feats + self.scale[layer] * delta_feat


        else:
            seq_len, batch_size, channels = feats.shape
            feats_spatial = feats.permute(1, 2, 0).reshape(batch_size, channels, H, W)
            fft_complex = torch.fft.fft2(feats_spatial, norm='ortho')
            fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
            A = torch.abs(fft_shifted)
            P = torch.angle(fft_shifted)
            Y, X = torch.meshgrid(
                torch.arange(H, device=feats.device),
                torch.arange(W, device=feats.device),
                indexing='ij'
            )
            cx, cy = H // 2, W // 2
            R = torch.sqrt((Y - cx) ** 2 + (X - cy) ** 2)

            low_ratio = 0.20
            high_ratio = 0.50

            r1 = min(H, W) * low_ratio
            r2 = min(H, W) * high_ratio
            G1 = torch.exp(-(R ** 2) / (2 * (r1 ** 2 + 1e-5)))
            G2 = torch.exp(-(R ** 2) / (2 * (r2 ** 2 + 1e-5)))

            mask_low = G1.unsqueeze(0).unsqueeze(0)

            mask_mid = (G2 - G1).unsqueeze(0).unsqueeze(0)

            mask_high = (1.0 - G2).unsqueeze(0).unsqueeze(0)

            A_low = A * mask_low
            A_mid = A * mask_mid
            A_high = A * mask_high

            A_low_tilde = self.freq_conv_low[layer](A_low)
            A_mid_tilde = self.freq_conv_mid[layer](A_mid)
            A_high_tilde = self.freq_conv_high[layer](A_high)

            router_logits = self.router[layer](A)
            router_weights = torch.softmax(router_logits, dim=1)  # [B, 3]

            W_low = router_weights[:, 0].view(batch_size, 1, 1, 1)
            W_mid = router_weights[:, 1].view(batch_size, 1, 1, 1)
            W_high = router_weights[:, 2].view(batch_size, 1, 1, 1)


            delta_A_total = W_low*A_low_tilde + W_mid*A_mid_tilde + W_high*A_high_tilde

            A_new_ishift = torch.fft.ifftshift(delta_A_total, dim=(-2, -1))
            P_ishift = torch.fft.ifftshift(P, dim=(-2, -1))
            fft_new = torch.polar(A_new_ishift, P_ishift)

            feats_mod_spatial = torch.fft.ifft2(fft_new, norm='ortho').real


            delta_feat_1 = feats_mod_spatial.reshape(batch_size, channels, -1).permute(2, 0, 1)
            delta_feat_1 = self.fft_norm(delta_feat_1)


            delta_feat = self.mlp_list2[layer](combined_feats)
            delta_feat= delta_feat   +  delta_feat_1
            feats = feats + self.scale[layer] * delta_feat


        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)

        if batch_first:
            feats = feats.permute(1, 0, 2)

        return feats