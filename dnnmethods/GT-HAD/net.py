import torch.nn as nn
import torch


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, patch_size=3, patch_stride=3, attn_drop=0.):
        super().__init__()
        self.psize = patch_size
        self.pstride = patch_stride

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.embed_dim = dim
        self.hidden_dim = self.embed_dim // 2

        self.N = self.pstride * self.pstride
        P = self.psize * self.psize
        self.scale = (self.hidden_dim * P) ** -0.5

        self.fc = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.fc_out = nn.Linear(self.hidden_dim, self.embed_dim, bias=True)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def _branch_forward(self, x_q, x_v, mask):
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)
        attn = attn + mask
        attn = self.softmax(attn)
        return attn @ x_v

    def _tokens_to_spatial(self, x_tok, B, H, W):
        x = x_tok.view(B, self.pstride, self.pstride, self.hidden_dim)
        x = x.unsqueeze(2).unsqueeze(4)
        x = x.expand(B, self.pstride, self.psize, self.pstride, self.psize,
                      self.hidden_dim).contiguous()
        return x.view(B, H, W, self.hidden_dim)

    def forward(self, x, fi=None):
        B, H, W, C = x.shape
        N = self.N
        P = self.psize ** 2

        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_patch = x_view.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_flat = x_patch.view(B, N, P, C)
        x_proj = self.fc(x_flat)
        x_q = x_proj.mean(dim=2)
        x_v = x_q

        afb_mask = torch.eye(N, device=x.device) * -100.0
        bfb_mask = torch.zeros(N, N, device=x.device)

        afb_tok = self._branch_forward(x_q, x_v, afb_mask)
        bfb_tok = self._branch_forward(x_q, x_v, bfb_mask)

        afb_out = self._tokens_to_spatial(afb_tok, B, H, W)
        bfb_out = self._tokens_to_spatial(bfb_tok, B, H, W)

        if fi is not None:
            gi = self.gate(fi)
            gi_h = gi[:, :self.hidden_dim]
            gi_h = gi_h.view(B, 1, 1, self.hidden_dim)
            self.last_gate = gi.unsqueeze(-1).unsqueeze(-1)
        else:
            gi_h = torch.full((B, 1, 1, self.hidden_dim), 0.5, device=x.device)
            self.last_gate = None

        x_fused = gi_h * bfb_out + (1 - gi_h) * afb_out
        x_out = self.fc_out(x_fused)
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_size=3, patch_stride=3, mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0., drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, patch_size=patch_size,
                              patch_stride=patch_stride, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, fi=None):
        B, H, W, C = x.shape

        x_norm = self.norm1(x.view(B, H * W, C)).view(B, H, W, C)
        x = x + self.attn(x_norm, fi=fi)

        x_flat = x.view(B, H * W, C)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.view(B, H, W, C)


class Net(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=3, patch_stride=3,
            mlp_ratio=2., attn_drop=0., drop=0., lambda_gate=0.01):
        super(Net, self).__init__()

        self.lambda_gate = lambda_gate
        self.conv_head = nn.Conv2d(in_chans, embed_dim, kernel_size=3,
                                   stride=1, padding=1)
        self.attn_layer = TransformerBlock(embed_dim, patch_size=patch_size,
            patch_stride=patch_stride, mlp_ratio=mlp_ratio,
            attn_drop=attn_drop, drop=drop)
        self.conv_tail = nn.Conv2d(embed_dim, in_chans, kernel_size=3,
                                   stride=1, padding=1)

    def forward(self, x):
        fi = self.conv_head(x)
        x_t = fi.permute(0, 2, 3, 1).contiguous()
        x_t = self.attn_layer(x_t, fi=fi)
        x_t = x_t.permute(0, 3, 1, 2).contiguous()
        return self.conv_tail(x_t)

    def compute_gate_loss(self):
        attn = self.attn_layer.attn
        if not hasattr(attn, 'last_gate') or attn.last_gate is None:
            return torch.tensor(0.0).to(next(self.parameters()).device)
        gi = attn.last_gate
        return -torch.mean(
            gi * torch.log(gi + 1e-8) +
            (1 - gi) * torch.log(1 - gi + 1e-8)
        )
