import torch.nn as nn
import torch
import pdb

def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        "Feature map must be divisible by window_size."
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, B):
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

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
    def __init__(self, dim, patch_size=3, patch_stride=3, attn_drop=0.,
            window_size=2, shift_size=1, num_heads=1):
        super().__init__()
        self.psize = patch_size
        self.pstride = patch_stride  # Wh, Ww

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.embed_dim = dim
        self.hidden_dim = self.embed_dim // 2
        self.fc = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.scale = (self.hidden_dim * self.psize ** 2) ** -0.5
        # creat a mask for the calculations of AFB and BFB
        self.N = self.psize * self.pstride # patch num
        mask = torch.eye(self.N)
        mask[mask == 1] = -100.0
        self.register_buffer("mask", mask)

        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.window_size <= 1:
            self.shift_size = 0
        if self.shift_size >= self.window_size:
            self.shift_size = self.window_size - 1

        # We keep BFB lightweight and local: Swin-style windows with one shifted step,
        # intentionally without hierarchy/patch-merging to preserve GT-HAD structure.
        self.bfb_wmsa = WindowAttention(dim=self.embed_dim, window_size=self.window_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=attn_drop)
        self.bfb_swmsa = ShiftedWindowAttention(dim=self.embed_dim,
            window_size=self.window_size, shift_size=self.shift_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=attn_drop)

    def _get_cur_match(self, block_idx=0, match_vec=None):
        cur_match = torch.index_select(match_vec, dim=0, index=block_idx)
        return cur_match.view(-1)

    def calculate_mask(self, block_idx=0, match_vec=None):
        B = block_idx.size(0)
        # jduge which blocks are searched
        cur_match = self._get_cur_match(block_idx=block_idx, match_vec=match_vec)

        # all blocks undergo AFB
        if cur_match.sum() == 0:
            mask = self.mask
        else:
            mask = self.mask.unsqueeze(0).repeat(B, 1, 1)
            # searched blocks undergo BFB
            mask[cur_match == 1] = 0

        return mask

    def attn_cal(self, attn, mask, v, shape):
        B, H, W, C = shape
        attn = attn + mask
        attn = self.softmax(attn)

        x_attn = (attn @ v)
        x_back = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        return x

    def _legacy_attention_forward(self, x, block_idx=0, match_vec=None):
        B, H, W, C = x.shape
        N = self.N
        P = self.psize ** 2

        # attention calculation
        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, P, C)
        x_q = self.fc(x_fc).view(B, N, -1)
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)

        v = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P * C)
        if torch.is_tensor(block_idx) and match_vec is not None:
            mask = self.calculate_mask(block_idx=block_idx, match_vec=match_vec)
        else:
            mask = self.mask
        x = self.attn_cal(attn, mask, v, [B, H, W, C])
        assert x.shape == (B, H, W, C), "AFB/BFB legacy path failed to restore shape."

        return x

    def _bfb_shifted_window_forward(self, x):
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            "BFB window attention requires H/W divisible by window_size."
        # Shifted windows provide cross-window context while remaining local, which
        # improves background consistency without switching to global attention.
        x = self.bfb_wmsa(x)
        x = self.bfb_swmsa(x)
        assert x.shape == (B, H, W, C), "BFB shifted-window path failed to restore shape."
        return x

    def forward(self, x, block_idx=0, match_vec=None):
        B, H, W, C = x.shape
        if not torch.is_tensor(block_idx) or match_vec is None:
            return self._legacy_attention_forward(x, block_idx=block_idx, match_vec=match_vec)

        cur_match = self._get_cur_match(block_idx=block_idx, match_vec=match_vec)
        afb_mask = cur_match != 1
        bfb_mask = cur_match == 1

        assert afb_mask.dtype == torch.bool and bfb_mask.dtype == torch.bool, \
            "Mixed-batch branch masks must be boolean."
        assert int(afb_mask.sum().item() + bfb_mask.sum().item()) == B, \
            "Mixed-batch correctness failed: AFB/BFB split does not cover batch."
        assert int((afb_mask & bfb_mask).sum().item()) == 0, \
            "Mixed-batch correctness failed: overlapping AFB/BFB indices."

        if int(bfb_mask.sum().item()) == 0:
            return self._legacy_attention_forward(x, block_idx=block_idx, match_vec=match_vec)

        if int(afb_mask.sum().item()) == 0:
            return self._bfb_shifted_window_forward(x)

        out = torch.empty_like(x)
        out[afb_mask] = self._legacy_attention_forward(
            x[afb_mask], block_idx=block_idx[afb_mask], match_vec=match_vec
        )
        out[bfb_mask] = self._bfb_shifted_window_forward(x[bfb_mask])
        assert out.shape == x.shape, "Mixed-batch output shape mismatch."

        # Preserving AFB behavior is critical for anomaly residual separability in HAD.
        # Only BFB receives the shifted-window replacement as a controlled modification.
        return out

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=2, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_mask=None):
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            "WindowAttention requires H/W divisible by window_size."
        windows = window_partition(x, self.window_size)
        num_windows_per_sample = (H // self.window_size) * (W // self.window_size)
        tokens = self.window_size * self.window_size

        qkv = self.qkv(windows).view(-1, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            assert attn_mask.shape == (num_windows_per_sample, tokens, tokens), \
                "Shifted window mask shape mismatch."
            attn = attn.view(B, num_windows_per_sample, self.num_heads, tokens, tokens)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(-1, tokens, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = window_reverse(out, self.window_size, H, W, B)
        assert out.shape == (B, H, W, C), "WindowAttention failed to restore feature shape."
        return out

class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, window_size=2, shift_size=1, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must satisfy 0 <= shift_size < window_size."
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_attn = WindowAttention(dim=dim, window_size=window_size,
            num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

    def _build_shift_mask(self, H, W, device):
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        B, H, W, C = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            "ShiftedWindowAttention requires H/W divisible by window_size."
        if self.shift_size == 0:
            return self.window_attn(x, attn_mask=None)

        shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        attn_mask = self._build_shift_mask(H, W, x.device)
        shifted_out = self.window_attn(shifted, attn_mask=attn_mask)
        out = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        assert out.shape == (B, H, W, C), "ShiftedWindowAttention failed to restore feature shape."
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, patch_size=3, patch_stride=3, mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0., drop=0.,
            window_size=2, shift_size=1, num_heads=1):
        super().__init__()

        # GTB: GDBN
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, patch_size=patch_size, patch_stride=patch_stride,
            attn_drop=attn_drop, window_size=window_size, shift_size=shift_size, num_heads=num_heads)
        # GTB: FFN
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, block_idx=0, match_vec=None):
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        x = self.norm1(x)
        # GDBN
        x = x.view(B, H, W, C)
        x = self.attn(x, block_idx=block_idx, match_vec=match_vec)
        # FFN
        x = x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, H, W, C)

        return x

class Net(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=3, patch_stride=3,
            mlp_ratio=2., attn_drop=0., drop=0., window_size=2, shift_size=1, num_heads=1):
        super(Net, self).__init__()
        # head
        self.conv_head = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # attn_layer
        self.attn_layer = TransformerBlock(embed_dim, patch_size=patch_size,
            patch_stride=patch_stride, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop,
            window_size=window_size, shift_size=shift_size, num_heads=num_heads)
        # tail
        self.conv_tail = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x, block_idx=0, match_vec=None):
        x = self.conv_head(x) # b, c, h, w
        x = x.permute(0, 2, 3, 1).contiguous() # b, h, w, c
        x = self.attn_layer(x, block_idx=block_idx, match_vec=match_vec)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_tail(x)

        return x
