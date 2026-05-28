import torch.nn as nn
import torch
import pdb

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
    """
    Attention module with differentiable soft gating for branch fusion.
    
    Replaces the original hard routing (CMM-based) with a learnable soft-gating mechanism.
    Both AFB and BFB branches always execute, and their outputs are fused using a 
    differentiable sigmoid gate that can be optimized end-to-end.
    """
    def __init__(self, dim, patch_size=3, patch_stride=3, attn_drop=0.):
        super().__init__()
        self.psize = patch_size
        self.pstride = patch_stride  # Wh, Ww

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.embed_dim = dim
        self.hidden_dim = self.embed_dim // 2
        self.fc = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.scale = (self.hidden_dim * self.psize ** 2) ** -0.5
        
        # Create masks for AFB and BFB branches
        self.N = self.psize * self.pstride # patch num
        self.afb_mask = torch.eye(self.N).cuda()
        self.afb_mask[self.afb_mask == 1] = -100.0
        self.bfb_mask = torch.zeros(self.N, self.N).cuda()
        
        # Learnable differentiable soft gating mechanism
        # Global Average Pooling -> MLP -> Sigmoid
        # This replaces the CMM-based hard routing with a fully differentiable gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def afb_forward(self, x_q, x_v, shape):
        """
        Augmented Feature Branch (AFB): Self-attention within patches.
        """
        B, H, W, C = shape
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)
        attn = attn + self.afb_mask
        attn = self.softmax(attn)
        
        x_attn = (attn @ x_v)
        x_back = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        
        return x

    def bfb_forward(self, x_q, x_v, shape):
        """
        Background Feature Branch (BFB): Cross-patch attention.
        All patches attend to all patches without masking.
        """
        B, H, W, C = shape
        attn = x_q @ x_q.transpose(-2, -1)
        attn = attn * self.scale
        attn = self.attn_drop(attn)
        attn = attn + self.bfb_mask
        attn = self.softmax(attn)
        
        x_attn = (attn @ x_v)
        x_back = x_attn.view(B, self.pstride, self.pstride, self.psize, self.psize, C)
        x = x_back.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        
        return x

    def forward(self, x, fi=None):
        """
        Forward pass with differentiable soft gating.
        
        Args:
            x: Input tensor [B, H, W, C]
            fi: Original feature map for gate computation [B, C, H, W]
            
        Returns:
            Soft-fused output of AFB and BFB branches [B, H, W, C]
        """
        B, H, W, C = x.shape
        N = self.N
        P = self.psize ** 2

        # Prepare query and value for attention computation
        x_view = x.view(B, self.pstride, self.psize, self.pstride, self.psize, C)
        x_fc = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, P, C)
        x_q = self.fc(x_fc).view(B, N, -1)
        
        v = x_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, N, P * C)
        
        shape = [B, H, W, C]
        
        # Both branches always execute (differentiable)
        afb_out = self.afb_forward(x_q, v, shape)
        bfb_out = self.bfb_forward(x_q, v, shape)
        
        # Compute learnable soft gate
        # Gate network takes the original feature map fi and outputs gating weights
        if fi is not None:
            gi = self.gate(fi)  # [B, C]
            # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
            gi = gi.unsqueeze(-1).unsqueeze(-1)
            # Prepare for fusion with spatial dimensions [B, H, W, C]
            gi_spatial = gi.view(B, C, 1, 1).expand(-1, -1, H, W)
            gi_spatial = gi_spatial.permute(0, 2, 3, 1)  # [B, H, W, C]
        else:
            # If fi not provided, use uniform gate (fallback)
            gi_spatial = torch.ones(B, H, W, C).to(x.device) * 0.5
        
        # Soft differentiable fusion: gi * BFB + (1 - gi) * AFB
        # This allows gradients to flow through both branches
        x_fused = gi_spatial * bfb_out + (1 - gi_spatial) * afb_out
        
        # Store gate for regularization loss computation
        self.last_gate = gi if fi is not None else None
        
        return x_fused


class TransformerBlock(nn.Module):
    """
    Transformer block with differentiable soft-gating mechanism.
    
    Pipeline: LayerNorm -> Soft-gated Attention -> FFN
    """
    def __init__(self, dim, patch_size=3, patch_stride=3, mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_drop=0., drop=0.):
        super().__init__()

        # GTB: Layer normalization
        self.norm1 = norm_layer(dim)
        # Attention with differentiable soft gating
        self.attn = Attention(dim, patch_size=patch_size, patch_stride=patch_stride, attn_drop=attn_drop)
        # GTB: FFN (Feed Forward Network)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, fi=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, H, W, C]
            fi: Original feature map from conv layer [B, C, H, W]
            
        Returns:
            Output tensor [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # Layer norm + soft-gated attention
        x_norm = x.view(B, H * W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.view(B, H, W, C)
        x_attn = self.attn(x_norm, fi=fi)
        
        # FFN with residual connection
        x_out = x + x_attn
        x_view = x_out.view(B, H * W, C)
        x_ffn = self.norm2(x_view)
        x_ffn = x_view + self.mlp(x_ffn)
        x_out = x_ffn.view(B, H, W, C)
        
        return x_out


class Net(nn.Module):
    """
    GT-HAD network with differentiable soft-gating mechanism.
    
    Architecture: Conv -> Gate Network -> AFB/BFB -> Soft Fusion -> FFN -> Reconstruction
    
    The gate network replaces the original CMM-based hard routing with a fully 
    differentiable mechanism that learns to route features through both branches.
    """
    def __init__(self, in_chans=3, embed_dim=96, patch_size=3, patch_stride=3,
            mlp_ratio=2., attn_drop=0., drop=0., lambda_gate=0.01):
        super(Net, self).__init__()
        
        # Hyperparameter for gate entropy regularization
        self.lambda_gate = lambda_gate
        
        # Convolutional head: produces initial feature map
        self.conv_head = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Transformer block with soft-gated attention
        self.attn_layer = TransformerBlock(embed_dim, patch_size=patch_size,
            patch_stride=patch_stride, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop=drop)
        
        # Convolutional tail: reconstruction head
        self.conv_tail = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Reconstructed output [B, C, H, W]
        """
        # Initial convolution to produce feature map fi
        fi = self.conv_head(x)  # [B, embed_dim, H, W]
        
        # Permute to [B, H, W, C] for transformer
        x_t = fi.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Soft-gated attention layer (passes fi for gate computation)
        x_t = self.attn_layer(x_t, fi=fi)
        
        # Permute back to [B, C, H, W]
        x_t = x_t.permute(0, 3, 1, 2).contiguous()
        
        # Reconstruction head
        out = self.conv_tail(x_t)
        
        return out

    def compute_gate_loss(self):
        """
        Compute entropy regularization loss for the gate.
        
        Prevents gate collapse (all 0s or all 1s) by encouraging 
        the gate to maintain balanced routing between AFB and BFB branches.
        
        Returns:
            gate_loss: Entropy regularization loss
        """
        if not hasattr(self.attn_layer.attn, 'last_gate') or self.attn_layer.attn.last_gate is None:
            return torch.tensor(0.0).to(next(self.parameters()).device)
        
        gi = self.attn_layer.attn.last_gate
        
        # Binary entropy regularization
        # Encourages gate to be neither too high nor too low
        gate_loss = -torch.mean(
            gi * torch.log(gi + 1e-8) +
            (1 - gi) * torch.log(1 - gi + 1e-8)
        )
        
        return gate_loss
