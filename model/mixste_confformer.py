import sys
import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from einops import rearrange, repeat
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., layer_idx=0,
                 total_layers=1):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layer_idx = layer_idx
        self.total_layers = total_layers

    def forward(self, x, confidence=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.total_layers > 1:
            conf_weight = max(0.0, 1.0 - float(self.layer_idx) / (self.total_layers - 1))
        else:
            conf_weight = 1.0
        nonlinear_c = torch.sin(confidence * torch.pi / 2)
        interp_mask = (1.0 - conf_weight) + conf_weight * nonlinear_c
        attn = attn * interp_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, layer_idx=0, total_layers=1):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, layer_idx=layer_idx,
                              total_layers=total_layers)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, confidence=None):
        x = x + self.drop_path(self.attn(self.norm1(x), confidence))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
def compute_temporal_confidence(poses, sigma=1.0, eps=1e-6, per_joint=True, order=2):
    """
    Compute temporal confidence based on motion smoothness (velocity, acceleration, or jerk).

    Args:
        poses: Tensor of shape (B, T, J, C)
        sigma: sensitivity to motion magnitude
        eps: numerical stability
        per_joint: if True, return (B, T, J); else return (B, T)
        order: int, order of motion derivative:
               1 → velocity (1st order)
               2 → acceleration (2nd order) ← default
               3 → jerk (3rd order)

    Returns:
        conf:
            - (B, T, J) if per_joint=True
            - (B, T)    if per_joint=False
    """
    B, T, J, C = poses.shape

    # Determine minimum required frames
    min_frames = order + 1
    if T < min_frames:
        if per_joint:
            return torch.ones(B, T, J, device=poses.device, dtype=poses.dtype)
        else:
            return torch.ones(B, T, device=poses.device, dtype=poses.dtype)

    # Compute motion derivative of given order
    motion = poses.clone()
    for _ in range(order):
        motion = motion[:, 1:] - motion[:, :-1]  # finite difference

    # motion shape: (B, T - order, J, C)

    if per_joint:
        # L2 norm over coordinate dim (C), keep joint dim (J)
        motion_norm = torch.norm(motion, dim=-1)  # (B, T - order, J)
        conf_mid = torch.exp(-sigma * motion_norm)  # (B, T - order, J)
        # Pad to full length T using replication
        pad_left = order // 2
        pad_right = order - pad_left
        conf = F.pad(conf_mid, (0, 0, pad_left, pad_right), mode='replicate')  # (B, T, J)
    else:
        # Average over joints first, then compute norm
        motion_norm = torch.norm(motion, dim=-1).mean(dim=-1)  # (B, T - order)
        conf_mid = torch.exp(-sigma * motion_norm)
        pad_left = order // 2
        pad_right = order - pad_left
        conf = F.pad(conf_mid, (pad_left, pad_right), mode='replicate')  # (B, T)

    return conf.clamp(min=eps, max=1.0)

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        depth = 8
        embed_dim = args.channel
        mlp_hidden_dim = args.d_hid

        drop_path_rate = 0.1
        drop_rate = 0.
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        num_heads = 8
        num_joints = args.n_joints

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(2, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, args.frames, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                layer_idx=i,
                total_layers=depth)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, depth=depth,
                layer_idx=i,
                total_layers=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, x):
        b, f, n, c = x.shape
        confidence = x[..., -1].clamp(0.0, 1.0)  # (b, f, n)
        x = x[..., :-1]
        temporal_confidence = compute_temporal_confidence(x, sigma=2.0, order=2)
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        confidence_spatial = rearrange(confidence, 'b f n -> (b f) n')
        confidence_spatial = confidence_spatial.unsqueeze(1)
        confidence_spatial=confidence_spatial.transpose(-2, -1)@confidence_spatial
        confidence_spatial = confidence_spatial.unsqueeze(1)
        x = self.STEblocks[0](x, confidence_spatial)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        confidence_temporal = rearrange(temporal_confidence, 'b f n -> (b n) f')
        confidence_temporal = confidence_temporal.unsqueeze(1)
        confidence_temporal = confidence_temporal.transpose(-2, -1) @ confidence_temporal
        confidence_temporal = confidence_temporal.unsqueeze(1)
        x = self.TTEblocks[0](x, confidence_temporal)
        x = self.Temporal_norm(x)

        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]


            x = steblock(x, confidence_spatial)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', f=f)


            x = tteblock(x, confidence_temporal)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        x = self.head(x)

        x = x.view(b, f, n, -1)

        return x


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser().parse_args()
    args.layers, args.channel, args.d_hid, args.frames = 8, 512, 1024, 243
    args.n_joints, args.out_joints = 17, 17
    args.token_num = 81
    args.layer_index = 3

    input_2d = torch.rand(1, args.frames, 17, 3)  # (x, y, confidence)

    with torch.no_grad():
        model = Model(args)
        model.eval()

        model_params = 0
        for parameter in model.parameters():
            model_params += parameter.numel()
        print('INFO: Trainable parameter count:', model_params / 1000000)

        print(input_2d.shape, 1)
        output = model(input_2d)
        print(output.shape, 2)

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(input_2d,))
    print('macs: ', macs / 1000000, 'params: ', params / 1000000)
    macs, params = clever_format([macs * 2, params], "%.3f")
    print(macs, params)
