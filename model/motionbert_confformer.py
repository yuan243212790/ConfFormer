"""MotionBERT/DSTformer with the author-provided ConfFormer modulation.

Input: (batch, frames, joints, 3), with channels (x, y, confidence).
Only xy enters the coordinate embedding. Requires PyTorch and einops.
"""
import torch
import torch.nn as nn
import math
import warnings
from collections import OrderedDict
from functools import partial
from einops import rearrange
import torch.nn.functional as F

class DropPath(nn.Module):
    """Per-sample stochastic depth, independent of the original repository."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        return x * mask / keep_prob


def compute_temporal_confidence(poses, sigma=1.0, eps=1e-6, per_joint=True):
    """
    Compute temporal confidence based on motion smoothness (acceleration).

    Args:
        poses: Tensor of shape (B, T, J, C)
        sigma: sensitivity to acceleration
        eps: numerical stability
        per_joint: if True, return (B, T, J); else return (B, T)

    Returns:
        conf:
            - (B, T, J) if per_joint=True
            - (B, T)    if per_joint=False
    """
    B, T, J, C = poses.shape

    if T < 3:
        if per_joint:
            return torch.ones(B, T, J, device=poses.device, dtype=poses.dtype)
        else:
            return torch.ones(B, T, device=poses.device, dtype=poses.dtype)

    # Velocity: (B, T-1, J, C)
    vel = poses[:, 1:] - poses[:, :-1]
    # Acceleration: (B, T-2, J, C)
    acc = vel[:, 1:] - vel[:, :-1]

    if per_joint:
        # Compute L2 norm over coordinate dim (C), keep joint dim (J)
        acc_norm = torch.norm(acc, dim=-1)  # (B, T-2, J)
        conf_mid =torch.exp(-sigma * acc_norm)  # (B, T-2, J)
        # Pad to full length T: replicate first/last along time
        conf = F.pad(conf_mid, (0, 0, 1, 1), mode='replicate')  # (B, T, J)
    else:
        # Original: average over joints
        acc_norm = torch.norm(acc, dim=-1).mean(dim=-1)  # (B, T-2)
        conf_mid = torch.exp(-sigma * acc_norm)
        conf = F.pad(conf_mid, (1, 1), mode='replicate')  # (B, T)

    return conf.clamp(min=eps, max=1.0)
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MLP(nn.Module):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla',
                 layer_idx=0, total_layers=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = st_mode
        self.conf_mode = 'attn'  # 固定使用 attn 模式
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, seqlen=1, joint_confidence=None):
        B, N, C = x.shape
        assert B % seqlen == 0
        local_batch = B // seqlen

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        processed_conf = None
        if joint_confidence is not None and self.conf_mode == 'attn':

            processed_conf = joint_confidence


        if self.mode == 'series':
            x = self.forward_spatial(q, k, v, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
            qkv2 = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            x = self.forward_temporal(q2, k2, v2, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
        elif self.mode == 'parallel':
            x_t = self.forward_temporal(q, k, v, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
            x_s = self.forward_spatial(q, k, v, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
            alpha = torch.cat([x_s, x_t], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(B, 1, C, 2)
            alpha = alpha.softmax(dim=-1)
            x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
        elif self.mode in ['vanilla', 'spatial']:
            x = self.forward_spatial(q, k, v, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
        elif self.mode == 'temporal':
            x = self.forward_temporal(q, k, v, processed_conf=processed_conf, seqlen=seqlen, local_batch=local_batch)
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v, processed_conf=None, seqlen=1, local_batch=None):
        B, H, N, D = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if processed_conf is not None and self.conf_mode == 'attn':
            # processed_conf = rearrange(processed_conf, 'b f n -> (b f) n')
            # Compute conf_weight based on layer position
            if self.total_layers > 1:
                conf_weight = max(0.0, 1.0 - float(self.layer_idx) / (self.total_layers - 1))
            else:
                conf_weight = 1.0

            # conf_flat = processed_conf.view(B, N)  # (B, N)
            # conf_mask = conf_flat.view(B, 1, 1, N)  # (B, 1, 1, N)
            nonlinear_c = torch.sin(processed_conf * torch.pi / 2)
            interp_mask = (1.0 - conf_weight) + conf_weight * nonlinear_c
            # interp_mask = (1.0 - conf_weight) + conf_weight * conf_mask
            attn = attn * interp_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, H * D)
        return x

    def forward_temporal(self, q, k, v, processed_conf=None, seqlen=8, local_batch=None):
        B, H, N, D = q.shape
        qt = q.reshape(local_batch, seqlen, H, N, D).permute(0, 2, 3, 1, 4)
        kt = k.reshape(local_batch, seqlen, H, N, D).permute(0, 2, 3, 1, 4)
        vt = v.reshape(local_batch, seqlen, H, N, D).permute(0, 2, 3, 1, 4)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B_local, H, N, T, T)

        if processed_conf is not None and self.conf_mode == 'attn':


            if self.total_layers > 1:
                conf_weight = max(0.0, 1.0 - float(self.layer_idx) / (self.total_layers - 1))
            else:
                conf_weight = 1.0

            # conf_mask = joint_temporal_conf.unsqueeze(1).unsqueeze(-2)
            nonlinear_c = torch.sin(processed_conf * torch.pi / 2)
            interp_mask = (1.0 - conf_weight) + conf_weight * nonlinear_c
            # interp_mask = (1.0 - conf_weight) + conf_weight * conf_mask

            attn = attn * interp_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, H * D)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st',
                 layer_idx=0, total_layers=1):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                proj_drop=drop, st_mode="spatial", layer_idx=layer_idx, total_layers=total_layers)
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                proj_drop=drop, st_mode="temporal", layer_idx=layer_idx, total_layers=total_layers)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x, seqlen=1, joint_confidence=None,temporal_confidence=None):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen=seqlen, joint_confidence=joint_confidence))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen=seqlen, joint_confidence=temporal_confidence))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self .drop_path(self.attn_t(self.norm1_t(x), seqlen=seqlen, joint_confidence=temporal_confidence))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen=seqlen, joint_confidence=joint_confidence))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=2,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()

        if dim_in != 3:
            raise ValueError("ConfFormer expects (x, y, confidence), so dim_in must be 3")
        mlp_hidden_dim = int(dim_feat * mlp_ratio)
        self.maxlen = maxlen
        self.num_joints = num_joints
        norm_layer = partial(nn.LayerNorm, eps=1e-6) if norm_layer is nn.LayerNorm else norm_layer

        self.joints_embed = nn.Linear(2, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_st",
                layer_idx=i,
                total_layers=depth,
            )
            for i in range(depth)])

        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_ts",
                layer_idx=i,
                total_layers=depth,
            )
            for i in range(depth)])

        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
        for i in range(depth):
            self.ts_attn[i].weight.data.fill_(0)
            self.ts_attn[i].bias.data.fill_(0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b, f, n, c = x.shape  # e.g., (8, 243, 17, 3)

        # Extract confidence: last channel is confidence
        if c != 3 or n != self.num_joints or f > self.maxlen:
            raise ValueError(f"Expected (B, T<= {self.maxlen}, {self.num_joints}, 3), got {tuple(x.shape)}")
        confidence = x[..., -1].clamp(0.0, 1.0)  # (b, f, n)
        x = x[..., :-1]

        temporal_confidence=compute_temporal_confidence(x, sigma=2.0)
        confidence_spatial = rearrange(confidence, 'b f n -> (b f) n')
        confidence_spatial = confidence_spatial.unsqueeze(1)
        confidence_spatial = confidence_spatial.transpose(-2, -1) @ confidence_spatial
        confidence_spatial = confidence_spatial.unsqueeze(1)

        confidence_temporal = rearrange(temporal_confidence, 'b f n -> (b n) f')
        confidence_temporal = confidence_temporal.unsqueeze(1)
        confidence_temporal = confidence_temporal.transpose(-2, -1) @ confidence_temporal
        confidence_temporal = rearrange(confidence_temporal, '(b n) f1 f2 -> b n f1 f2',n=n)
        confidence_temporal = confidence_temporal.unsqueeze(1)

        # Embed input
        x = rearrange(x, 'b f n c -> (b f) n c')
        x = self.joints_embed(x)
        x = x + self.pos_embed

        x = rearrange(x, '(b f) n c -> b f n c', f=f)
        x = x + self.temp_embed[:, :f]
        x = rearrange(x, 'b f n c -> (b f) n c')

        x = self.pos_drop(x)

        for i, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            current_bf = x.shape[0]
            current_f = current_bf // b  # infer current frame length



            # Forward blocks with aligned confidence
            x_st = blk_st(x, current_f, joint_confidence=confidence_spatial,temporal_confidence=None)
            x_ts = blk_ts(x, current_f, joint_confidence=None,temporal_confidence=confidence_temporal)

            att = self.ts_attn[i]
            alpha = torch.cat([x_st, x_ts], dim=-1)
            alpha = att(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]

        x = self.norm(x)

        # Restore batch and frame dimensions.
        x = rearrange(x, '(b f) n c -> (b c) n f', b=b)
        x = rearrange(x, '(b c) n f -> b f n c', b=b)

        x = self.pre_logits(x)
        x = self.head(x)

        return x

    def get_representation(self, x):
        return self.forward(x)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="MotionBERT with ConfFormer attention")
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DSTformer(maxlen=args.frames).to(device).eval()
    inputs = torch.rand(args.batch, args.frames, 17, 3, device=device)
    with torch.no_grad():
        outputs = model(inputs)
    print(f"{tuple(inputs.shape)} -> {tuple(outputs.shape)} on {device}")
