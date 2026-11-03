"""KTPFormer backbone with ConfFormer reliability-conditioned STE/TTE attention.

Input: (batch, frames, 17, 3), containing preprocessed x, y and confidence.
The KPA/TPA topology front end and model parameter names follow the supplied
KTPFormer implementation. Confidence modulates each subsequent STE/TTE block.
"""
import math
from functools import partial

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 changedim=False, currentdim=0, depth=0):
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


class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        # spatial/temporal local topology
        self.adj = adj

        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)

        adj = (adj.T + adj) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class KPA(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(KPA, self).__init__()

        self.gconv = LearnableGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class TPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, p_dropout=None):
        super(TPA, self).__init__()

        self.gconv = LearnableGraphConv(input_dim, output_dim, adj_temporal)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class StackedTPA(nn.Module):
    def __init__(self, adj_temporal, input_dim, output_dim, hid_dim, p_dropout):
        super(StackedTPA, self).__init__()

        self.gconv1 = TPA(adj_temporal, input_dim, hid_dim, p_dropout)
        self.gconv2 = TPA(adj_temporal, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False,
                 vis=False, layer_idx=0,
                 total_layers=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis
        self.layer_idx = layer_idx
        self.total_layers = total_layers

    def forward(self, x,confidence,vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
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

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KPAttention(nn.Module):
    def __init__(self, adj, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.kpa = KPA(adj, 2, dim, p_dropout=None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, 17, dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, vis=False):

        x = self.kpa(x)

        x = x+self.Spatial_pos_embed

        x = self.pos_drop(x)

        res = x.clone()

        x = self.norm1(x)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = res + self.drop_path(x)

        return x


class TPAttention(nn.Module):
    def __init__(self, adj_temporal, num_frame, dim, num_heads=8, drop_path=0, drop_rate=0, norm_layer=nn.LayerNorm,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.tpa = StackedTPA(adj_temporal, dim, dim, dim, p_dropout=None)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, vis=False):

        x = self.tpa(x)

        x = x+self.Temporal_pos_embed

        x = self.pos_drop(x)

        res = x.clone()

        x = self.norm1(x)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Now x shape (3, B, heads, N, C//heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        if self.comb == True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb == False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.comb == True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            # print(x.shape)
            x = rearrange(x, 'B H N C -> B N (H C)')
            # print(x.shape)
        elif self.comb == False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = res + self.drop_path(x)

        return x


class KPA_Block(nn.Module):

    def __init__(self, adj, dim, num_heads, mlp_ratio=4., attention=KPAttention, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        # self.norm1 = norm_layer(dim)
        self.attn = attention(
            adj, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop, norm_layer=nn.LayerNorm,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        # x = x + self.drop_path(self.attn(x, vis=vis))

        x = self.attn(x, vis=vis)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x


class TPA_Block(nn.Module):

    def __init__(self, adj_temporal, num_frame, dim, num_heads, mlp_ratio=4., attention=TPAttention, qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        # self.norm1 = norm_layer(dim)
        self.attn = attention(
            adj_temporal, num_frame, dim, num_heads=num_heads, drop_path=drop_path, drop_rate=drop,
            norm_layer=nn.LayerNorm, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        # x = x + self.drop_path(self.attn(x, vis=vis))
        x = self.attn(x, vis=vis)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False, layer_idx=0, total_layers=1):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            comb=comb, vis=vis, layer_idx=layer_idx,
                              total_layers=total_layers)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, confidence,vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x),confidence, vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
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
class KTPFormer(nn.Module):
    def __init__(self, adj, adj_temporal, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3  #### output dimension is num_joints * 3

        self.pos_drop = nn.Dropout(p=drop_rate)

        num_drop_path_rate = depth + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_drop_path_rate)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + 1], norm_layer=norm_layer ,layer_idx=i,
                total_layers=depth)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + 1], norm_layer=norm_layer, comb=False,
                changedim=False, currentdim=i + 1, depth=depth,layer_idx=i,
                total_layers=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.kpattention = KPA_Block(
            adj, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, attention=KPAttention,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)

        self.tpattention = TPA_Block(
            adj_temporal, num_frame, dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio,
            attention=TPAttention, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def KPA_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )

        x = self.kpattention(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        return x

    def TPA_foward(self, x):
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _ = x.shape

        x = self.tpattention(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x,confidence_spatial,confidence_temporal):
        assert len(x.shape) == 4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]

            # x += self.Spatial_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = steblock(x, vis=True)
            x = steblock(x,confidence_spatial)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            # x += self.Temporal_pos_embed
            # x = self.pos_drop(x)
            # if i==7:
            #     x = tteblock(x, vis=True)
            #     exit()
            x = tteblock(x,confidence_temporal)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

        # x = rearrange(x, 'b f n cw -> (b n) f cw', n=n)
        # x = self.weighted_mean(x)
        # x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # x = x.view(b, f, -1)
        return x

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("Expected input shape (batch, frames, joints, 3): (x, y, confidence)")
        b, f, n, c = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x shape:(b f n c)
        # torch.cuda.synchronize()
        # st = time.time()
        confidence = x[..., 2].clamp(0.0, 1.0)  # (b, f, n)
        x = x[..., :2]
        temporal_confidence = compute_temporal_confidence(x, sigma=2.0, order=2)

        confidence_spatial = rearrange(confidence, 'b f n -> (b f) n')
        confidence_spatial = confidence_spatial.unsqueeze(1)
        confidence_spatial = confidence_spatial.transpose(-2, -1) @ confidence_spatial
        confidence_spatial = confidence_spatial.unsqueeze(1)

        confidence_temporal = rearrange(temporal_confidence, 'b f n -> (b n) f')
        confidence_temporal = confidence_temporal.unsqueeze(1)
        confidence_temporal = confidence_temporal.transpose(-2, -1) @ confidence_temporal
        confidence_temporal = confidence_temporal.unsqueeze(1)

        x = self.KPA_forward(x)
        # now x shape is (b n) f cw
        # et = time.time()
        # print('STE_forward  ', (et-st)*2000)

        # st = time.time()
        x = self.TPA_foward(x)
        # et = time.time()
        # print('TTE_foward  ', (et-st)*2000)

        # now x shape is (b n) f cw
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        # st = time.time()
        x = self.ST_foward(x,confidence_spatial,confidence_temporal)
        # et = time.time()
        # print('ST_foward  ', (et-st)*2000)

        # st = time.time()
        x = self.head(x)
        # et = time.time()
        # print('head  ', (et-st)*2000)
        # now x shape is (b f (n * 3))

        x = x.view(b, f, n, -1)

        return x
