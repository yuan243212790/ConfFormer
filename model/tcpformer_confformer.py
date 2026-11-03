"""TCPFormer backbone with ConfFormer SCA/TMS self-attention.

Official source: https://github.com/AsukaCamellia/TCPFormer
This self-contained module retains the official proxy cross-attention and learned
fusion; observed-joint reliability modulates spatial/temporal self-attention.
Input: (B,T,J,3) containing normalized x, y and detector confidence.
"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial', vis='no', layer_idx=0,
                 total_layers=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.vis = vis
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layer_idx = layer_idx
        self.total_layers = total_layers
    def forward(self, x,confidence=None):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v,confidence)
        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v,confidence)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v,confidence):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        if self.total_layers > 1:
            conf_weight = max(0.0, 1.0 - float(self.layer_idx) / (self.total_layers - 1))
        else:
            conf_weight = 1.0
        if confidence is None:
            interp_mask = 1.0
        else:
            nonlinear_c = torch.sin(confidence * torch.pi / 2)
            interp_mask = (1.0 - conf_weight) + conf_weight * nonlinear_c
        attn = attn * interp_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v,confidence):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        if self.total_layers > 1:
            conf_weight = max(0.0, 1.0 - float(self.layer_idx) / (self.total_layers - 1))
        else:
            conf_weight = 1.0
        if confidence is None:
            interp_mask = 1.0
        else:
            nonlinear_c = torch.sin(confidence * torch.pi / 2)
            interp_mask = (1.0 - conf_weight) + conf_weight * nonlinear_c
        attn = attn * interp_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)

class CrossAttention(nn.Module):

    def __init__(self,dim_in,dim_out,num_heads = 8,qkv_bias = False,qkv_scale = None,attn_drop=0.,proj_drop=0.,
                 mode = 'temporal',back_att = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qkv_scale or head_dim**(-0.5)
        self.wq = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.wk = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.wv = nn.Linear(dim_in,dim_in,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in,dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.back_att = back_att

    def forward(self,q,kv):
        #batch_size temporal_frame_num num_joints feature_dim
        b , t , j , d = q.shape
        t_sup = kv.shape[1]
        q = self.wq(q).reshape(b,t,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)
        k = self.wk(kv).reshape(b,t_sup,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)
        v = self.wv(kv).reshape(b,t_sup,j,self.num_heads,d//self.num_heads).permute(0,3,2,1,4)

        attn = (q @ k.transpose(-2,-1))*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn@v  # b h j t c
        out = out.permute(0,3,2,1,4).reshape(b,t,j,d)
        out = self.proj(out)
        out = self.proj_drop(out)
        if self.back_att:
            return attn,out
        else:
            return out



# x = torch.ones((16,9,17,256))
# kv = torch.ones((16,18,17,256))

# net = CrossAttention(dim_in=256,dim_out=256,head_num=8)

# out = net(x,kv)


class Sum_Attention(nn.Module):


    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial', layer_idx=0, total_layers=1):
        super().__init__()
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,att_map,weight,confidence=None):
        B, T, J, C = x.shape

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)

        q, k, v = qkv[0], qkv[1], qkv[2]
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        if confidence is not None:
            strength = max(0.0, 1.0 - self.layer_idx / (self.total_layers - 1)) if self.total_layers > 1 else 1.0
            attn = attn * ((1.0 - strength) + strength * torch.sin(confidence * torch.pi / 2))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)


        attn = weight*attn + (1-weight)*att_map

        attn = self.attn_drop(attn)
        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x









class MIBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='temporal', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243,is_local = None, layer_idx=0, total_layers=1):
        super().__init__()
        self.norm_full = nn.LayerNorm(dim)
        self.norm_center = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.full_center = CrossAttention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,back_att=True)
        self.center_full = CrossAttention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,back_att=True)
        self.mlp_1 = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.mlp_2 = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_4 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_5 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_6 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_7 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_8 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)


        self.norm_sa_self = nn.LayerNorm(dim)
        self.map_sa_self = Attention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,layer_idx=layer_idx,total_layers=total_layers)
        self.norm_mlp_self = nn.LayerNorm(dim)
        self.mlp_sa_self = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

        self.norm_sa_1 = nn.LayerNorm(dim)
        self.map_sum = Sum_Attention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,layer_idx=layer_idx,total_layers=total_layers)
        self.norm_sa_2 = nn.LayerNorm(dim)
        self.mlp_sa = MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

        
        self.sg = nn.Sigmoid()
        self.att_weight = nn.Parameter(torch.rand(1))

    def forward(self, x,pose_query,confidence=None):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:

            attn_map_1,out_1 = self.center_full(self.norm_center(pose_query),self.norm_full(x))
            pose_query = pose_query + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0)*out_1)
            pose_query = pose_query + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0)*self.mlp_1(self.norm_1(pose_query)))

            attn_map_2,out_2 = self.full_center(self.norm_full(x),self.norm_center(pose_query))
            x = x + self.drop_path(self.layer_scale_3.unsqueeze(0).unsqueeze(0)*out_2)
            x = x + self.drop_path(self.layer_scale_4.unsqueeze(0).unsqueeze(0)*self.mlp_2(self.norm_2(x)))

            attn_map = attn_map_2 @ attn_map_1

            norm_weight = self.sg(self.att_weight)

            x = x + self.drop_path(self.layer_scale_7.unsqueeze(0).unsqueeze(1)*self.map_sa_self(self.norm_sa_self(x),confidence))
            x = x + self.drop_path(self.layer_scale_8.unsqueeze(0).unsqueeze(1)*self.mlp_sa_self(self.norm_mlp_self(x)))

            x = x + self.drop_path(self.layer_scale_5.unsqueeze(0).unsqueeze(1)*self.map_sum(self.norm_sa_1(x),attn_map,norm_weight,confidence))
            x = x + self.drop_path(self.layer_scale_6.unsqueeze(0).unsqueeze(1)*self.mlp_sa(self.norm_sa_2(x)))

        return x,pose_query
    






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



class TransBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243, layer_idx=0, total_layers=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer_type = mixer_type
        if mixer_type != 'attention':
            raise ValueError("This backbone uses the official self-attention TransBlock path")
        self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                               proj_drop=drop, mode=mode, layer_idx=layer_idx, total_layers=total_layers)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)


        


    def forward(self, x,confidence=None):
        """
        x: tensor with shape [B, T, J, C]
        """

        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x),confidence))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x),confidence))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class DSTFormerBlock(nn.Module):


    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243, layer_idx=0, total_layers=1):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim


        self.att_spatial = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers)
        self.att_temporal = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers)



        self.graph_spatial = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='temporal', mixer_type="attention",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers)
        self.graph_temporal = TransBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='spatial', mixer_type='attention',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x,confidence=None):
        """
        x: tensor with shape [B, T, J, C]
        """

        x_attn = self.att_temporal(self.att_spatial(x,confidence[0]),confidence[1])
        x_graph = self.graph_temporal(self.graph_spatial(x,confidence[1]),confidence[0])


        alpha = torch.cat((x_attn, x_graph), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]

        return x


class MemoryInducedBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,mode='temporal', layer_idx=0, total_layers=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.local_attention_list = nn.ModuleList([
                Attention(dim,dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop=drop,mode=mode,layer_idx=layer_idx,total_layers=total_layers) for i in range(3)
            ])
        self.loacl_mlps =  nn.ModuleList([
                MLP(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop) for i in range(3)
            ])
        self.layer_scale = nn.ParameterList([
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True),
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        ])
        self.local_norms = nn.ModuleList([
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim)
        ])

        self.cross_temporal = MIBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers)      

    def forward(self, x,pose_query,confidence=None):
        """
        x: tensor with shape [B, T, J, C]
        """
        lengths = [chunk.shape[1] for chunk in torch.chunk(x,3,dim=1)]
        starts = [0, lengths[0], lengths[0] + lengths[1]]
        local_confidence = [confidence[..., start:start+length, start:start+length] for start,length in zip(starts,lengths)]
        x = list(torch.chunk(x,3,dim=1))

        for i in range(3):
            x[i] = x[i] + self.drop_path(self.layer_scale[i].unsqueeze(0).unsqueeze(0) * self.local_attention_list[i](self.local_norms[i](x[i]), local_confidence[i]))
            x[i] = x[i] + self.drop_path(self.layer_scale[i+3].unsqueeze(0).unsqueeze(0) * self.loacl_mlps[i](self.local_norms[i+3](x[i])))

        x = torch.cat(x,dim=1)

        x,pose_query = self.cross_temporal(x,pose_query,confidence)

        return x,pose_query









def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243,type = None):

    layers = []
    for layer_idx in range(n_layers):
        total_layers = n_layers
        if type == 'temporal':
            layers.append(MemoryInducedBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers))
        else:
            layers.append(DSTFormerBlock(dim=dim,
                                          mlp_ratio=mlp_ratio,
                                          act_layer=act_layer,
                                          attn_drop=attn_drop,
                                          drop=drop_rate,
                                          drop_path=drop_path_rate,
                                          num_heads=num_heads,
                                          use_layer_scale=use_layer_scale,
                                          layer_scale_init_value=layer_scale_init_value,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qkv_scale,
                                          use_adaptive_fusion=use_adaptive_fusion,
                                          hierarchical=hierarchical,
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len,
                                          use_tcn=use_tcn,
                                          graph_only=graph_only,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,layer_idx=layer_idx,total_layers=total_layers))
    layers = nn.Sequential(*layers)

    return layers


class MemoryInducedTransformer(nn.Module):


    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False, graph_only=False,
                 neighbour_num=4, n_frames=243, layer_idx=0, total_layers=1):

        super().__init__()

        self.n_frames = n_frames
        if dim_in != 2:
            raise ValueError("dim_in must be 2: confidence is used for modulation, not coordinate embedding")
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.norm = nn.LayerNorm(dim_feat)
        self.layers_num = n_layers
        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

        self.temporal_layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    use_temporal_similarity=use_temporal_similarity,
                                    temporal_connection_len=temporal_connection_len,
                                    use_tcn=use_tcn,
                                    graph_only=graph_only,
                                    neighbour_num=neighbour_num,
                                    n_frames=n_frames,
                                    type='temporal')
        

        self.center_pose = nn.Parameter(torch.randn(int(n_frames/3),num_joints,dim_feat))
        self.center_pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        """
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError("Expected input (B,T,J,3): x, y, confidence")
        b,t,j,c = x.shape
        if t != self.n_frames or t < 3:
            raise ValueError(f"Expected {self.n_frames} frames (at least 3), got {t}")
        scores = x[..., 2].clamp(0.0, 1.0)
        x = x[..., :2]
        temporal = compute_temporal_confidence(x, sigma=2.0, order=2)
        spatial_affinity = (scores.unsqueeze(-1) * scores.unsqueeze(-2)).unsqueeze(1)
        temporal = temporal.transpose(1, 2)
        temporal_affinity = (temporal.unsqueeze(-1) * temporal.unsqueeze(-2)).unsqueeze(1)
        pose_query = self.center_pose.unsqueeze(0).repeat(b,1,1,1)
        pose_query = pose_query + self.center_pos_embed
        x = self.joints_embed(x)  #
        x = x + self.pos_embed

        for layer,temporal_layer in zip(self.layers,self.temporal_layers):
            x = layer(x,(spatial_affinity,temporal_affinity))
            x,pose_query = temporal_layer(x,pose_query,temporal_affinity)


        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x





class Model(MemoryInducedTransformer):
    """Adapter for this project's Model(args) training entry point."""
    def __init__(self, args):
        super().__init__(n_layers=getattr(args, 'depth', 16), dim_in=2,
                         dim_feat=getattr(args, 'dim_feat', 128),
                         dim_rep=getattr(args, 'dim_rep', 512), num_heads=8,
                         num_joints=args.n_joints, n_frames=args.frames)
