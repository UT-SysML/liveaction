import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm8(nn.Module):
    def __init__(self, num_features, eps=1e-7, affine=True):
        super().__init__()
        self.groupnorm = torch.nn.GroupNorm(num_groups=8,
                                      num_channels=num_features,
                                      eps=eps,
                                      affine=affine)
    def forward(self, x):
        return self.groupnorm(x)

class ConvNormActND(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size=3, stride=1, groups=1,
                 bias=False, norm_layer=None, act_layer=nn.GELU):
        super().__init__()
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        Norm = norm_layer or [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim-1]
        padding = kernel_size // 2
        self.conv = Conv(in_channels, out_channels, kernel_size, stride,
                         padding=padding, groups=groups, bias=bias)
        self.norm = Norm(out_channels)
        self.act = act_layer() if act_layer else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class MBConvND(nn.Module):
    def __init__(self, dim, in_channels, expand_ratio=6, norm_layer=None, act_layer=nn.GELU):
        super().__init__()
        mid_channels = in_channels * expand_ratio
        self.inverted_conv = ConvNormActND(dim, in_channels, mid_channels, 1, norm_layer=norm_layer, act_layer=act_layer, bias=True)
        self.depth_conv = ConvNormActND(dim, mid_channels, mid_channels, 3, groups=mid_channels, norm_layer=norm_layer, act_layer=act_layer, bias=True)
        self.point_conv = ConvNormActND(dim, mid_channels, in_channels, 1, norm_layer=norm_layer, act_layer=act_layer, bias=False)

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class ResidualBlockND(nn.Module):
    def __init__(self, main):
        super().__init__()
        self.main = main

    def forward(self, x):
        return x + self.main(x)

class LiteMLAND(nn.Module):
    def __init__(self, dim, in_channels, head_dim=32, norm_layer=None, kernel_func=nn.ReLU, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        heads = in_channels // head_dim
        total_dim = heads * head_dim
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        self.qkv = Conv(in_channels, 3 * total_dim, 1, bias=False)
        self.kernel_func = kernel_func()
        self.proj = ConvNormActND(dim, total_dim, in_channels, 1, norm_layer=norm_layer, act_layer=None)

    def forward(self, x):
        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]
        N = torch.prod(torch.tensor(spatial_dims))
        qkv = self.qkv(x).reshape(B, -1, 3, N).permute(0, 3, 1, 2)
        q, k, v = qkv.unbind(-1)

        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 1), mode='constant', value=1.)

        kv = torch.einsum('bnc,bnv->bcv', k, v)
        out = torch.einsum('bnc,bcv->bnv', q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = out.permute(0, 2, 1).reshape(B, -1, *spatial_dims)
        return self.proj(out)

class EfficientVitBlockND(nn.Module):
    def __init__(self, dim, in_channels, norm_layer=None, act_layer=nn.GELU, head_dim=32):
        super().__init__()
        self.context_module = ResidualBlockND(LiteMLAND(dim, in_channels, norm_layer=norm_layer, head_dim=head_dim))
        self.local_module = ResidualBlockND(MBConvND(dim, in_channels, norm_layer=norm_layer, act_layer=act_layer))

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x

class EfficientVitLargeStageND(nn.Module):
    def __init__(self, dim, in_chs, depth, norm_layer=None, act_layer=nn.GELU, head_dim=32):
        super().__init__()
        self.blocks = nn.Sequential(*[
            EfficientVitBlockND(dim, in_chs, norm_layer=norm_layer, act_layer=act_layer, head_dim=32)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(x)
