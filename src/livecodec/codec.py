import torch
import torch.nn as nn
import einops
import numpy as np
from PIL import Image
from timm.models.efficientvit_mit import GELUTanh
from torchvision.transforms.v2.functional import pil_to_tensor, to_pil_image
from torch.distributions.laplace import Laplace
from einops import rearrange
from einops.layers.torch import Rearrange
from .evit import EfficientVitLargeStageND, GroupNorm8
from .monarch import FactorizedConvND, FactorizedResBlockGNND

def compand(x, eps=0.1, power=0.4):
    return x.sign() * ((x.abs()+eps)**power - eps**power)
def decompand(y, eps=0.1, power=0.4):
    return y.sign() * ((y.abs()+eps**power)**(1/power) - eps)

class LaplaceCompand(nn.Module):
    def __init__(self, num_channels):
        super(LaplaceCompand, self).__init__()
        self.sigma = nn.Parameter(42.0*torch.ones(num_channels))

    def forward(self, x):
        shape = [1, -1] + [1] * (x.dim() - 2)
        sigma = self.sigma.view(shape).clamp(min=1e-6)
        laplace = Laplace(loc=0., scale=sigma)
        cdf = laplace.cdf(x)
        out = 254 * cdf - 127
        return out

class QuantizeLF8(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.compand = LaplaceCompand(num_channels)

    def forward(self, x):
        x = self.compand(x)
        if self.training:
            x += torch.rand_like(x) - 0.5
        return x

class AutoCodecND(nn.Module):
    def __init__(
        self,
        dim=2,
        input_channels=3,
        J=4,
        latent_dim=12,
        encoder_depth=6,
        encoder_kernel_size=3,
        decoder_depth=6,
        decoder_kernel_size=3,
        lightweight_encode=True,
        lightweight_decode=False,
    ):
        super().__init__()
        assert dim in (1, 2, 3), "Dimension should be 1, 2 or 3."
        self.hidden_dim = input_channels * (2 ** (dim * J))
        self.dim = dim
        self.J = J
        self.latent_dim = latent_dim
        self.lightweight_encode = lightweight_encode
        self.lightweight_decode = lightweight_decode

        if dim == 1:
            from tft.wavelet import WPT1D, IWPT1D, DWT1DForward, DWT1DInverse
            self.wt = DWT1DForward(J=1, wave='bior4.4')
            self.wpt = WPT1D(wt=self.wt, J=self.J)
            self.iwt = DWT1DInverse(wave='bior4.4')
            self.iwpt = IWPT1D(iwt=self.iwt, J=self.J)
            conv_layer = torch.nn.Conv1d
        elif dim == 2:
            from tft.wavelet import WPT2D, IWPT2D, DWT2DForward, DWT2DInverse
            self.wt = DWT2DForward(J=1, wave='bior4.4')
            self.wpt = WPT2D(wt=self.wt, J=self.J)
            self.iwt = DWT2DInverse(wave='bior4.4')
            self.iwpt = IWPT2D(iwt=self.iwt, J=self.J)
            conv_layer = torch.nn.Conv2d
        elif dim == 3:
            from tft.wavelet import WPT3D, IWPT3D, DWT3DForward, DWT3DInverse
            self.wt = DWT3DForward(J=1, wave='bior4.4')
            self.wpt = WPT3D(wt=self.wt, J=self.J)
            self.iwt = DWT3DInverse(wave='bior4.4')
            self.iwpt = IWPT3D(iwt=self.iwt, J=self.J)
            conv_layer = torch.nn.Conv3d

        if lightweight_encode:
            self.encoder_blocks = nn.Sequential(
                *[FactorizedResBlockGNND(dim, self.hidden_dim, kernel_size=encoder_kernel_size) for _ in range(encoder_depth)]
            )
        else:
            self.encoder_blocks = EfficientVitLargeStageND(
                dim=dim,
                in_chs=self.hidden_dim,
                depth=encoder_depth,
                norm_layer=GroupNorm8,
                act_layer=GELUTanh
            )

        self.conv_down = FactorizedConvND(dim, self.hidden_dim, latent_dim, kernel_size=1, bias=False)
        self.quantize = QuantizeLF8(latent_dim)
        self.conv_up = FactorizedConvND(dim, latent_dim, self.hidden_dim, kernel_size=1, bias=False)

        if lightweight_decode:
            self.decoder_blocks = nn.Sequential(
                *[FactorizedResBlockGNND(dim, self.hidden_dim, kernel_size=decoder_kernel_size) for _ in range(decoder_depth)]
            )
        else:
            self.decoder_blocks = EfficientVitLargeStageND(
                dim=dim,
                in_chs=self.hidden_dim,
                depth=decoder_depth,
                norm_layer=GroupNorm8,
                act_layer=GELUTanh
            )

    def encode(self, x):
        x = self.wpt(x)
        x = 12.8 * compand(x, power=0.4)
        x = self.encoder_blocks(x)
        x = self.conv_down(x)
        return x

    def decode(self, x):
        x = self.conv_up(x)
        x = self.decoder_blocks(x)
        x = decompand(x / 12.8, power=0.4)
        x = self.iwpt(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        rate = self.quantize.compand(x).std().log2()
        x = self.quantize(x)
        x = self.decode(x)
        return x, rate
        
def to_bytes(x, n_bits):
    max_value = 2**(n_bits - 1) - 1
    min_value = -max_value - 1
    if x.min() < min_value or x.max() > max_value:
        raise ValueError(f"Tensor values should be in the range [{min_value}, {max_value}].")
    return (x + (max_value + 1)).to(torch.uint8)

def from_bytes(x, n_bits):
    max_value = 2**(n_bits - 1) - 1
    return (x.to(torch.float32) - (max_value + 1))

def concatenate_channels(x, C):
    batch_size, N, h, w = x.shape
    if N % C != 0 or int((N // C)**0.5) ** 2 * C != N:
        raise ValueError(f"Number of channels must satisfy N = {C} * (n^2) for some integer n.")
    
    n = int((N // C)**0.5)
    x = rearrange(x, 'b (c nh nw) h w -> b (nh h) (nw w) c', c=C, nh=n, nw=n)
    return x

def split_channels(x, N, C):
    batch_size, _, H, W = x.shape
    n = int((N // C)**0.5)
    h = H // n
    w = W // n
    
    x = rearrange(x, 'b c (nh h) (nw w) -> b (c nh nw) h w', c=C, nh=n, nw=n)
    return x

def latent_to_pil(latent, n_bits, C):
    latent_bytes = to_bytes(latent, n_bits)
    concatenated_latent = concatenate_channels(latent_bytes, C)
    
    if C == 1:
        mode = 'L'
        concatenated_latent = concatenated_latent.squeeze(-1)
    elif C == 3:
        mode = 'RGB'
    elif C == 4:
        mode = 'CMYK'
    else:
        raise ValueError(
            f"Unsupported number of channels C={C}. Supported values are 1 (L), 3 (RGB), and 4 (CMYK)"
        )
    
    pil_images = []
    for i in range(concatenated_latent.shape[0]):
        pil_image = Image.fromarray(concatenated_latent[i].numpy(), mode=mode)
        pil_images.append(pil_image)
    
    return pil_images

def pil_to_latent(pil_images, N, n_bits, C):
    tensor_images = [pil_to_tensor(img).unsqueeze(0) for img in pil_images]
    tensor_images = torch.cat(tensor_images, dim=0)
    split_latent = split_channels(tensor_images, N, C)
    latent = from_bytes(split_latent, n_bits)
    return latent