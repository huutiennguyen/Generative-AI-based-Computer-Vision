

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Creates sinusoidal positional embeddings for a given 1D tensor of time steps.
    This module is a standard way to encode positions or time into a format that
    a neural network can easily process.

    Args:
        dim (int): The dimension of the embedding vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        log_freq_base = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -log_freq_base)
        time_freqs = time[:, None] * inv_freq[None, :]
        embeddings = torch.cat((time_freqs.sin(), time_freqs.cos()), dim=-1)

        return embeddings

class MultiScaleInputBlock(nn.Module):
    """
    Processes an input tensor using multiple parallel convolutional paths
    with different kernel sizes, then concatenates the results. This allows
    the model to capture features at various receptive fields simultaneously.

    Args:
        - in_channels (int): Number of channels in the input image.
        - out_channels (int): Number of channels in the output feature map.
        - kernel_sizes (list[int]): A list of kernel sizes for parallel convolutional paths.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes): # kernel_sizes=[1,3,7,15]
        super().__init__()
        if not kernel_sizes or out_channels % len(kernel_sizes) != 0:
            raise ValueError("out_channels must be divisible by the number of kernel sizes.")

        self.out_channels_per_path = out_channels // len(kernel_sizes)

        self.convs = nn.ModuleList()
        for k_size in kernel_sizes:
            padding = k_size // 2
            self.convs.append(
                nn.Conv2d(in_channels, self.out_channels_per_path, kernel_size=k_size, padding=padding)
            )

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)

class SEBlock(nn.Module):
    """
    Implements a Squeeze-and-Excitation block for channel-wise feature recalibration.
    This block learns to explicitly model interdependencies between channels, allowing
    it to adaptively re-weight feature maps to emphasize informative features and
    suppress less useful ones.

    Args:
        - channels (int): The number of channels in the input feature map.
        - reduction_ratio (int): The reduction ratio for the bottleneck in the
                               excitation network.
    """
    def __init__(self, channels: int, reduction_ratio: int):
        super().__init__()
        squeezed_channels = max(1, channels // reduction_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, squeezed_channels, bias=False),
            nn.SiLU(),
            nn.Linear(squeezed_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Resnet(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int, *, time_emb_dim: int = None, groups: int = 8,
                 use_se: bool = True, se_reduction_ratio: int = 8 # use squeeze excite
                 ):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.SiLU(),
                          nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels, eps=1e-05)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-05)
        self.act2 = nn.SiLU()

        self.se = SEBlock(out_channels, reduction_ratio=se_reduction_ratio) if use_se else nn.Identity()
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None):
        identity = self.residual_connection(x)
        h = self.act1(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and time_emb is not None:
            time_additive = self.time_mlp(time_emb)
            h = rearrange(time_additive, "b c -> b c 1 1") + h
        h = self.act2(self.norm2(self.conv2(h)))
        h = self.se(h)

        return h + identity

class Attention(nn.Module):
    """
    Multi-Head Self-Attention mechanism allows the model to 
    focus on different regions of the image and capture global relationships.
    """
    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        similarity = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        attn = similarity.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    """
    A complete Transformer block used in the bottleneck of the U-Net. 
    It consists of an Attention layer and a Feed-Forward Network (MLP).
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, mlp_mult: int = 2):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.attn = Attention(dim, heads, dim_head)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_mult, dim, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.attn(self.norm(x)) + x
        x = self.ff(self.norm(x)) + x
        return x

def Downsample(dim_in, dim_out=None):
    """
    Downsamples by rearranging pixels from space to depth (Pixel Unshuffle),
    followed by a 1x1 convolution to adjust channels.
    """
    if dim_out is None:
        dim_out = dim_in

    return nn.Sequential(
        # Takes (B, C, H*2, W*2) -> (B, C*4, H, W)
        Rearrange('b c (h s) (w t) -> b (c s t) h w', s=2, t=2),
        # Adjusts the channel dimension from C*4 to the desired dim_out
        nn.Conv2d(dim_in * 4, dim_out, 1)
    )

class Upsample(nn.Module): # PixelShuffleUpsample
    """
    Upsamples by first increasing channels with a standard convolution
    and then rearranging pixels from the channel dimension to space.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The first convolution increases the number of channels by a factor of 4 (2*2)
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        # The Rearrange layer performs the actual pixel shuffle
        self.shuffle = Rearrange('b (c s t) h w -> b c (h s) (w t)', s=2, t=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x
    
class CFMUNet(nn.Module):
    def __init__(self,
                 dim: int = 64,                   # base number of feature channels in the model.
                 dim_mults: tuple = (1, 2, 4),    # multipliers for the number of channels at each encoder/decoder level.
                 channels: int = 1,               # number of input channels (e.g., 1 for grayscale MRI slices).
                 time_dim_mult: int = 4,          # multiplier to determine time embedding dimension, i.e., time_dim = dim * time_dim_mult.
                 transformer_heads: int = 8,      # number of attention heads in each Transformer block.
                 transformer_dim_head: int = 32,  # dimensionality of each attention head in the Transformer.
                 kernel_sizes: list = [1,3,7,15]
                 ):

        super().__init__()
        self.channels = channels

        init_dim = dim                       # 64
        dim_stage_1 = dim * dim_mults[0]     # 64 * 1 = 64
        dim_stage_2 = dim * dim_mults[1]     # 64 * 2 = 128
        dim_stage_3 = dim * dim_mults[2]     # 64 * 4 = 256
        bottleneck_dim = dim_stage_3 * 2     # 512

        # input (use MultiScaleInputBlock) & time embedding
        self.init_conv = MultiScaleInputBlock(
            in_channels=channels * 2,
            out_channels=init_dim,
            kernel_sizes=kernel_sizes
        )

        time_dim = dim * time_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(dim),
            nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )

        # encoder
        self.down_block_1 = Resnet(init_dim, dim_stage_1, time_emb_dim=time_dim)    # shape: 256x256 -> 256x256 | dim: 64 -> 64
        self.down_sample_1 = Downsample(dim_stage_1)                                # shape: 256x256 -> 128x128 | dim: 64 -> 64

        self.down_block_2 = Resnet(dim_stage_1, dim_stage_2, time_emb_dim=time_dim) # shape: 128x128 -> 128x128 | dim: 64 -> 128
        self.down_sample_2 = Downsample(dim_stage_2)                                # shape: 128x128 -> 64x64   | dim: 128 -> 128

        self.down_block_3 = Resnet(dim_stage_2, dim_stage_3, time_emb_dim=time_dim) # shape: 64x64 -> 64x64     | dim: 128 -> 256
        self.down_sample_3 = Downsample(dim_stage_3)                                # shape: 64x64 -> 32x32     | dim: 256 -> 256

        # bottleneck
        self.mid_transformer = TransformerBlock(
            dim_stage_3, heads=transformer_heads, dim_head=transformer_dim_head
        )

        # decoder
        self.up_sample_1 = Upsample(dim_stage_3, dim_stage_3)                                     # shape: 32x32 -> 64x64     | dim: 512 -> 256
        self.up_block_1 = Resnet(dim_stage_3 + dim_stage_3, dim_stage_3, time_emb_dim=time_dim)   # shape: 64x64              | dim: 512 -> 256 (concat Re)

        self.up_sample_2 = Upsample(dim_stage_3, dim_stage_2)                                     # shape: 64x64 -> 128x128   | dim: 256 -> 128
        self.up_block_2 = Resnet(dim_stage_2 + dim_stage_2, dim_stage_2, time_emb_dim=time_dim)   # shape: 128x128            | dim: 256 -> 128 (concat Re)

        self.up_sample_3 = Upsample(dim_stage_2, dim_stage_1)                                     # shape: 128x128 -> 256x256 | dim: 128 -> 64
        self.up_block_3 = Resnet(dim_stage_1 + dim_stage_1, init_dim, time_emb_dim=time_dim)      # shape: 256x256            | dim: 128 -> 64 (concat Re)

        # output
        self.final_conv = nn.Conv2d(init_dim, channels, 1)

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, x_cond: torch.Tensor):
        x = torch.cat((x_t, x_cond), dim=1)
        x = self.init_conv(x)
        t_emb = self.time_mlp(time)

        # encoder
        h1 = self.down_block_1(x, t_emb)
        x = self.down_sample_1(h1)

        h2 = self.down_block_2(x, t_emb)
        x = self.down_sample_2(h2)

        h3 = self.down_block_3(x, t_emb)
        x = self.down_sample_3(h3)

        # bottleneck
        x = self.mid_transformer(x)

        # decoder
        x = self.up_sample_1(x)
        x = torch.cat([x, h3], dim=1)
        x = self.up_block_1(x, t_emb)

        x = self.up_sample_2(x)
        x = torch.cat([x, h2], dim=1)
        x = self.up_block_2(x, t_emb)

        x = self.up_sample_3(x)
        x = torch.cat([x, h1], dim=1)
        x = self.up_block_3(x, t_emb)

        # output
        return self.final_conv(x)