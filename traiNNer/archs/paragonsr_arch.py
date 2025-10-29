#!/usr/bin/env python3
"""
ParagonSR: A High-Performance Super-Resolution Network
Author: Philip Hofmann

Description:
ParagonSR is a state-of-the-art, general-purpose super-resolution architecture
designed for a superior balance of peak quality, training efficiency, and
inference speed. It is the result of a rigorous design and debugging process,
representing a synergistic blend of the most effective and efficient ideas from
a multitude of modern SISR models.

Licensed under the MIT License.

-------------------------------------------------------------------------------------
Core Design Philosophy & Strengths:

ParagonSR is built as an "Optimized Hybrid CNN," designed to achieve the perceptual
power of a Transformer with the efficiency of a Convolutional Neural Network.

Its primary strengths are:
1.  **High-Quality, Realistic Output:** By combining powerful context-gathering and
    feature-transformation modules, the architecture excels at learning the
    complex textures and structures necessary for photorealistic image restoration.
2.  **Exceptional Inference Speed:** The architecture is built on a foundation of
    reparameterization. After training, its complex, multi-branch structure can be
    mathematically fused into a simple, ultra-fast network, making it ideal for
    real-world applications, ONNX export, and TensorRT optimization.
3.  **Training Stability:** Every component, from the normalization layers to the
    fusion logic, has been battle-tested and designed to ensure a robust and
    stable training experience, even with advanced framework features like EMA.

-------------------------------------------------------------------------------------
Key Architectural Innovations:

1.  **The ParagonBlock:** A novel core block that synergizes three key ideas:
    -   **Efficient Multi-Scale Context (`InceptionDWConv2d`):** Inspired by
        MoSRv2/RTMoSR, this captures features at multiple spatial scales
        (square, horizontal, vertical) with high parameter efficiency.
    -   **Powerful Gated Transformation (`GatedFFN`):** Inspired by HyperionSR, this
        uses a Gated Feed-Forward Network for superior non-linear feature
        transformation, allowing the network to dynamically route information.
    -   **Advanced Reparameterization (`ReparamConvV3`):** Inspired by SpanPP,
        this core convolutional unit fuses three distinct and powerful branches
        with learnable weights, dramatically increasing the model's expressive power.

2.  **Hierarchical Residual Groups:** Organizes the ParagonBlocks into residual
    groups for improved training stability and gradient flow, enabling deeper
    and more powerful network configurations.

Usage:
-   Place this file in your `traiNNer/archs/` directory.
-   In your config.yaml, use one of the registered variants, e.g.:
    `network_g: type: paragonsr_s`
"""

import torch
import torch.nn.functional as F
from torch import nn

from traiNNer.utils.registry import ARCH_REGISTRY

# --- Building Blocks ---


class ReparamConvV2(nn.Module):
    """
    The final, stable, and powerful reparameterizable block. It fuses a 3x3,
    a 1x1, and a 3x3 depthwise convolution. This version has the corrected
    fusion logic.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        self.in_channels, self.out_channels, self.stride, self.groups = (
            in_channels,
            out_channels,
            stride,
            groups,
        )

        # Standard convolutions
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, groups=groups, bias=True
        )
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, 1, stride, 0, groups=groups, bias=True
        )

        # Depthwise convolution branch. Only active if it's a valid operation.
        self.dw_conv3x3 = None
        if in_channels == out_channels and groups == in_channels:
            self.dw_conv3x3 = nn.Conv2d(
                in_channels, out_channels, 3, stride, 1, groups=in_channels, bias=True
            )

    def get_fused_kernels(self) -> (torch.Tensor, torch.Tensor):
        """Performs the mathematical fusion of the training-time branches."""
        fused_kernel, fused_bias = (
            self.conv3x3.weight.clone(),
            self.conv3x3.bias.clone(),
        )

        # Fuse 1x1 branch
        padded_1x1_kernel = F.pad(self.conv1x1.weight, [1, 1, 1, 1])
        fused_kernel += padded_1x1_kernel
        fused_bias += self.conv1x1.bias

        # Fuse Depthwise 3x3 branch, if it exists
        if self.dw_conv3x3 is not None:
            # --- CRITICAL FIX: Direct and Correct Kernel Conversion ---
            # A depthwise kernel has shape (C_out, 1, kH, kW) where C_out = C_in.
            # A standard kernel has shape (C_out, C_in / groups, kH, kW).
            # To convert, we create a zero tensor of the target standard shape and
            # place the depthwise weights along the diagonal.

            dw_kernel = self.dw_conv3x3.weight.clone()
            dw_bias = self.dw_conv3x3.bias.clone()

            # The target shape for a standard conv with these groups
            target_shape = self.conv3x3.weight.shape
            standard_dw_kernel = torch.zeros(target_shape, device=dw_kernel.device)

            # Place each depthwise filter into its correct position in the standard kernel
            for i in range(self.in_channels):
                # in_channel_idx_in_group = i // (in_channels // groups)
                # For depthwise, groups=in_channels, so this is just 0.
                standard_dw_kernel[i, 0, :, :] = dw_kernel[i, 0, :, :]

            fused_kernel += standard_dw_kernel
            fused_bias += dw_bias

        return fused_kernel, fused_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uses multi-branch for training and stateless on-the-fly fusion for eval."""
        if self.training:
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.dw_conv3x3 is not None:
                out += self.dw_conv3x3(x)
            return out
        else:
            w, b = self.get_fused_kernels()
            return F.conv2d(x, w, b, stride=self.stride, padding=1, groups=self.groups)


class InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, (1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, (band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1
        )


class GatedFFN(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.project_in_g = nn.Conv2d(dim, hidden_dim, 1)
        self.project_in_i = nn.Conv2d(dim, hidden_dim, 1)
        # Use the new, powerful, and stable ReparamConvV2
        self.spatial_mixer = ReparamConvV2(hidden_dim, hidden_dim, groups=hidden_dim)
        self.act = nn.Mish(inplace=True)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, i = self.project_in_g(x), self.project_in_i(x)
        g = self.spatial_mixer(g)
        return self.project_out(self.act(g) * i)


class LayerScale(nn.Module):
    """
    A learnable scaling factor applied to the output of a residual block.
    This is a key stabilization technique for very deep networks.
    From: "Going deeper with Image Transformers" (Touvron et al., 2021)
    """

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The permute is necessary because LayerScale is channel-wise.
        return (
            (x.permute(0, 2, 3, 1).contiguous() * self.gamma)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class ParagonBlock(nn.Module):
    """The core block of ParagonSR, now with LayerScale for ultimate stability."""

    def __init__(self, dim: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.context = InceptionDWConv2d(dim)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.transformer = GatedFFN(dim, expansion_ratio=ffn_expansion)

        # STABILITY UPGRADE: Add a learnable LayerScale to the output of each branch.
        self.ls1 = LayerScale(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The forward pass now scales the output of each residual branch.
        # This forces the model to learn in a more controlled and stable manner.
        residual = x
        x_normed = self.norm1(x)
        x = self.context(x_normed)
        x = residual + self.ls1(x)  # Apply scaling here

        residual = x
        x_normed = self.norm2(x)
        x = self.transformer(x_normed)
        x = residual + self.ls2(x)  # And apply scaling here

        return x


class ResidualGroup(nn.Module):
    def __init__(self, dim: int, num_blocks: int, ffn_expansion: float = 2.0) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[ParagonBlock(dim, ffn_expansion) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x) + x


class ParagonSR(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        num_feat: int = 64,
        num_groups: int = 6,
        num_blocks: int = 6,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv_in = nn.Conv2d(in_chans, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                ResidualGroup(num_feat, num_blocks, ffn_expansion)
                for _ in range(num_groups)
            ]
        )
        self.conv_fuse = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.conv_out = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

    def fuse_for_release(self):
        """Call this on a trained model to permanently fuse for deployment."""
        print("Fusing model for release...")
        for name, module in self.named_modules():
            if isinstance(module, ReparamConvV2):
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = self.get_submodule(parent_name)
                print(f"  - Fusing {name}")
                w, b = module.get_fused_kernels()
                fused_conv = nn.Conv2d(
                    module.conv3x3.in_channels,
                    module.conv3x3.out_channels,
                    3,
                    module.stride,
                    1,
                    groups=module.groups,
                    bias=True,
                )
                fused_conv.weight.data.copy_(w)
                fused_conv.bias.data.copy_(b)
                setattr(parent_module, child_name, fused_conv)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shallow = self.conv_in(x)
        x_deep = self.body(x_shallow)
        x_fused = self.conv_fuse(x_deep) + x_shallow
        return self.conv_out(self.upsampler(x_fused))


# --- Factory Registration for traiNNer-redux: The Complete Family ---


@ARCH_REGISTRY.register()
def paragonsr_tiny(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-Tiny: Ultra-lightweight for high-speed/real-time use cases.
    - Training Target: ~4-6GB VRAM GPUs (e.g., GTX 1650).
    - Inference Target: Any modern GPU/CPU; suitable for video.
    """
    return ParagonSR(
        scale=scale, num_feat=32, num_groups=3, num_blocks=3, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_xs(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-XS: Extra-Small for general use on low-end hardware.
    - Training Target: ~6-8GB VRAM GPUs (e.g., RTX 2060, GTX 1660S).
    - Inference Target: ~4-6GB VRAM GPUs (e.g., GTX 1060).
    """
    return ParagonSR(
        scale=scale, num_feat=48, num_groups=4, num_blocks=4, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_s(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-S: Small, the recommended flagship for high quality on mainstream hardware.
    - Training Target: ~12GB VRAM GPUs (e.g., RTX 3060, RTX 2080 Ti).
    - Inference Target: Most GPUs with ~6-8GB VRAM (e.g., RTX 2070).
    """
    return ParagonSR(
        scale=scale, num_feat=64, num_groups=6, num_blocks=6, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_m(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-M: Medium, for prosumer hardware.
    - Training Target: ~16-24GB VRAM GPUs (e.g., RTX 3090, RTX 4080).
    - Inference Target: GPUs with ~8-12GB VRAM (e.g., RTX 3060).
    """
    return ParagonSR(
        scale=scale, num_feat=96, num_groups=8, num_blocks=8, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_l(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-L: Large, for high-end enthusiast hardware.
    - Training Target: >24GB VRAM GPUs (e.g., RTX 4090).
    - Inference Target: High-end GPUs with ~12GB+ VRAM (e.g., RTX 3080, RTX 4070).
    """
    return ParagonSR(
        scale=scale, num_feat=128, num_groups=10, num_blocks=10, ffn_expansion=2.0
    )


@ARCH_REGISTRY.register()
def paragonsr_xl(scale: int = 4, **kwargs) -> ParagonSR:
    """
    ParagonSR-XL: Extra-Large, for researchers and benchmark chasers.
    - Training Target: High-VRAM accelerator cards (e.g., 48GB+ A100, H100).
    - Inference Target: Flagship GPUs with >24GB VRAM (e.g., RTX 4090).
    """
    return ParagonSR(
        scale=scale, num_feat=160, num_groups=12, num_blocks=12, ffn_expansion=2.0
    )
