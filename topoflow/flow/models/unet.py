"""Neural network architectures for image processing, specifically a U-Net for MNIST."""

import math
from typing import List, Optional

import torch
import torch.nn as nn


class FourierEncoder(nn.Module):
    """Encodes time information using a Fourier feature mapping.

    This encoder converts a scalar time input into a higher-dimensional
    representation using sine and cosine functions with learnable frequencies.
    It's based on the technique described in
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183.
    """

    def __init__(self, dim: int):
        """Initializes the FourierEncoder.

        Args:
            dim: The output dimension of the encoding. Must be an even number.
        """
        super().__init__()
        assert dim % 2 == 0, "FourierEncoder dimension must be even."
        self.half_dim = dim // 2
        # Learnable random frequencies
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FourierEncoder.

        Args:
            t: The input time tensor. Expected shape: (batch_size, 1, 1, 1) or (batch_size, 1).

        Returns:
            The Fourier-encoded time embedding. Shape: (batch_size, dim).
        """
        t = t.view(-1, 1)  # Reshape to (batch_size, 1)
        freqs = t * self.weights * 2 * math.pi  # (batch_size, half_dim)
        sin_embed = torch.sin(freqs)  # (batch_size, half_dim)
        cos_embed = torch.cos(freqs)  # (batch_size, half_dim)
        # Concatenate sine and cosine components and scale
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)


class ResidualLayer(nn.Module):
    """A residual block with two convolutional layers, SiLU activation, and BatchNorm.

    It can optionally incorporate time and conditional embeddings.
    """

    def __init__(
        self, channels: int, time_embed_dim: Optional[int], y_embed_dim: Optional[int]
    ):
        """Initializes the ResidualLayer.

        Args:
            channels: The number of input and output channels for the convolutional layers.
            time_embed_dim: The dimension of the time embedding. If None, time embedding
                is not used.
            y_embed_dim: The dimension of the conditional (label) embedding. If None,
                conditional embedding is not used.
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.time_adapter = None
        if time_embed_dim is not None:
            # Transforms time embedding to match 'channels' for addition
            self.time_adapter = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, channels),
            )
        self.y_adapter = None
        if y_embed_dim is not None:
            # Transforms conditional embedding to match 'channels' for addition
            self.y_adapter = nn.Sequential(
                nn.Linear(y_embed_dim, y_embed_dim),
                nn.SiLU(),
                nn.Linear(y_embed_dim, channels),
            )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: Optional[torch.Tensor] = None,
        y_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the ResidualLayer.

        Args:
            x: The input feature map. Shape: (batch_size, channels, height, width).
            t_embed: The time embedding. Shape: (batch_size, time_embed_dim).
                Defaults to None.
            y_embed: The conditional embedding. Shape: (batch_size, y_embed_dim).
                Defaults to None.

        Returns:
            The output feature map after the residual block.
            Shape: (batch_size, channels, height, width).
        """
        res = x.clone()  # Store residual connection

        # Initial convolutional block
        x = self.block1(x)

        # Add time embedding if provided
        if t_embed is not None and self.time_adapter is not None:
            # Adapt time_embed to (bs, c, 1, 1) for broadcasting
            t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + t_embed

        # Add conditional (label) embedding if provided
        if y_embed is not None and self.y_adapter is not None:
            # Adapt y_embed to (bs, c, 1, 1) for broadcasting
            y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)
            x = x + y_embed

        # Second convolutional block
        x = self.block2(x)

        # Add back residual connection
        x = x + res

        return x


class Encoder(nn.Module):
    """An encoder block consisting of multiple residual layers and a downsampling convolution."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        t_embed_dim: Optional[int],
        y_embed_dim: Optional[int],
    ):
        """Initializes the Encoder block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels after downsampling.
            num_residual_layers: The number of ResidualLayer blocks to use.
            t_embed_dim: The dimension of the time embedding.
            y_embed_dim: The dimension of the conditional (label) embedding.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_in, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )
        self.downsample = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: Optional[torch.Tensor],
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the Encoder block.

        Args:
            x: The input feature map. Shape: (batch_size, channels_in, height, width).
            t_embed: The time embedding. Shape: (batch_size, t_embed_dim).
            y_embed: The conditional embedding. Shape: (batch_size, y_embed_dim).

        Returns:
            The output feature map after residual blocks and downsampling.
            Shape: (batch_size, channels_out, height // 2, width // 2).
        """
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Downsample
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    """A mid-block in the U-Net.

    Typically consisting of multiple residual layers without downsampling or upsampling.
    """

    def __init__(
        self,
        channels: int,
        num_residual_layers: int,
        t_embed_dim: Optional[int],
        y_embed_dim: Optional[int],
    ):
        """Initializes the Midcoder block.

        Args:
            channels: The number of input and output channels for this block.
            num_residual_layers: The number of ResidualLayer blocks to use.
            t_embed_dim: The dimension of the time embedding.
            y_embed_dim: The dimension of the conditional (label) embedding.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: Optional[torch.Tensor],
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the Midcoder block.

        Args:
            x: The input feature map. Shape: (batch_size, channels, height, width).
            t_embed: The time embedding. Shape: (batch_size, t_embed_dim).
            y_embed: The conditional embedding. Shape: (batch_size, y_embed_dim).

        Returns:
            The output feature map after the residual blocks.
            Shape: (batch_size, channels, height, width).
        """
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class Decoder(nn.Module):
    """A decoder block consisting of an upsampling operation and multiple residual layers."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        num_residual_layers: int,
        t_embed_dim: Optional[int],
        y_embed_dim: Optional[int],
    ):
        """Initializes the Decoder block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels after upsampling.
            num_residual_layers: The number of ResidualLayer blocks to use.
            t_embed_dim: The dimension of the time embedding.
            y_embed_dim: The dimension of the conditional (label) embedding.
        """
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),  # Added align_corners for consistency
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualLayer(channels_out, t_embed_dim, y_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        t_embed: Optional[torch.Tensor],
        y_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass for the Decoder block.

        Args:
            x: The input feature map. Shape: (batch_size, channels_in, height, width).
            t_embed: The time embedding. Shape: (batch_size, t_embed_dim).
            y_embed: The conditional embedding. Shape: (batch_size, y_embed_dim).

        Returns:
            The output feature map after upsampling and residual blocks.
            Shape: (batch_size, channels_out, 2 * height, 2 * width).
        """
        # Upsample
        x = self.upsample(x)

        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class MNISTUNet(nn.Module):
    """A U-Net architecture specifically designed for MNIST-like image data.

    This U-Net incorporates time and optional conditional (label) embeddings
    to guide the network's output. It follows a standard encoder-decoder
    structure with skip connections.
    """

    def __init__(
        self,
        channels: List[int],
        num_residual_layers: int,
        t_embed_dim: Optional[int] = None,
        y_embed_dim: Optional[int] = None,
        use_residual: bool = True,
        batch_norm: bool = False,
    ):
        """Initializes the MNISTUNet.

        Args:
            channels: A list of integers specifying the number of channels at each
                level of the U-Net. `channels[0]` is the initial channel count,
                and `channels[-1]` is the bottleneck channel count.
            num_residual_layers: The number of residual layers within each Encoder,
                Midcoder, and Decoder block.
            t_embed_dim: The dimension of the time embedding. If None, time embedding
                is not used. Defaults to None.
            y_embed_dim: The dimension of the conditional (label) embedding. If None,
                conditional embedding is not used. Defaults to None.
            use_residual: If True, uses skip connections from encoder outputs to
                decoder inputs. Defaults to True.
            batch_norm: If True, applies BatchNorm2d after the midcoder. Defaults to False.
        """
        super().__init__()
        # Initial convolution to increase channels from 1 to channels[0]
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Initialize time embedder if dimension is provided
        self.time_embedder = None
        if t_embed_dim is not None:
            self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y (conditional) embedder if dimension is provided
        self.y_embedder = None
        # Assuming 11 classes for MNIST (0-9 + one for unconditional if needed)
        if y_embed_dim is not None:
            self.y_embedder = nn.Embedding(num_embeddings=11, embedding_dim=y_embed_dim)

        encoders = []
        decoders = []
        # Create encoder blocks
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(
                Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )
        # Create decoder blocks (in reverse order for U-Net structure)
        for curr_c, next_c in zip(
            channels[1:], channels[:-1]
        ):  # Corrected zip for decoder channels
            decoders.append(
                Decoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim)
            )

        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(channels[-1])

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(
            reversed(decoders)
        )  # Decoders are reversed to match encoder order

        # Midcoder block at the bottleneck
        self.midcoder = Midcoder(
            channels[-1], num_residual_layers, t_embed_dim, y_embed_dim
        )

        # Final convolution to reduce channels back to 1
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

        self.channels = channels
        self.use_residual = use_residual

    def encode(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        """Encodes the input image through the U-Net's downsampling path.

        Args:
            x: Input image tensor. Shape: (batch_size, 1, 32, 32).
            t: Time tensor. Shape: (batch_size, 1, 1, 1). Defaults to None.
            y: Conditional label tensor. Shape: (batch_size,). Defaults to None.

        Returns:
            A tuple containing:
                - torch.Tensor: The encoded feature map at the bottleneck.
                    Shape: (batch_size, channels[-1], h_bottleneck, w_bottleneck).
                - Optional[torch.Tensor]: The processed time embedding.
                    Shape: (batch_size, time_embed_dim) or None.
                - Optional[torch.Tensor]: The processed conditional embedding.
                    Shape: (batch_size, y_embed_dim) or None.
                - List[torch.Tensor]: A list of feature maps from each encoder stage
                    to be used as residual connections in the decoder.
        """
        # Embed time and conditional input if provided
        t_embed = None
        if t is not None and self.time_embedder is not None:
            t_embed = self.time_embedder(t)

        y_embed = None
        if y is not None and self.y_embedder is not None:
            y_embed = self.y_embedder(y)

        # Initial convolution
        x = self.init_conv(x)

        residuals = []

        # Pass through encoder blocks and store residuals
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed)
            if self.use_residual:
                residuals.append(x.clone())

        # Pass through midcoder
        x = self.midcoder(x, t_embed, y_embed)
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        return x, t_embed, y_embed, residuals

    def decode(
        self,
        x: torch.Tensor,
        t_embed: Optional[torch.Tensor],
        y_embed: Optional[torch.Tensor],
        residuals: List[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes the bottleneck feature map through the U-Net's upsampling path.

        Args:
            x: The bottleneck feature map. Shape: (batch_size, channels[-1], h, w).
            t_embed: The time embedding. Shape: (batch_size, t_embed_dim).
            y_embed: The conditional embedding. Shape: (batch_size, y_embed_dim).
            residuals: A list of residual feature maps from the encoder path,
                used for skip connections.

        Returns:
            The output image after decoding. Shape: (batch_size, 1, 32, 32).
        """
        # Decoders
        # The residuals list is processed in reverse order (pop from end)
        # because encoders were appended in forward order.
        for decoder in self.decoders:
            if self.use_residual:
                res = residuals.pop()  # Get the last added residual
                x = x + res  # Add skip connection
            x = decoder(x, t_embed, y_embed)

        # Final convolution to output the image
        x = self.final_conv(x)
        return x

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the MNIST U-Net.

        Args:
            x: Input image tensor. Shape: (batch_size, 1, 32, 32).
            t: Time tensor. Shape: (batch_size, 1, 1, 1).
            y: Conditional label tensor. Shape: (batch_size,). Defaults to None.

        Returns:
            The U-Net's output, typically representing noise or a vector field.
            Shape: (batch_size, 1, 32, 32).
        """
        # Encode the input
        x, t_embed, y_embed, residuals = self.encode(x, t, y)
        # Decode the bottleneck representation
        x = self.decode(x, t_embed, y_embed, residuals)

        return x

    @property
    def bottleneck_dim(self) -> int:
        """Calculates the flattened dimension of the bottleneck layer.

        Returns:
            The total number of elements in the bottleneck feature map
            (channels * height * width).
        """
        h, w = 32, 32
        for _ in self.encoders:
            h, w = h // 2, w // 2
        dim = self.channels[-1] * h * w
        return dim

    @property
    def bottleneck_shape(self) -> tuple[int, int, int]:
        """Calculates the spatial shape of the bottleneck layer.

        Returns:
            A tuple (channels, height, width) representing the shape of the
            feature map at the bottleneck.
        """
        h, w = 32, 32
        for _ in self.encoders:
            h, w = h // 2, w // 2
        return (self.channels[-1], h, w)
