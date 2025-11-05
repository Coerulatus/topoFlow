"""Data augmentation utilities for 3D molecular data."""

import torch
import math
from torch_geometric.data import Data


class RandomRotateTranslate:
    """Randomly rotate and translate 3D coordinates in a Data object."""

    def __init__(self, translate_scale: float = 0):
        """Initialize the RandomRotateTranslate augmentation.

        Args:
            translate_scale (float): Max absolute value for translation along each axis.
        """
        self.translate_scale = translate_scale

    def __call__(self, data: Data) -> Data:
        """Apply random rotation and translation to the 3D coordinates in the Data object."""
        # Generate a random 3D rotation matrix using QR decomposition
        A = torch.randn(3, 3)
        Q, R = torch.linalg.qr(A)
        M = Q @ torch.diag(torch.sign(torch.diagonal(R)))  # ensure right-handed system

        # Random translation vector
        t = torch.empty(3).uniform_(-self.translate_scale, self.translate_scale)

        # Apply transformation
        for attr in ["x_lig", "x_pocket"]:
            if hasattr(data, attr):
                x = getattr(data, attr)
                x_rot = x @ M.T + t  # Rotate then translate
                setattr(data, attr, x_rot)

        return data
