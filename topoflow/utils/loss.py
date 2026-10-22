"""Loss functions and guidance modules for diffusion models. Taken from TABASCO paper."""

from typing import List
from abc import ABC, abstractmethod

import torch

from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from lightning.pytorch import LightningModule
from torch_geometric.data import Data
from topoflow.utils.molecules import data_to_rdmol


def apply_mask(x, padding_mask):
    """Zero out padded atoms.

    Args:
        x: Tensor `(*, N, D)`.
        padding_mask: Bool/int tensor `(*, N)`.

    Returns:
        Tensor: Masked tensor.

    Note: Based on the NVIDIA-Digital-Bio/proteina function of same name
    """
    real_mask = (1 - padding_mask.int())[..., None]  # [*, N, 1]
    return x * real_mask


class InterpolantGuidance(ABC):
    """Interface for guidance modules that modify intermediate samples."""

    @abstractmethod
    def __call__(self, lightning_module: LightningModule, x_t, t: int):
        """Modify the partial sample `x_t` produced at timestep `t`."""
        pass


class UFFBoundGuidance(InterpolantGuidance):
    """Move atom coordinates towards locations that are within UFF distance bounds of their current conformation."""

    def __init__(
        self,
        atom_decoder,
        regress_to_center: bool = False,
    ):
        """Initializes the UFFBoundGuidance module.

        Args:
            atom_decoder: Decoder that converts tensor representations to RDKit molecules.
            lr: Step size applied to the sign of the gradient.
            n_steps: Number of gradient steps taken per guidance call.
            regress_to_center: If True, pull distances to interval centers instead of edges.
        """
        self.atom_decoder = atom_decoder
        self.regress_to_center = regress_to_center

    def _get_uff_bounds(self, mol: Chem.Mol):
        """Return the 2D matrix of UFF lower/upper bounds scaled to tensor space."""
        if mol is None:
            return None

        bound_matrix_params = {
            "set15bounds": True,
            "scaleVDW": True,
            "doTriangleSmoothing": True,
            "useMacrocycle14config": False,
        }

        bounds = GetMoleculeBoundsMatrix(mol, **bound_matrix_params)

        return torch.from_numpy(bounds).to(dtype=torch.float32)

    def __call__(
        self,
        u_t_theta_lig,
        lig,
        t,
        batch_lig,
        verbose=True,
        mean=None,
        std=None,
        velocity_scale: float = 1.0,
        t_threshold: float = None,
    ):
        """Return distance-bound loss on predicted endpoint coordinates."""
        if t_threshold is not None and t is not None:
            t_flat = t.view(-1)
            active_mask = t_flat > t_threshold
            if not active_mask.any():
                return torch.tensor(
                    0.0, device=lig.device, dtype=lig.dtype, requires_grad=True
                )
            lig = lig[active_mask]
            u_t_theta_lig = u_t_theta_lig[active_mask]
            t = t[active_mask]
            batch_lig = batch_lig[active_mask]
            _, batch_lig = torch.unique(batch_lig, sorted=True, return_inverse=True)

        scale = velocity_scale if velocity_scale and velocity_scale != 0.0 else 1.0
        endpoint = lig + (u_t_theta_lig / scale) * (1 - t)
        if mean is not None and std is not None:
            endpoint_pos = endpoint[:, :3] * std + mean
            endpoint = torch.cat([endpoint_pos, endpoint[:, 3:]], dim=-1)

        data = Data(
            x_lig=endpoint[:, :3].detach(),
            h_lig=endpoint[:, 3:].detach(),
            batch_lig=batch_lig,
        )
        endpoint_mols = data_to_rdmol(
            data, atom_decoder=self.atom_decoder, sanitize=True, verbose=verbose
        )
        if not isinstance(endpoint_mols, list):
            endpoint_mols = [endpoint_mols]
        endpoint_bounds = [self._get_uff_bounds(mol) for mol in endpoint_mols]

        loss = self._compute_loss(endpoint, endpoint_bounds, batch_lig)

        return loss

    def _loss_to_interval_edge(self, lb, ub, pwd):
        """Loss to the interval edge."""
        if lb <= pwd <= ub:
            dist_loss = 0
        elif pwd > ub:
            dist_loss = (pwd - ub) ** 2
        elif pwd <= lb:
            dist_loss = (pwd - lb) ** 2
        else:
            raise ValueError(f"Invalid distance: {pwd}, lb: {lb}, ub: {ub}")

        return dist_loss

    def _loss_to_interval_center(self, lb, ub, pwd):
        """Loss to the center of the interval."""
        interval_center = (ub + lb) / 2
        if lb <= pwd <= ub:
            dist_loss = 0
        else:
            dist_loss = (pwd - interval_center) ** 2

        return dist_loss

    def loss_from_bounds(self, tensor_repr, bounds):
        """Return mean squared deviation of pairwise distances from allowed bounds."""
        pairwise_dist = torch.cdist(
            tensor_repr[:, :3],
            tensor_repr[:, :3],
        )
        if pairwise_dist.shape != bounds.shape:
            print("Shapes do not match in loss_from_bounds. Returning 0 loss.")
            return torch.tensor(0.0, device=tensor_repr.device, requires_grad=True)

        bounds = bounds.to(device=pairwise_dist.device)

        num_atoms = pairwise_dist.shape[0]
        if num_atoms < 2:
            return torch.tensor(0.0, device=tensor_repr.device, requires_grad=True)

        ub_matrix = torch.triu(bounds, diagonal=0)
        lb_matrix = torch.tril(bounds, diagonal=0).transpose(0, 1)
        pw_dist_matrix = torch.triu(pairwise_dist, diagonal=0)

        above_ub_mask = pw_dist_matrix > ub_matrix
        below_lb_mask = pw_dist_matrix < lb_matrix

        if self.regress_to_center:
            interval_center = (ub_matrix + lb_matrix) / 2
            vectorized_loss = (
                pw_dist_matrix[above_ub_mask] - interval_center[above_ub_mask]
            ).pow(2).sum() + (
                pw_dist_matrix[below_lb_mask] - interval_center[below_lb_mask]
            ).pow(
                2
            ).sum()
        else:
            vectorized_loss = (
                pw_dist_matrix[above_ub_mask] - ub_matrix[above_ub_mask]
            ).pow(2).sum() + (
                pw_dist_matrix[below_lb_mask] - lb_matrix[below_lb_mask]
            ).pow(
                2
            ).sum()

        num_pairs = max(1.0, num_atoms * (num_atoms - 1) / 2)

        return vectorized_loss / num_pairs

    def _compute_loss(self, pred, bounds, batch):
        """Aggregate distance-bound loss over a batch, skipping None bounds."""
        len_valid_bounds = sum(1 for bound in bounds if bound is not None)
        if len_valid_bounds == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        batch_ids = torch.unique(batch, sorted=True)
        loss_sum = torch.tensor(0.0, device=pred.device)
        for i, tensor_bound in enumerate(bounds):
            if tensor_bound is None or i >= len(batch_ids):
                continue
            loss_sum = loss_sum + self.loss_from_bounds(
                pred[batch == batch_ids[i]], tensor_bound
            )
        loss = loss_sum / len_valid_bounds

        return loss
