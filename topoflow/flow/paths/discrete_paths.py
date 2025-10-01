"""Conditional probability path for discrete (categorical) data."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from topoflow.flow.paths.abstract_paths import ConditionalProbabilityPath
from topoflow.flow.utils import DistributionNodes  # if you use variable-sized data


class DiscreteConditionalProbabilityPath(ConditionalProbabilityPath):
    """Conditional probability path for discrete (categorical) data.

    Design choices:
      - Distributions are represented by logits over `n_classes`.
      - The path interpolates logits linearly:
            logits_t = (1 - alpha(t)) * logits_simple + alpha(t) * logits_data
        where logits_simple is typically the zero vector (uniform prior).
      - Sampling: returns either probabilities p_t = softmax(logits_t)
        or a relaxed Gumbel-softmax sample (useful for gradient-based training).
      - The 'score' returned is the derivative of log p_t w.r.t. logits,
        which for a categorical is (one_hot(x) - p_t).
      - The 'vector field' is the time derivative of logits_t: d/dt logits_t.
    """

    def __init__(
        self,
        p_data,
        alpha,
        device="cpu",
        n_classes: int = None,
        distribution_hist=None,
    ):
        """Initializes the DiscreteConditionalProbabilityPath.

        Args:
            p_data: data distribution (must implement sample(batch_size, ...)). For discrete
                    datasets it should yield integer labels or one-hot/logit representations.
            alpha: time-dependent scalar function with call signature alpha(t) and .dt(t) for derivative.
                   alpha(t) should map t in [0,1] -> [0,1] (interpreted as interpolation weight).
            device: torch device.
            n_classes: number of categories. If None, tries to infer from p_data.
            distribution_hist: optional size distribution for variable-sized discrete data.
        """
        # p_simple will be an implicit uniform logits (zeros). We still pass a dummy to base class:
        # create a "simple" distribution that returns logits of shape [1, n_classes] when sampled
        if n_classes is None:
            try:
                n_classes = p_data.n_classes
            except AttributeError:
                raise ValueError(
                    "n_classes must be provided or p_data must expose n_classes"
                )

        # Inherit p_simple as a placeholder; the base class expects p_simple.sample to exist.
        class _DummySimple:
            def __init__(self, n):
                self.n = n

            def sample(self, *args, **kwargs):
                # return a "logits-like" tensor of shape [1, n_classes] — base only used to deduce dims
                return torch.zeros(1, self.n)

        p_simple = _DummySimple(n_classes)
        super().__init__(p_simple, p_data, device=device)

        self.alpha = alpha
        self.alpha.device = device
        self.n_classes = n_classes

        if distribution_hist is not None:
            self.size_distribution = DistributionNodes(distribution_hist)
        else:
            try:
                self.size_distribution = DistributionNodes(
                    p_data.dataset.distribution_hist
                )
            except Exception:
                # size_distribution is optional for discrete fixed size datasets
                self.size_distribution = None

    # ---------------------------
    # Helpers to normalize input types
    # ---------------------------
    # def _to_logits_from_z(self, z: torch.Tensor) -> torch.Tensor:
    #     """
    #     Turn conditioning variable z into logits over classes.

    #     Accepts:
    #      - integer labels tensor shape [N] with dtype long
    #      - one-hot tensors [N, n_classes]
    #      - logits [N, n_classes] (float)

    #     Returns:
    #      logits [N, n_classes] (float)
    #     """
    #     if z.dtype in (torch.long, torch.int):
    #         # integer labels -> one-hot logits: use large positive for the true class
    #         one_hot = F.one_hot(z, num_classes=self.n_classes).to(torch.float32)
    #         # convert one-hot to logits: give true class a high logit (e.g., +10), others 0
    #         logits = one_hot * 10.0  # design choice — sharp logits for observed data
    #     else:
    #         # assume float: could be probabilities, one-hot, or logits
    #         if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
    #             # single-dim numeric — treat as labels
    #             z_long = z.view(-1).long()
    #             one_hot = F.one_hot(z_long, num_classes=self.n_classes).to(torch.float32)
    #             logits = one_hot * 10.0
    #         else:
    #             # float tensor of shape [N, n_classes]
    #             # if values sum to 1, treat as probabilities and convert to logits
    #             if torch.allclose(z.sum(dim=-1), torch.ones(z.size(0), device=z.device), atol=1e-4):
    #                 # avoid log(0)
    #                 eps = 1e-12
    #                 logits = torch.log(z.clamp(min=eps))
    #             else:
    #                 # already logits (or something similar)
    #                 logits = z.to(torch.float32)
    #     return logits.to(self.device)

    def sample_simple_distribution(
        self, batch_size: int, hard: bool = True
    ) -> torch.Tensor:
        """Samples from the simple (prior) distribution: uniform categorical over n_classes.

        Args:
            batch_size: Number of samples to generate.
            hard: If True, returns one-hot vectors (N x n_classes).
                If False, returns probabilities (uniform distribution).

        Returns:
            Tensor of shape [N, n_classes].
        """
        if hard:
            # Sample integer labels uniformly
            labels = torch.randint(
                low=0, high=self.n_classes, size=(batch_size,), device=self.device
            )
            one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
            return one_hot
        else:
            # Just return uniform probabilities
            return torch.full(
                (batch_size, self.n_classes),
                1.0 / self.n_classes,
                device=self.device,
                dtype=torch.float32,
            )

    def sample_conditioning_variable(
        self, batch_size: int, pocket_only=False, force_batch_size=False
    ):
        """Samples the conditioning variable `z`.

        Args:
            batch_size: The number of samples to generate.
            pocket_only: If True, only samples pocket atoms.
            force_batch_size: If True, ensures exactly `batch_size` samples are returned.

        Returns:
            A tuple containing:
                - `z`: The conditioning variable sampled from the data distribution.
                  [n_atoms, *dims]
                - `batch_lig`: The ligands associated with the samples.
                  [n_atoms, *dims]
                - `batch_pocket`: The pockets associated with the samples.
                  [n_atoms, *dims]
        """
        z, batch_all, batch_lig, batch_pocket, batch_size = self.p_data.sample(
            batch_size, force_batch_size
        )
        if pocket_only:
            z = z[torch.where(batch_pocket != -1)[0], :]  # Only keep pocket atoms
            batch_lig = None
        else:
            batch_lig = batch_lig.to(self.device)
        z = z.to(self.device)
        batch_all = batch_all.to(self.device)
        batch_pocket = batch_pocket.to(self.device)

        return z, batch_all, batch_lig, batch_pocket, batch_size

    def sample_conditional_path(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        start_points: Optional[torch.Tensor] = None,
        relaxed: bool = True,
        temperature: float = 0.5,
        hard: bool = False,
    ) -> torch.Tensor:
        """Sample from p_t(x | z).

        During training we typically want a differentiable sample, so we use a relaxed=True for
        a Gumbel-softmax sample. During sampling we can use relaxed=False to get actual probabilities.
        Hard=True can be used with relaxed=True to get straight-through one-hot samples. It introduces
        bias but can be useful if the model expects discrete inputs.

        Args:
          z: conditioning variable, given as logits or labels.
          t: time tensor shaped to broadcast (see base class)
          start_points: ignored for discrete (kept for API compatibility)
          relaxed: whether to return a relaxed Gumbel-soft sample (if True) or raw probabilities p_t.
          temperature: Gumbel-soft temperature if relaxed=True.
          hard: if True and relaxed=True returns straight-through discrete samples.

        Returns:
          If relaxed=True:
             tensor of shape [N, n_classes] with relaxed samples in the probability simplex
             (or hard one-hot if hard=True, using straight-through).
          If relaxed=False:
             returns p_t = softmax(logits_t) [N, n_classes] (probabilities).
        """
        # convert conditioning to logits
        # logits_data = self._to_logits_from_z(z)  # [N, C]
        # simple prior logits (uniform) -> zeros
        logits_simple = torch.zeros_like(z, device=self.device)

        # broadcast alpha(t) to per-sample weight: alpha_t shape should broadcast to [N, 1]
        alpha_t = self.alpha(t).view(-1, *([1] * (z.dim() - 1))).to(self.device)

        logits_t = (1.0 - alpha_t) * logits_simple + alpha_t * z  # [N, C]
        p_t = F.softmax(logits_t, dim=-1)

        if relaxed:
            # Gumbel-softmax (reparameterized) gives differentiable samples
            # sample Gumbel noise
            gumbel = -torch.log(-torch.log(torch.rand_like(p_t).clamp(min=1e-12)))
            y = F.softmax(
                (torch.log(p_t.clamp(min=1e-12)) + gumbel) / max(temperature, 1e-8),
                dim=-1,
            )
            if hard:
                # straight-through hard one-hot
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
            return y
        else:
            # return probabilities
            return p_t

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Returns the conditional vector field u_t(x|z).

        In discrete/logit space the 'vector field' is the derivative of logits_t wrt t:
            u_t = d/dt logits_t = alpha.dt(t) * (logits_data - logits_simple)
        In the discrete case this does not depend on x. It only depends on z (the conditioning logits),
        the time t, and the path alpha.

        Args:
          x: not used for logits derivative (kept for API compatibility)
          z: conditioning variable (logits or labels)
          t: time

        Returns:
          tensor of shape [N, n_classes]: time derivative of logits at t.
        """
        # logits_simple = torch.zeros_like(z, device=self.device) # uniform prior logits

        d_alpha = self.alpha.dt(t).view(-1, *([1] * (z.dim() - 1))).to(self.device)
        u = d_alpha * (z)  # - logits_simple) # logits_simple is zero
        return u

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Discrete conditional score: gradient of log p_t(x | z) w.r.t. logits_t.

        For categorical distribution with logits logits_t and probability p_t = softmax(logits_t):
            d/d logits_t log p_t(x) = one_hot(x) - p_t

        Args:
          x: either integer labels [N] or one-hot / prob vectors [N, C]
          z: conditioning variable (logits/labels) -> used to build logits_t
          t: time

        Returns:
          Tensor of shape [N, C] giving gradients w.r.t logits_t (one_hot - p_t).
        """
        logits_data = self._to_logits_from_z(z)
        logits_simple = torch.zeros_like(logits_data, device=self.device)

        alpha_t = (
            self.alpha(t).view(-1, *([1] * (logits_data.dim() - 1))).to(self.device)
        )
        logits_t = (1.0 - alpha_t) * logits_simple + alpha_t * logits_data
        p_t = F.softmax(logits_t, dim=-1)

        # build one-hot from x if x is integer labels
        if x.dtype in (torch.long, torch.int) or (x.ndim == 1):
            one_hot = (
                F.one_hot(x.view(-1).long(), num_classes=self.n_classes)
                .to(p_t.dtype)
                .to(self.device)
            )
        else:
            # assume the provided x is already a one-hot/hard vector; if it is a probability vector
            # we still use it as soft target (this yields the expected gradient).
            one_hot = x.to(p_t.dtype).to(self.device)
            if one_hot.shape != p_t.shape:
                raise ValueError(
                    "x shape incompatible with p_t shape for computing discrete score."
                )

        score = one_hot - p_t
        return score
