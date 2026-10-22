"""ODE simulator classes."""

from abc import ABC, abstractmethod
import torch
from torchdiffeq import odeint


class ODE(ABC):
    """Abstract class representing an ODE."""

    @abstractmethod
    def drift_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Drift coefficient of the ODE.

        Args:
            xt: Current state of the system.
            t: Current time.
            **kwargs: Additional keyword arguments for the ODE.

        Returns:
            Drift coefficient at the current state and time.
        """
        pass


class Simulator(ABC):
    """Class for simulating dynamics."""

    @abstractmethod
    def step(
        self, xt: torch.Tensor, t: torch.Tensor, dt: float, **kwargs
    ) -> torch.Tensor:
        """Take a step in the simulation.

        Args:
            xt: Current state of the system.
            t: Current time.
            dt: Time step size.
            **kwargs: Additional keyword arguments for the simulator.

        Returns:
            Updated state of the system after taking a step.
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """Simulate the dynamics over a sequence of time steps.

        Args:
            x: Initial state of the system. [batch_size, dim]
            ts: Sequence of time steps. [batch_size, time_steps]
            **kwargs: Additional keyword arguments for the simulator.

        Returns:
            State at final time step.
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            dt = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, dt.unsqueeze(1), **kwargs)
        return x

    @torch.no_grad()
    def simulate_trajectory(
        self, x: torch.Tensor, ts: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Simulate the dynamics over a sequence of time steps.

        Args:
            x: Initial state of the system. [batch_size, dim]
            ts: Sequence of time steps. [batch_size, time_steps]
            **kwargs: Additional keyword arguments for the simulator.

        Returns:
            Simulated states at each time step.
        """
        trajectory = [x.clone()]
        for t_idx in range(ts.shape[1] - 1):
            t = ts[:, t_idx]
            dt = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, dt, **kwargs)
            trajectory.append(x.clone())
        return torch.stack(trajectory, 1)


class TorchDiffEqSimulator:
    """Class for simulating ODEs using torchdiffeq."""

    def __init__(self, ode: ODE, solver: str = "rk4"):
        """Initializes the simulator with an ODE and a solver.

        Args:
            ode: The ODE to be simulated.
            solver: The solver to be used. Supported solvers are 'dopri5' and
                'rk4'. Defaults to 'rk4'.
        """
        self.ode = ode
        if solver not in ["dopri5", "rk4"]:
            raise ValueError(
                f"Unsupported solver: {solver}. Supported solvers are 'dopri5' and 'rk4'."
            )
        self.solver = solver

    @torch.no_grad()
    def simulate(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        batch_all: torch.Tensor,
        batch_lig: torch.Tensor,
        batch_pocket: torch.Tensor,
        z: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Simulates the dynamics over a sequence of time steps.

        Args:
            x: Initial state of the system. [batch_size, *dims]
            ts: Sequence of time steps. [batch_size, time_steps]
            batch_all: The combined batch indices for all samples. [batch_size, *dims]
            batch_lig: The ligands associated with the samples. [batch_size, *dims]
            batch_pocket: The pockets associated with the samples. [batch_size, *dims]
            z: Optional sampled data.
            mask: Optional mask to apply during simulation to zero part of x. [batch_size, *dims]

        Returns:
            State at final time step.
        """
        if self.solver == "dopri5":
            t = torch.tensor([ts[0], ts[-1]], device=ts.device)
        else:
            t = ts[:]
        ode_func = lambda t, x: self.ode.drift_coefficient(
            x, t, batch_all, batch_lig, batch_pocket, z=z
        )

        if mask is not None:
            x *= mask

        sol = odeint(ode_func, x, t, method=self.solver)
        return sol.transpose(0, 1)

    def simulate_inpaint(
        self,
        x,
        inverse_traj,
        ts,
        t_max_inpaint,
        batch_all,
        batch_lig,
        batch_pocket,
        z,
        mask,
        end_points=None,
        force_scaffold=False,
        ts_inverse=None,
    ):
        """Generate molecule with inpainting using the inverse trajectory.

        The inpainting is done by replacing the masked parts of x with the
        corresponding parts from the inverse trajectory at each time step.

        Args:
        x : torch.Tensor
            Initial state of the system. [batch_size, *dims]
        inverse_traj : torch.Tensor
            Inverse trajectory obtained from the reverse simulation. [batch_size, time_steps, *dims]
        ts : torch.Tensor
            Sequence of time steps. [time_steps]
        t_max_inpaint : float
            Maximum time to consider for inpainting.
        batch_all : torch.Tensor
            The combined batch indices for all samples. [batch_size, *dims]
        batch_lig : torch.Tensor
            The ligands associated with the samples. [batch_size, *dims]
        batch_pocket : torch.Tensor
            The pockets associated with the samples. [batch_size, *dims]
        z : torch.Tensor
            Optional sampled data.
        mask : torch.Tensor
            Mask to apply during inpainting. [batch_size, *dims]
        end_points : torch.Tensor, optional
            Ground truth data for the scaffold. [batch_size, *dims]
        force_scaffold : bool, optional
            If True, fixes the scaffold atoms to end_points throughout the simulation.
        ts_inverse : torch.Tensor, optional
            Time grid corresponding to inverse_traj.

        Returns:
            torch.Tensor: Inpainted states at each time step.
        """
        ref_ts = (
            ts_inverse
            if ts_inverse is not None
            else torch.linspace(
                0, t_max_inpaint, inverse_traj.shape[1], device=ts.device
            )
        )

        def ode_func(t, x):
            if force_scaffold and end_points is not None:
                x[mask] = end_points[mask]
            elif t <= t_max_inpaint:
                t_idx = (ref_ts - t).abs().argmin()
                inverse_state = inverse_traj[:, t_idx]
                x[mask] = inverse_state[mask]
            return self.ode.drift_coefficient(
                x, t, batch_all, batch_lig, batch_pocket, z=z
            )

        sol = odeint(ode_func, x, ts, method=self.solver)
        return sol.transpose(0, 1)
