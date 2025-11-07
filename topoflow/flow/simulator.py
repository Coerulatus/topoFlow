"""ODE and SDE simulator classes."""

from abc import ABC, abstractmethod
import time

import torch
import torch.nn.functional as F
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


class SDE(ODE):
    """Abstract class representing an SDE."""

    @abstractmethod
    def diffusion_coefficient(
        self, xt: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Diffusion coefficient of the SDE.

        Args:
            xt: Current state of the system.
            t: Current time.
            **kwargs: Additional keyword arguments for the SDE.

        Returns:
            Diffusion coefficient at the current state and time.
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


class EulerMaruyamaSimulator(Simulator):
    """Class for simulating SDEs using the Euler-Maruyama method."""

    def __init__(self, sde: SDE):
        """Initializes the simulator with an SDE.

        Args:
            sde: The SDE to be simulated.
        """
        self.sde = sde

    def step(
        self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Takes a step in the simulation using the Euler-Maruyama method.

        Args:
            xt: Current state of the system. [batch_size, dim]
            t: Current time. [batch_size]
            dt: Time step size. [batch_size]
            **kwargs: Additional keyword arguments for the SDE.

        Returns:
            Updated state of the system after taking a step.
        """
        drift = self.sde.drift_coefficient(xt, t, **kwargs)
        dt = dt.view(-1, *([1] * (drift.ndim - 1)))
        diffusion = self.sde.diffusion_coefficient(xt, t, **kwargs)
        dW = torch.randn_like(xt) * (dt**0.5)  # Brownian increment
        return xt + drift * dt + diffusion * dW


class EulerSimulator(Simulator):
    """Class for simulating ODEs using the Euler method."""

    def __init__(self, ode: ODE):
        """Initializes the simulator with an ODE.

        Args:
            ode: The ODE to be simulated.
        """
        self.ode = ode

    def step(
        self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Takes a step in the simulation using the Euler method.

        Args:
            xt: Current state of the system. [batch_size, dim]
            t: Current time. [batch_size]
            dt: Time step size. [batch_size]
            **kwargs: Additional keyword arguments for the ODE.

        Returns:
            Updated state of the system after taking a step.
        """
        drift = self.ode.drift_coefficient(xt, t, **kwargs)
        dt = dt.view(-1, *([1] * (drift.ndim - 1)))
        return xt + drift * dt


class TorchDiffEqSimulator:
    """Class for simulating ODEs using torchdiffeq."""

    def __init__(self, ode: ODE, solver: str = "rk4"):
        """Initializes the simulator with an ODE.

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
        self, x, inverse_traj, ts, batch_all, batch_lig, batch_pocket, z, mask
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

        Returns:
            torch.Tensor: Inpainted states at each time step.
        """

        def ode_func(t, x):
            t_idx = (ts - t).abs().argmin()
            inverse_state = inverse_traj[:, t_idx]
            x[mask] = inverse_state[mask]
            return self.ode.drift_coefficient(
                x, t, batch_all, batch_lig, batch_pocket, z=z
            )

        sol = odeint(ode_func, x, ts, method=self.solver)
        return sol.transpose(0, 1)

    def step(self, xt, t, dt):
        """Step method for TorchDiffEqSimulator."""
        pass

    def integration_error(self, x: torch.Tensor, ts: torch.Tensor):
        """Computes the integration error over a sequence of time steps.

        Args:
            x: Initial state of the system. [batch_size, *dims]
            ts: Sequence of time steps. [batch_size, time_steps]
            y: Conditioning variable. (Label for guided data) [batch_size, 1].
                Defaults to None.
            guidance_scale: Scale for guidance. Defaults to 0.0.

        Returns:
            A tuple containing:
                - torch.Tensor: Mean L2 norm difference between DOPRI5 and RK4 solutions.
                - float: Time taken for DOPRI5 integration.
                - float: Time taken for RK4 integration.
        """
        ode_func = lambda t, x: self.ode.drift_coefficient(x, t)

        time_start = time.time()
        sol_dopri = odeint(ode_func, x, ts, method="dopri5")
        time_dopri = time.time() - time_start

        time_start = time.time()
        sol_rk4 = odeint(ode_func, x, ts, method="rk4")
        time_rk4 = time.time() - time_start

        return (
            torch.mean(torch.norm(sol_dopri - sol_rk4, p=2, dim=1)),
            time_dopri,
            time_rk4,
        )
