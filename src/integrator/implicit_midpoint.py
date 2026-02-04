import torch as tc
from torch import Tensor
from typing import Callable
from tqdm import tqdm


def implicit_midpoint(x0: Tensor, f_grad: Callable, dt: float, num_steps: int,
                      max_iters: int = 10, tol: float = 1e-12) -> Tensor:
    """
    Symplectic implicit midpoint integrator for general Hamiltonians.

    Args:
        x0          Initial condition (num_traj, num_features)
        f_grad      Dynamics function f(x) = dx/dt
        dt          Step size
        num_steps   Number of integration steps
        max_iters   Iterations for fixed-point solver
        tol         Convergence tolerance
    Returns:
        traj        Trajectories (num_traj, num_steps+1, num_features)
    """
    num_trajs, num_features = x0.shape
    traj = tc.empty(num_trajs, num_steps + 1, num_features, device=x0.device, dtype=x0.dtype)
    traj[:, 0, :] = x0

    for step_idx in tqdm(range(num_steps)):
        x_prev = traj[:, step_idx, :]

        # Initial guess: explicit Euler
        x_next = x_prev + dt * f_grad(x_prev)

        for _ in range(max_iters):
            x_mid = 0.5 * (x_prev + x_next)
            F = x_prev + dt * f_grad(x_mid) - x_next  # residual

            err = F.norm(dim=-1).max().item()
            if err < tol:
                break

            # Fixed-point iteration (simple relaxation)
            x_next = x_prev + dt * f_grad(x_mid)

        traj[:, step_idx + 1, :] = x_next.detach()

    return traj