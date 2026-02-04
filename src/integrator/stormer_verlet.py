import torch as tc
from torch import Tensor
from typing import Callable
from tqdm import tqdm


def stormer_verlet(x0: Tensor, f_grad: Callable, dt: float, num_steps: int,
                   scheme="p_half") -> Tensor:
    """
    Returns the integrated trajectory given initial conditions.

    Scheme "p_half" (explicit if Hamiltonian separable: dqdt does not depend on q and dpdt not on p)
        p_half = p_prev - 0.5*dt * H_grad_q(q_prev, p_half)
               = p_prev + 0.5*dt * dpdt(q_prev, _)                  ===> requires dpdt not depend on p
        q_next = q_prev + 0.5*dt * (H_grad_p(q_prev, p_half) + H_grad_p(q_next, p_half))
               = q_prev + dt * H_grad_p(_, p_half)                  ===> requires dqdt not depend on q
               = q_prev + dt * dqdt(_, p_half)
        p_next = p_half - 0.5*dt * H_grad_q(q_next, p_half)
               = p_half + 0.5*dt * dpdt(q_next, p_half)

    Scheme "q_half" (explicit if Hamiltonian is separable: dqdt does not depend on q and dpdt not on p)
        q_half = q_prev + 0.5*dt * H_grad_p(q_half, p_prev)
               = q_prev + 0.5*dt * dqdt(q_half, p_prev)             ==> requires dqdt not depend on q
        p_next = p_prev - 0.5*dt * (H_grad_q(q_half, p_prev) + H_grad_q(q_half, p_next))
               = p_prev - dt * H_grad_q(q_half, p_prev)             ==> requires dpdt not depend on p
               = p_prev + dt * dpdt(q_half, p_prev)
        q_next = q_half + 0.5*dt * H_grad_p(q_half, p_next)
               = q_half + 0.5*dt * dqdt(q_half, p_next)

    Args:
        x0              Initial conditions of shape (num_traj, num_features)
        f_grad          Gradient function (returns dynamics dx/dt given x)
        dt              Time step size
        num_steps       Number of integration steps to take
    Returns:
        traj        Resulting trajectory of shape (num_traj, num_steps, num_features)
    """
    assert(
        len(x0.shape) == 2
    ), f"Expected shape (num_traj, num_features) for the initial condition x0"
    num_trajs, num_features = x0.shape
    traj = tc.empty(num_trajs, num_steps + 1, num_features, device=x0.device, dtype=x0.dtype)
    traj[:, 0, :] = x0

    for step_idx in tqdm(range(0, num_steps)):
        x_prev = traj[:, step_idx, :]
        q_prev, p_prev = tc.tensor_split(x_prev, 2, dim=-1)

        if scheme == "p_half":
            # Step 1: Compute dpdt_prev to compute p half
            dxdt_prev = f_grad(x_prev)
            _, dpdt_prev = tc.tensor_split(dxdt_prev, 2, dim=-1)

            # Step 2: Compute p next (half)
            p_half = p_prev + 0.5 * dt * dpdt_prev

            # Step 3: Compute q next
            dxdt_half = f_grad(tc.cat([q_prev, p_half], dim=-1))
            dqdt_half, _ = tc.tensor_split(dxdt_half, 2, dim=-1)
            q_next = q_prev + dt * dqdt_half

            # Step 4: Compute p next (other half)
            dxdt_next = f_grad(tc.cat([q_next, p_half], dim=-1))
            _, dpdt_next = tc.tensor_split(dxdt_next, 2, dim=-1)
            p_next = p_half + 0.5 * dt * dpdt_next
        elif scheme == "q_half":
            # Step 1: Compute dpdt_prev to compute p half
            dxdt_prev = f_grad(x_prev)
            dqdt_prev, _ = tc.tensor_split(dxdt_prev, 2, dim=-1)

            # Step 2: Compute q next (half)
            q_half = q_prev + 0.5 * dt * dqdt_prev

            # Step 3: Compute p next
            dxdt_half = f_grad(tc.cat([q_half, p_prev], dim=-1))
            _, dpdt_half = tc.tensor_split(dxdt_half, 2, dim=-1)
            p_next = p_prev + dt * dpdt_half

            # Step 4: Compute q next (other half)
            dxdt_next = f_grad(tc.cat([q_half, p_next], dim=-1))
            dqdt_next, _ = tc.tensor_split(dxdt_next, 2, dim=-1)
            q_next = q_half + 0.5 * dt * dqdt_next
        else:
            raise ValueError("Unknown scheme for Stormer-Verlet: Use 'p_half' or 'q_half'")

        x_next = tc.cat([q_next, p_next], dim=-1)
        traj[:, step_idx + 1, :] = x_next.detach()

    return traj