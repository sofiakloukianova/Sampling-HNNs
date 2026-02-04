import torch as tc
from torch import Tensor
from typing import Callable
from tqdm import tqdm


def symplectic_euler(x0: Tensor, f_grad: Callable, dt: float, num_steps: int,
                     scheme="p_first") -> Tensor:
    """
    Returns the integrated trajectory given initial conditions.

    Scheme "p_first" (explicit if dp/dt does not depend on p):
        p_next = p_prev - dt * H_grad_q(q_prev, p_next)
               = p_prev + dt * dpdt(q_prev, _)              ===> should not depend on p_next to be explicit
        q_next = q_prev + dt * H_grad_p(q_prev, p_next)
               = q_prev + dt * dqdt(q_prev, p_next)

    Scheme "q_first" (explicit if dq/dt does not depend on q):
        q_next = q_prev + dt * H_grad_p(q_next, p_prev)
               = q_prev + dt * dqdt(_, p_prev)              ===> should not depend on q_next to be explicit
        p_next = p_prev - dt * H_grad_q(q_next, p_prev)
               = p_prev + dt * dpdt(q_next, p_prev)

    Args:
        x0          Initial conditions of shape (num_traj, num_features)
        f_grad      Gradient function (returns dynamics dx/dt given x)
        dt          Time step size
        num_steps   Number of integration steps to take
        scheme      "p_first" or "q_first" (see above)
    Returns:
        traj        Resulting trajectory of shape (num_traj, num_steps+1, num_features)
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

        if scheme == "p_first":
            # Use this if dpdt (dedq) does not depend on p

            # Step 1: Update p (done explicitly, should not depend on p_next)
            dxdt_prev = f_grad(x_prev)
            _, dpdt_prev = tc.tensor_split(dxdt_prev, 2, dim=-1)
            p_next = p_prev + dt * dpdt_prev
            # Step 2: Update q (exact)
            dxdt_next = f_grad(tc.cat([q_prev, p_next], dim=-1))
            dqdt_next, _ = tc.tensor_split(dxdt_next, 2, dim=-1)
            q_next = q_prev + dt * dqdt_next
        elif scheme == "q_first":
            # Use this if dqdt (dedp) does not depend on q

            # Step 1: Update q (done explicitly, should not depend on q_next)
            dxdt_prev = f_grad(x_prev)
            dqdt_prev, _ = tc.tensor_split(dxdt_prev, 2, dim=-1)
            q_next = q_prev + dt * dqdt_prev

            # Step 2: Update p (exact)
            dxdt_next = f_grad(tc.cat([q_next, p_prev], dim=-1))
            _, dpdt_next = tc.tensor_split(dxdt_next, 2, dim=-1)
            p_next = p_prev + dt * dpdt_next
        else:
            raise ValueError("Unknown Symplectic Euler scheme: Use 'p_first' or 'q_first'")

        x_next = tc.cat([q_next, p_next], dim=-1)
        traj[:, step_idx + 1, :] = x_next.detach()

    return traj