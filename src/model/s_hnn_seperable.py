import torch.nn as nn
from numpy import ndarray
import numpy as np
from typing import Any

from error import l2_error_rel

from .base import BaseModel
from .s_mlp import S_MLP


import torch.nn as nn
from numpy import ndarray
import numpy as np
from typing import Any

from error import l2_error_rel

from .base import BaseModel
from .s_mlp import S_MLP


class S_HNN_SEP(nn.Module, BaseModel):
    """
    Separable Hamiltonian Neural Network with two MLPs:
    - mlp_pot: models potential energy V(q)
    - mlp_kin: models kinetic energy T(p)
    - H(q,p) = V(q) + T(p)
    """

    T = BaseModel.T

    mlp_pot: S_MLP  # Potential energy network V(q)
    mlp_kin: S_MLP  # Kinetic energy network T(p)

    def __init__(self, input_dim, hidden_dim, activation, resample_duplicates,
                 rcond, random_seed, elm_bias_start, elm_bias_end, **kwargs):
        super(S_HNN_SEP, self).__init__()

        self.is_torch_model = False

        # input_dim should be even (half for q, half for p)
        assert input_dim % 2 == 0, "input_dim must be even for separable HNN"
        self.dof = input_dim // 2

        # MLP for Potential energy V(q) - takes only q coordinates
        self.mlp_pot = S_MLP(
            self.dof, hidden_dim, 1, activation,
            resample_duplicates, rcond, random_seed,
            elm_bias_start, elm_bias_end
        )

        # MLP for Kinetic energy T(p) - takes only p coordinates
        self.mlp_kin = S_MLP(
            self.dof, hidden_dim, 1, activation,
            resample_duplicates, rcond, random_seed + 1,
            elm_bias_start, elm_bias_end
        )

    def forward(self, x):
        return BaseModel.forward(self, x)

    # ------------------- energy components -------------------

    def V(self, q: ndarray) -> ndarray:
        """Potential energy V(q) – expects only q (N, dof)."""
        assert isinstance(q, ndarray)
        return self.mlp_pot.forward(q)  # (N, 1)

    def T_energy(self, p: ndarray) -> ndarray:
        """Kinetic energy T(p) – expects only p (N, dof)."""
        assert isinstance(p, ndarray)
        return self.mlp_kin.forward(p)  # (N, 1)

    def potential(self, x: ndarray) -> ndarray:
        """Convenience: V(q) from full state x = [q, p]."""
        assert isinstance(x, ndarray)
        q, _ = np.split(x, 2, axis=1)
        return self.V(q)

    def kinetic(self, x: ndarray) -> ndarray:
        """Convenience: T(p) from full state x = [q, p]."""
        assert isinstance(x, ndarray)
        _, p = np.split(x, 2, axis=1)
        return self.T_energy(p)

    # ------------------- total Hamiltonian -------------------

    def H(self, x: T, create_graph=False) -> T:
        """
        Compute total Hamiltonian H(q,p) = V(q) + T(p)
        """
        assert isinstance(x, np.ndarray)

        q, p = np.split(x, 2, axis=1)
        V = self.V(q)
        T = self.T_energy(p)
        return V + T

    # ------------------- gradients -------------------

    def V_grad(self, q: ndarray, create_graph=False) -> ndarray:
        """
        ∂V/∂q, shape (N, dof), given q only.
        """
        assert isinstance(q, ndarray)

        linear_pot: Any = self.mlp_pot.pipeline[-1]
        grad_last_hidden_V = self.mlp_pot.compute_grad_last_hidden_wrt_input(q)  # (N, dof, M)
        grad_q = grad_last_hidden_V @ linear_pot.weights                        # (N, dof, 1)
        return grad_q.reshape(q.shape)                                          # (N, dof)

    def T_grad(self, p: ndarray, create_graph=False) -> ndarray:
        """
        ∂T/∂p, shape (N, dof), given p only.
        """
        assert isinstance(p, ndarray)

        linear_kin: Any = self.mlp_kin.pipeline[-1]
        grad_last_hidden_T = self.mlp_kin.compute_grad_last_hidden_wrt_input(p)  # (N, dof, M)
        grad_p = grad_last_hidden_T @ linear_kin.weights                         # (N, dof, 1)
        return grad_p.reshape(p.shape)                                           # (N, dof)

    def H_grad(self, x: T, create_graph=False) -> T:
        """
        ∇_x H(q,p) = (∂V/∂q, ∂T/∂p)
        """
        assert isinstance(x, ndarray)

        q, p = np.split(x, 2, axis=1)
        grad_q = self.V_grad(q, create_graph=create_graph)  # (N, dof)
        grad_p = self.T_grad(p, create_graph=create_graph)  # (N, dof)

        # concat along feature dimension → (N, 2*dof)
        grad = np.concatenate([grad_q, grad_p], axis=1)
        return grad

    def dt(self, x: T, create_graph=False) -> T:
        assert isinstance(x, ndarray)

        q, p = np.split(x, 2, axis=1)

        # Compute gradients: dV/dq and dT/dp
        grad_V_q = self.V_grad(q, create_graph=create_graph)  # (N, dof)
        grad_T_p = self.T_grad(p, create_graph=create_graph)  # (N, dof)

        # Hamilton's Equations for separable Hamiltonian:
        # q_dot = dH/dp = dT/dp
        # p_dot = -dH/dq = -dV/dq
        q_dot = grad_T_p
        p_dot = -grad_V_q

        return np.hstack((q_dot, p_dot))

    # ------------------- eval helpers -------------------

    def evaluate_H(self, x, H_true) -> float | None:
        assert isinstance(x, ndarray)
        assert isinstance(H_true, ndarray)

        H_pred = self.H(x)
        assert isinstance(H_pred, ndarray)

        error = l2_error_rel(H_true, H_pred)
        assert isinstance(error, float)
        return error

    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        assert isinstance(x, ndarray)
        assert isinstance(H_grad_true, ndarray)

        H_grad_pred = self.H_grad(x)
        assert isinstance(H_grad_pred, ndarray)

        error = l2_error_rel(H_grad_true, H_grad_pred)
        assert isinstance(error, float)
        return error

    def init_params(self):
        self.mlp_pot.init_params()
        self.mlp_kin.init_params()

    # ------------------- last-layer accessors for sampler -------------------

    def H_last_layer_pot(self) -> Any:
        return self.mlp_pot.pipeline[-1]

    def H_last_layer_kin(self) -> Any:
        return self.mlp_kin.pipeline[-1]

    # Keep this for backwards compatibility if something still calls H_last_layer()
    def H_last_layer(self) -> Any:
        return self.mlp.pipeline[-1]
