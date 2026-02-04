import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

class ThreeBodyHamiltonian(BaseHamiltonian):
    """
    3-body gravitational Hamiltonian in 2D.
    state: [qx1, qx2, qx3, qy1, qy2, qy3, px1, px2, px3, py1, py2, py3]
    """
    def __init__(self, m1=1., m2=1., m3=1, g=1):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.g = g

    def H(self, x):
        x = x.reshape(-1, 12)

        # q: [qx1,qy1,qx2,qy2,qx3,qy3]
        q = x[:, :6]
        qx1, qx2, qx3 = q[:, 0], q[:, 2], q[:, 4]
        qy1, qy2, qy3 = q[:, 1], q[:, 3], q[:, 5]

        # p: [px1,py1,px2,py2,px3,py3]  (p is momentum)
        p = x[:, 6:12]
        px1, px2, px3 = p[:, 0], p[:, 2], p[:, 4]
        py1, py2, py3 = p[:, 1], p[:, 3], p[:, 5]

        # Kinetic energy: p^2/(2m)
        T = 0.5 * (
                (px1 ** 2 + py1 ** 2) / self.m1 +
                (px2 ** 2 + py2 ** 2) / self.m2 +
                (px3 ** 2 + py3 ** 2) / self.m3
        )

        # Potential energy with small epsilon
        def U_pair(qx_i, qy_i, qx_j, qy_j, mi, mj, eps=1e-9):
            dx = qx_i - qx_j
            dy = qy_i - qy_j
            r = np.sqrt(dx ** 2 + dy ** 2) + eps
            return -self.g * mi * mj / r

        U = (U_pair(qx1, qy1, qx2, qy2, self.m1, self.m2) +
             U_pair(qx1, qy1, qx3, qy3, self.m1, self.m3) +
             U_pair(qx2, qy2, qx3, qy3, self.m2, self.m3))

        return (T + U).reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 12)
        q = x[:, :6].reshape(-1, 3, 2)  # (B, 3, 2)
        p = x[:, 6:].reshape(-1, 3, 2)  # (B, 3, 2)

        dq = np.zeros_like(q)
        dp = np.zeros_like(p)

        # dH/dp = p / m
        masses = np.array([self.m1, self.m2, self.m3], dtype=float)
        dp = p / masses[None, :, None]

        # dH/dq = +sum_j G m_i m_j (q_i - q_j) / |q_i - q_j|^3
        eps = 1e-9
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                r_ij = q[:, i, :] - q[:, j, :]
                dist3 = (np.linalg.norm(r_ij, axis=1, keepdims=True) + eps) ** 3
                dq[:, i, :] += self.g * masses[i] * masses[j] * r_ij / dist3

        grad = np.concatenate([dq.reshape(-1, 6), dp.reshape(-1, 6)], axis=1)
        return grad

    def type(self):
        return HamiltonianType.THREE_BODY