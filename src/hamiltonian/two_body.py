import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

class TwoBodyHamiltonian(BaseHamiltonian):
    """
    2-body problem in 2D (gravitational)
    state: [qx1, qx2, qy1, qy2, px1, px2, py1, py2]        """
    def __init__(self, m1=1., m2=1., g=1):
        self.m1 = m1
        self.m2 = m2
        self.g = g

    def H(self, x):
        x = x.reshape(-1, 8)
        qx1, qx2, qy1, qy2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        px1, px2, py1, py2 = x[:, 4], x[:, 5], x[:, 6], x[:, 7]

        # Kinetic energy
        T = 0.5 * self.m1 * (px1 ** 2 + py1 ** 2) + 0.5 * self.m2 * (px2 ** 2 + py2 ** 2)

        # Potential energy
        dx = qx1 - qx2
        dy = qy1 - qy2
        r = np.sqrt(dx ** 2 + dy ** 2)
        U = -self.g * self.m1 * self.m2 / r

        return (T + U).reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 8)
        qx1, qx2, qy1, qy2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        px1, px2, py1, py2 = x[:, 4], x[:, 5], x[:, 6], x[:, 7]

        dx = qx1 - qx2
        dy = qy1 - qy2
        r3 = (dx ** 2 + dy ** 2) ** 1.5

        # dU/dq
        dU_dx1 = self.g * self.m1 * self.m2 * dx / r3
        dU_dy1 = self.g * self.m1 * self.m2 * dy / r3
        dU_dx2 = -dU_dx1
        dU_dy2 = -dU_dy1

        # ∂H/∂q
        dq1x = dU_dx1
        dq1y = dU_dy1
        dq2x = dU_dx2
        dq2y = dU_dy2

        # ∂H/∂p
        dp1x = px1 / self.m1
        dp1y = py1 / self.m1
        dp2x = px2 / self.m2
        dp2y = py2 / self.m2

        grad = np.stack([dq1x, dq2x, dq1y, dq2y, dp1x, dp2x, dp1y, dp2y], axis=1)
        return grad

    def type(self):
        return HamiltonianType.TWO_BODY