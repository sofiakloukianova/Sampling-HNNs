import numpy as np

from .type import HamiltonianType
from .spring import Spring
from .single_pendulum import SinglePendulum
from .lotka_volterra import LotkaVolterra
from .double_pendulum import DoublePendulum
from .henon_heiles import HenonHeiles
from .two_body import TwoBodyHamiltonian
from .three_body import ThreeBodyHamiltonian


class Hamiltonian():
    @staticmethod
    def new(target: HamiltonianType, **kwargs):
        """
        Returns the target Hamiltonian DOF, domain limits, and object.

        @param target: target Hamiltonian system type

        @return input_dim, domain_limits (q and p limits), target_hamiltonian
        """
        match target:
            case HamiltonianType.SPRING:
                q_lims = [ [-1., 1.] ]
                p_lims = [ [-1., 1.] ]

                return 2, (q_lims, p_lims), Spring()

            case HamiltonianType.SINGLE_PENDULUM:
                q_lims = [ [-np.pi, np.pi ] ]
                p_lims = [ [-1, 1] ]

                return 2, (q_lims, p_lims), SinglePendulum()

            case HamiltonianType.DOUBLE_PENDULUM:
                q_lims = [ [-np.pi, np.pi], [-np.pi, np.pi] ]
                p_lims = [ [-1., 1.], [-1., 1.] ]

                return 4, (q_lims, p_lims), DoublePendulum()

            case HamiltonianType.HENON_HEILES:
                q_lims = [ [ -5, 5 ], [-5, 5] ]
                p_lims = [ [ -5, 5 ], [-5, 5] ]

                return 4, (q_lims, p_lims), HenonHeiles()

            case HamiltonianType.TWO_BODY:
                # 2 bodies x 2D positions + 2D momenta = 8-dimensional input
                q_lims = [[-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.]]  # [qx1,qx2,qy1,qy2]
                p_lims = [[-2., 2.], [-2., 2.], [-2., 2.], [-2., 2.]]  # [px1,px2,py1,py2]

                input_dim = 8  # 4 DOF × 2 (q & p)
                return input_dim, (q_lims, p_lims), TwoBodyHamiltonian()

            case HamiltonianType.THREE_BODY:
                # 3 bodies x (x,y) positions + (px,py) = 12-dimensional state
                q_lims = [[-3., 3.]] * 6  # qx1,qx2,qx3,qy1,qy2,qy3
                p_lims = [[-3., 3.]] * 6  # px1,px2,px3,py1,py2,py3
                input_dim = 12
                return input_dim, (q_lims, p_lims), ThreeBodyHamiltonian()

            case _:
                raise NotImplementedError(f"Hamiltonian type {type} is not implemented yet.")
