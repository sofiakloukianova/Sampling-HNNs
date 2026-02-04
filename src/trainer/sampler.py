from typing import Any
import numpy as np

from model.sampled_network_type import SampledNetworkType
from trainer.param_sampler import ParameterSampler
from .base import BaseTrainer


class Sampler(BaseTrainer):
    sampling_type: SampledNetworkType
    param_sampler: ParameterSampler

    def __init__(self, sampling_type, param_sampler, **kwargs):
        super(Sampler, self).__init__()

        # for ELM, A_PRIORI distribution should be set for parameter sampler, and vice versa
        assert ((sampling_type is SampledNetworkType.ELM and param_sampler is ParameterSampler.A_PRIORI) or
               (sampling_type is not SampledNetworkType.ELM and param_sampler is not ParameterSampler.A_PRIORI))

        self.sampling_type = sampling_type
        self.param_sampler = param_sampler

        ### NEW ###
        self.cond_number = None

    def train(self, model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device, dtype=None, train_H_truths=None):
        super(Sampler, self).train(model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device, dtype, train_H_truths)

        from model.s_mlp import S_MLP
        from model.s_hnn import S_HNN
        from model.s_hnn_seperable import S_HNN_SEP

        # device parameter is ignored in sampling, it is only relevant for traditional network training
        # sampler only utilizes the cpu
        assert isinstance(model, S_MLP) or isinstance(model, S_HNN) or isinstance(model, S_HNN_SEP)

        if isinstance(model, S_MLP):
            # odenet directly outputs time derivatives
            model.pipeline.fit(train_inputs, train_dt_truths)

        if isinstance(model, S_HNN_SEP):
            assert len(model.mlp_pot.pipeline) == 2
            assert len(model.mlp_kin.pipeline) == 2

            q, p = np.split(train_inputs, 2, axis=1)
            q0, p0 = np.split(train_input_x_0, 2, axis=1)
            q_dot_truths, p_dot_truths = np.split(train_dt_truths, 2, axis=1)

            V0_truth = train_input_x_0_H_truth  # used as function value constraint for V

            # ------------- configure dense layers for POT and KIN -------------

            dense_pot: Any = model.mlp_pot.pipeline[0]
            dense_kin: Any = model.mlp_kin.pipeline[0]

            # Both must be sampling layers
            assert dense_pot.sample_uniformly
            assert dense_kin.sample_uniformly

            # set parameter sampler
            dense_pot.parameter_sampler = self.param_sampler.value
            dense_pot.__post_init__()
            dense_kin.parameter_sampler = self.param_sampler.value
            dense_kin.__post_init__()

            # Initial sampling phase
            if self.sampling_type is SampledNetworkType.SWIM:
                assert train_H_truths is not None
                dense_pot.sample_uniformly = False
                dense_kin.sample_uniformly = False
                dense_pot.fit(q, train_H_truths)
                dense_kin.fit(p, train_H_truths)
            else:
                # unsupervised sampling
                dense_pot.fit(q)
                dense_kin.fit(p)

            grad_last_hidden_q = model.mlp_pot.compute_grad_last_hidden_wrt_input(q)
            last_hidden_q0 = dense_pot.transform(q0)

            c_pot = self.__fit_separable_linear_layer(
                grad_last_hidden=grad_last_hidden_q,
                last_hidden_out_x_0=last_hidden_q0,
                target_grad=-p_dot_truths,  # dp/dt = -∂V/∂q
                target_value=V0_truth,  # V(q0) ≈ H(x0)
                rcond=model.mlp_pot.rcond,
            ).reshape(-1, 1)

            pot_layer: Any = model.H_last_layer_pot()
            pot_layer.weights = c_pot[:-1].reshape((-1, 1))
            pot_layer.biases = c_pot[-1].reshape((1, 1))
            pot_layer.layer_width = pot_layer.weights.shape[1]
            pot_layer.n_parameters = pot_layer.weights.size + pot_layer.biases.size

            grad_last_hidden_p = model.mlp_kin.compute_grad_last_hidden_wrt_input(p)
            last_hidden_p0 = dense_kin.transform(p0)

            c_kin = self.__fit_separable_linear_layer(
                grad_last_hidden=grad_last_hidden_p,
                last_hidden_out_x_0=last_hidden_p0,
                target_grad=q_dot_truths,  # dq/dt = ∂T/∂p
                target_value=0.0,  # T(p0) ≈ 0 ⇒ H(x0) ≈ V0_truth
                rcond=model.mlp_kin.rcond,
            ).reshape(-1, 1)

            kin_layer: Any = model.H_last_layer_kin()
            kin_layer.weights = c_kin[:-1].reshape((-1, 1))
            kin_layer.biases = c_kin[-1].reshape((1, 1))
            kin_layer.layer_width = kin_layer.weights.shape[1]
            kin_layer.n_parameters = kin_layer.weights.size + kin_layer.biases.size

            if self.sampling_type is SampledNetworkType.A_SWIM:
                # approximate Hamiltonian on training set
                H_pred = model.H(train_inputs)

                # resample with approximate values (supervised)
                dense_pot.sample_uniformly = False
                dense_kin.sample_uniformly = False

                dense_pot.fit(q, H_pred)
                dense_kin.fit(p, H_pred)

                # recompute gradients
                grad_last_hidden_q = model.mlp_pot.compute_grad_last_hidden_wrt_input(q)
                last_hidden_q0 = dense_pot.transform(q0)

                grad_last_hidden_p = model.mlp_kin.compute_grad_last_hidden_wrt_input(p)
                last_hidden_p0 = dense_kin.transform(p0)

                # solve again for V
                c_pot = self.__fit_separable_linear_layer(
                    grad_last_hidden=grad_last_hidden_q,
                    last_hidden_out_x_0=last_hidden_q0,
                    target_grad=-p_dot_truths,
                    target_value=V0_truth,
                    rcond=model.mlp_pot.rcond,
                ).reshape(-1, 1)

                pot_layer.weights = c_pot[:-1].reshape((-1, 1))
                pot_layer.biases = c_pot[-1].reshape((1, 1))

                # solve again for T
                c_kin = self.__fit_separable_linear_layer(
                    grad_last_hidden=grad_last_hidden_p,
                    last_hidden_out_x_0=last_hidden_p0,
                    target_grad=q_dot_truths,
                    target_value=0.0,
                    rcond=model.mlp_kin.rcond,
                ).reshape(-1, 1)

                kin_layer.weights = c_kin[:-1].reshape((-1, 1))
                kin_layer.biases = c_kin[-1].reshape((1, 1))


        elif isinstance(model, S_HNN):
            assert len(model.mlp.pipeline) == 2 # only support shallow networks for now

            # set the parameter sampler in the dense layer
            dense_layer: Any = model.mlp.pipeline[0]
            assert dense_layer.sample_uniformly
            dense_layer.parameter_sampler = self.param_sampler.value
            dense_layer.__post_init__() # init is required to use the actual param sampler, not the string

            # sample hidden layer weights (unsupervised if not SWIM, and with uniform sampling of the inputs)
            if self.sampling_type is SampledNetworkType.SWIM:
                assert train_H_truths is not None
                dense_layer.sample_uniformly = False
                dense_layer.fit(train_inputs, train_H_truths)
            else:
                dense_layer.fit(train_inputs)

            grad_last_hidden = model.mlp.compute_grad_last_hidden_wrt_input(train_inputs)
            last_hidden_out_x_0 = dense_layer.transform(train_input_x_0)

            # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
            c = self.__fit_HNN_linear_layer(grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_H_truth, rcond=model.mlp.rcond).reshape(-1,1)

            linear_layer: Any = model.H_last_layer()
            linear_layer.weights = c[:-1].reshape((-1,1))
            linear_layer.biases = c[-1].reshape((1,1))
            linear_layer.layer_width = linear_layer.weights.shape[1]
            linear_layer.n_parameters = linear_layer.weights.size + linear_layer.biases.size

            # case: A-SWIM
            if self.sampling_type is SampledNetworkType.A_SWIM:
                # approximate the Hamiltonian values (target function values) which we need in other sampling methods
                train_preds = model.H(train_inputs)

                # resample with approximate values, for this we disable uniform sampling of inputs
                dense_layer.sample_uniformly = False

                # sample hidden layer weights (supervised with approximate values)
                dense_layer.fit(train_inputs, train_preds)

                grad_last_hidden = model.mlp.compute_grad_last_hidden_wrt_input(train_inputs)
                last_hidden_out_x_0 = dense_layer.transform(train_input_x_0)

                # solve again to fit the linear layer
                c = self.__fit_HNN_linear_layer(grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_H_truth, rcond=model.mlp.rcond).reshape(-1,1)

                linear_layer.weights = c[:-1].reshape((-1,1))
                linear_layer.biases = c[-1].reshape((1,1))

    def __fit_HNN_linear_layer(self, grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_truth, rcond):
        """
        Fits the last layer of the model by solving least squares,
        builds the matrix A and vector b and solves the linear equation for x (weights)

        @param grad_last_hidden         : gradients of hidden layer output w.r.t. input (K,D,M)
        @param last_hidden_out_x_0      : hidden layer output of x0 (1,M)
        @param y_train_derivs_true      : derivatives of target function w.r.t. X (K*D)
        @param train_input_x_0_truth    : true function value at input x0
        @param rcond                    : how approximately to solve the least squares

        @return c                       : solved x (weights of the final linear layer)
        """
        grad_last_hidden_q_part, grad_last_hidden_p_part = np.split(grad_last_hidden, 2, axis=1)

        (num_points, dof, last_hidden_width) = grad_last_hidden_q_part.shape
        assert num_points == grad_last_hidden_p_part.shape[0]
        assert dof == grad_last_hidden_p_part.shape[1]
        assert last_hidden_width == grad_last_hidden_p_part.shape[2]

        grad_last_hidden_q_part = grad_last_hidden_q_part.reshape(num_points*dof, last_hidden_width)
        grad_last_hidden_p_part = grad_last_hidden_p_part.reshape(num_points*dof, last_hidden_width)

        # for dof>1 we use reshape
        # grad_last_hidden_q_part = np.squeeze(grad_last_hidden_q_part, axis=1)
        # grad_last_hidden_p_part = np.squeeze(grad_last_hidden_p_part, axis=1)

        # Hamilton's Equations
        A = np.concatenate(( grad_last_hidden_p_part, -grad_last_hidden_q_part ), axis=0)
        A = np.concatenate(( A, last_hidden_out_x_0 ), axis=0)
        A = np.column_stack((A, np.concatenate(( np.zeros(A.shape[0] - 1), np.ones(1) ), axis=0) )) # for the bias term

        q_dot_truths, p_dot_truths = np.split(train_dt_truths, 2, axis=1)
        b = np.concatenate((
            q_dot_truths.ravel(),
            p_dot_truths.ravel(),
            train_input_x_0_truth.ravel(),
        ))

        # (ND + 1, M + 1)
        ### NEW ###
        cond_A = np.linalg.cond(A)
        self.cond_number = cond_A
        print(f"Condition number of A: {cond_A}")

        c = np.linalg.lstsq(A, b, rcond=rcond)[0]


        return c.reshape(-1, 1) # final shape (M+1, 1) == [weights, bias] of shapes (M,1) and (1,1)

    def __fit_separable_linear_layer(
            self,
            grad_last_hidden,
            last_hidden_out_x_0,
            target_grad,
            target_value,
            rcond: float,
    ) :
        """
        Fits last linear layer for a shallow network:

            y(x) = w^T h(x) + b

        using gradient information:

            ∂y/∂x ≈ target_grad

        and one function-value constraint:

            y(x0) ≈ target_value
        """

        # grad_last_hidden: (K, D, M)
        num_points, dof, last_hidden_width = grad_last_hidden.shape

        # flatten gradient rows
        A = grad_last_hidden.reshape(num_points * dof, last_hidden_width)
        b = target_grad.reshape(num_points * dof)

        # add row for function value at x0
        A = np.vstack((A, last_hidden_out_x_0))  # (ND+1, M)
        target_value = np.asarray(target_value).reshape(1)
        b = np.concatenate((b, target_value))  # (ND+1,)

        # add bias column
        bias_col = np.concatenate((np.zeros(A.shape[0] - 1), np.ones(1)))
        A = np.column_stack((A, bias_col))  # (ND+1, M+1)

        # condition number logging
        cond_A = np.linalg.cond(A)
        self.cond_number = cond_A
        print(f"[SEPARABLE] Condition number of A: {cond_A}")

        c = np.linalg.lstsq(A, b, rcond=rcond)[0]
        return c.reshape(-1, 1)