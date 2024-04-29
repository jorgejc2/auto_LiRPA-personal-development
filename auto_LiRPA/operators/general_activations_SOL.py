from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function
from collections import OrderedDict
from .base import *
from .clampmult import multiply_by_A_signs
from .activation_base import BoundActivation, BoundOptimizableActivation
from .solver_utils import grb
from ..utils import unravel_index, prod
from time import time

from ..SOL.functions import *
from ..SOL import OptimalLinearBounder

general_function = torch_tanh
general_grad_function = torch_tanh_grad

def activation_to_numpy(to_numpy=True):

    def decorator(func):
        def wrapper(*args, **kwargs):
            assert len(args) == 1, "Activation function should only take one argument"
            x = args[0]

            # convert input to Tensor if it is not already
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if isinstance(x, list):
                x = torch.Tensor(x)

            # Call the original PyTorch function
            result = func(x, **kwargs)

            # convert output back to numpy if desired
            if to_numpy:
                result = result.detach().cpu().numpy()


            # Return the result
            return result

        return wrapper

    return decorator

@activation_to_numpy(True)
def numpy_general_activation(x):
    return general_function(x)

@activation_to_numpy(True)
def numpy_general_grad_activation(x):
    return general_grad_function(x)

class BoundGeneralActivationsSOL(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0,
                 options=None, activation=('tanh', None, None, None), precompute=True):
        super().__init__(attr, inputs, output_index, options)
        self.options = options
        self.ibp_intermediate = True
        self.splittable = True
        self.relu_options = options.get('relu', 'adaptive')  # FIXME: use better names.
        self.use_sparse_spec_alpha = options.get('sparse_spec_alpha', False)
        self.use_sparse_features_alpha = options.get('sparse_features_alpha', False)
        self.alpha_lookup_idx = self.alpha_indices = None
        self.beta = self.masked_beta = self.sparse_betas = None
        self.split_beta_used = False
        self.history_beta_used = False
        self.flattened_nodes = None
        self.patch_size = {}
        self.cut_used = False
        self.cut_module = None

        # additional parameters that's need to be known
        self.alpha_size = 4
        # Alpha dimension is (4, output_shape, batch, *shape)

        # TODO: Figure out how to get this information for our general activation function
        self.activation_name = activation[0]
        # save the Torch activation functions and their NumPy counterparts
        self.torch_general_function = general_function
        self.torch_general_grad_function = general_grad_function
        self.numpy_general_function = numpy_general_activation
        self.numpy_general_grad_function = numpy_general_grad_activation

        self.L1 = 5
        self.L2 = 5

        # TODO: Notice that BoundTanh has a precompute_relaxation method. This may be where we calculate the SOL bounds
        # if precompute:
        #     self.precompute_relaxation(self.activation_forward, self.activation_backward)

    def init_opt_parameters(self, start_nodes):
        """

        :param start_nodes:
        :return:
        """
        # TODO: Fix this code to have more alphas. We need 4 sets of alphas whereas this code only has 2 sets since
        #  it was copied from BoundReLU
        ref = self.inputs[0].lower # a reference variable for getting the shape
        batch_size = ref.size(0)
        self.alpha = OrderedDict()
        self.alpha_lookup_idx = OrderedDict()  # For alpha with sparse spec dimention.
        self.alpha_indices = None  # indices of non-zero alphas.
        verbosity = self.options.get('verbosity', 0)

        alpha_shape = self.shape  # Full alpha.
        # Now we start to create alphas for all start nodes.
        for start_node in start_nodes:
            ns, output_shape, unstable_idx = start_node[:3]
            if isinstance(output_shape, (list, tuple)):
                if len(output_shape) > 1:
                    size_s = prod(output_shape)  # Conv layers.
                else:
                    size_s = output_shape[0]
            else:
                size_s = output_shape

            # alpha shape is (2, spec, batch, this_layer_shape). "this_layer_shape" may still be sparse.
            self.alpha[ns] = torch.empty([self.alpha_size, size_s, batch_size, *alpha_shape],
                                         dtype=torch.float, device=ref.device, requires_grad=True)
            # initialize alphas to be the SOL slopes
            self.alpha[ns].data[0, ...].copy_(self.linear_bounds[0, :, 0, 0])
            self.alpha[ns].data[2, ...].copy_(self.linear_bounds[0, :, 0, 0])
            self.alpha[ns].data[1, ...].copy_(self.linear_bounds[0, :, 1, 0])
            self.alpha[ns].data[3, ...].copy_(self.linear_bounds[0, :, 1, 0])
            if verbosity > 0:
                print(f'layer {self.name} start_node {ns} using full alpha {list(self.alpha[ns].size())} with unstable '
                      f'size {None} total_size {size_s} output_shape {output_shape}')
            # alpha_lookup_idx can be used for checking if sparse alpha is used or not.
            self.alpha_lookup_idx[ns] = None

    def select_alpha_by_idx(self, last_lA, last_uA, unstable_idx, start_node, alpha_lookup_idx):
        """
        Potentially need not be implemented as we are not adding sparse alpha capabilites
        :param last_lA:
        :param last_uA:
        :param unstable_idx:
        :param start_node:
        :param alpha_lookup_idx:
        :return:
        """
        raise NotImplementedError(f'select_alpha_by_idx is not implemented for {self}.')

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None,
                       unstable_idx=None, reduce_bias=True, **kwargs):
        """
        start_node: the name of the layer where the backward bound propagation starts.
                    Can be the output layer or an intermediate layer.
        unstable_idx: indices for the unstable neurons, whose bounds need to be computed.
                      Either be a tuple (for patches) or a 1-D tensor.
        """

        def _bound_oneside_using_SOL_relaxations(last_A, sign=-1):
            if last_A is None:
                return None, 0
            if sign == -1:
                w_pos, b_pos, w_neg, b_neg = (
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0),
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0))
            else:
                w_pos, b_pos, w_neg, b_neg = (
                    self.uw.unsqueeze(0), self.ub.unsqueeze(0),
                    self.lw.unsqueeze(0), self.lb.unsqueeze(0))
            w_pos = maybe_unfold_patches(w_pos, last_A)
            w_neg = maybe_unfold_patches(w_neg, last_A)
            b_pos = maybe_unfold_patches(b_pos, last_A)
            b_neg = maybe_unfold_patches(b_neg, last_A)
            if self.batch_dim == 0:
                _A, _bias = multiply_by_A_signs(
                    last_A, w_pos, w_neg, b_pos, b_neg, reduce_bias=reduce_bias)
            elif self.batch_dim == -1:
                # FIXME: why this is different from above?
                assert reduce_bias
                mask = torch.gt(last_A, 0.).to(torch.float)
                _A = last_A * (mask * w_pos.unsqueeze(1) +
                               (1 - mask) * w_neg.unsqueeze(1))
                _bias = last_A * (mask * b_pos.unsqueeze(1) +
                                  (1 - mask) * b_neg.unsqueeze(1))
                if _bias.ndim > 2:
                    _bias = torch.sum(_bias, dim=list(range(2, _bias.ndim)))
            else:
                raise NotImplementedError

            return _A, _bias

        def _bound_oneside_using_optimized_relaxations(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0
            # Obtain the new linear relaxation coefficients based on the signs in last_A.
            same_slope = True if self.relu_options == "same-slope" else False
            _A, _bias = multiply_by_A_signs(
                last_A, d_pos, d_neg, b_pos, b_neg, reduce_bias=reduce_bias, same_slope=same_slope)
            if isinstance(last_A, Patches):
                # Save the patch size, which will be used in init_alpha() to determine the number of optimizable parameters.
                A_prod = _A.patches
                if start_node is not None:
                    if last_A.unstable_idx is not None:
                        # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                        self.patch_size[start_node.name] = [
                            last_A.output_shape[1], A_prod.size(1),
                            last_A.output_shape[2], last_A.output_shape[3],
                            A_prod.size(-3), A_prod.size(-2), A_prod.size(-1)]
                    else:
                        # Regular patches.
                        self.patch_size[start_node.name] = A_prod.size()
            return _A, _bias


        self._start = start_node.name
        if self.opt_stage not in ['opt', 'reuse']:
            # we must be in the init stage, therefore we perform SOL on the elements of the activation function to
            # get our initial bounds

            self._get_optimal_SOL_bounds(x)

            # add linear relaxations using SOL bounds
            self.add_linear_relaxation(mask=None, type='upper',
                                       k=self.linear_bounds[0, :, 0, 0],
                                       x0=torch.zeros_like(x.upper, requires_grad=True),
                                       y0=self.linear_bounds[0, :, 0, 1])
            self.add_linear_relaxation(mask=None, type='lower',
                                       k=self.linear_bounds[0, :, 1, 0],
                                       x0=torch.zeros_like(x.upper, requires_grad=True),
                                       y0=self.linear_bounds[0, :, 1, 1])



            lA, lbias = _bound_oneside_using_SOL_relaxations(last_lA, sign=-1)
            uA, ubias = _bound_oneside_using_SOL_relaxations(last_uA, sign=+1)

            return [(lA, uA)], lbias, ubias


        assert self.batch_dim == 0

        # at this point, we are performing our optimizations
        # we do not need to relax the bounds, simply clip alphas

        ub_upper_d, ub_upper_b, lb_upper_d, lb_upper_b, ub_lower_d, ub_lower_b, lb_lower_d, lb_lower_b = self._get_relaxed_bounds(
            start_node)

        uA, ubias = _bound_oneside_using_optimized_relaxations(
            last_uA, ub_upper_d,
            ub_lower_d,
            ub_upper_b, ub_lower_b)
        lA, lbias = _bound_oneside_using_optimized_relaxations(
            last_lA, lb_lower_d,
            lb_upper_d,
            lb_lower_b, lb_upper_b)

        return [(lA, uA)], lbias, ubias


    def dump_optimized_params(self):
        ret = {'alpha': self.alpha}
        if self.use_sparse_spec_alpha:
            ret['alpha_lookup_idx'] = self.alpha_lookup_idx
        if self.use_sparse_features_alpha:
            ret['alpha_indices'] = self.alpha_indices
        return ret

    def restore_optimized_params(self, alpha):
        self.alpha = alpha['alpha']
        if self.use_sparse_spec_alpha:
            self.alpha_lookup_idx = alpha['alpha_lookup_idx']
        if self.use_sparse_features_alpha:
            self.alpha_indices = alpha['alpha_indices']

    def get_unstable_idx(self):
        """

        :return:
        """
        # TODO: This logic of for ReLU. In our case, a node is unstable (alpha can be optimized) if the interval is
        #  in a convex or concave region. It may be best to precompute these regions rather than using the second
        #  derivative for faster checking
        self.alpha_indices = torch.logical_and(
            self.inputs[0].lower < 0, self.inputs[0].upper > 0).any(dim=0).nonzero(as_tuple=True)

    def clip_alpha(self):
        # TODO: We will need to create a dictionary that gets the range of sounds alphas for each activation element
        #  FIXME: We should delete this function and clip alpha ourselves. The parent class has clip_alpha which does
        #   nothing so removing this method is fine
        pass

    def forward(self, x):
        # TODO: We will need to call our general activation function here instead of ReLU
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]

        return self.torch_general_function(x)


    def _backward_relaxation(self, last_lA, last_uA, x, start_node, unstable_idx):
        """
        This is the main function called by bound_backward which will calculate the bounds on our activation elements
        and backward propagate our bounds.
        :param last_lA:
        :param last_uA:
        :param x:
        :param start_node:
        :param unstable_idx:
        :return:
        """
        # TODO: Correct this to be tailored to our general activation function
        # Usage of output constraints requires access to bounds of the previous iteration
        # (see _clear_and_set_new)
        if x is not None:
            apply_output_constraints_to = self.options['optimize_bound_args']['apply_output_constraints_to']
            lower = x.lower
            upper = x.upper
        else:
            lower = self.lower
            upper = self.upper

        # Upper bound slope and intercept according to CROWN relaxation.
        upper_d, upper_b = self._relu_upper_bound(lower, upper, self.leaky_alpha)

        flag_expand = False

        ub_lower_d = lb_lower_d = None
        ub_upper_d = lb_upper_d = None
        ub_upper_b = lb_upper_b = None

        lower_b = None  # ReLU does not have lower bound intercept (=0).
        alpha_lookup_idx = None  # For sparse-spec alpha.
        if self.opt_stage in ['opt', 'reuse']:
            # Alpha-CROWN.
            lower_d = None
            selected_alpha, alpha_lookup_idx = self.select_alpha_by_idx(last_lA, last_uA,
                                                                        unstable_idx, start_node, alpha_lookup_idx)
            # The first dimension is lower/upper intermediate bound.
            if last_lA is not None:
                lb_lower_d = selected_alpha[0]
            if last_uA is not None:
                ub_lower_d = selected_alpha[1]

            if self.alpha_indices is not None:
                # Sparse alpha on the hwc dimension. We store slopes for unstable neurons in this layer only.
                # Recover to full alpha first.
                sparse_alpha_shape = lb_lower_d.shape if lb_lower_d is not None else ub_lower_d.shape
                full_alpha_shape = sparse_alpha_shape[:-1] + self.shape
                if lb_lower_d is not None:
                    lb_lower_d = self.reconstruct_full_alpha(
                        lb_lower_d, full_alpha_shape, self.alpha_indices)
                if ub_lower_d is not None:
                    ub_lower_d = self.reconstruct_full_alpha(
                        ub_lower_d, full_alpha_shape, self.alpha_indices)

            lb_lower_d, ub_lower_d, zero_coeffs = self._relu_mask_alpha(lower, upper, lb_lower_d, ub_lower_d)
            self.zero_backward_coeffs_l = self.zero_backward_coeffs_u = zero_coeffs
            flag_expand = True  # we already have the spec dimension.

            if self.relu_options == "same-slope":
                # same-slope with optimized lower_d
                # We force upper_d to be the same as lower_d, and compute the corresponding upper_b
                lb_upper_d, lb_upper_b, ub_upper_d, ub_upper_b = self._relu_upper_opt_same_slope(lb_lower_d, ub_lower_d,
                                                                                                 upper_d, lower, upper)

        else:
            # FIXME: the shape can be incorrect if unstable_idx is not None.
            # This will cause problem if some ReLU layers are optimized, some are not.
            lower_d = self._relu_lower_bound_init(upper_d)

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        upper_d = upper_d.unsqueeze(0)
        upper_b = upper_b.unsqueeze(0)
        if not flag_expand:
            # FIXME: The following lines seem unused since
            # flag_expand must be true when self.optstage in ['opt, 'reuse']
            if self.opt_stage in ['opt', 'reuse']:
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None

                if self.relu_options == "same-slope":
                    upper_d = None
                    lb_upper_d = lb_upper_d.unsqueeze(0) if last_lA is not None else None
                    lb_upper_b = lb_upper_b.unsqueeze(0) if last_lA is not None else None
                    ub_upper_d = ub_upper_d.unsqueeze(0) if last_uA is not None else None
                    ub_upper_b = ub_upper_b.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)

        if self.opt_stage in ['opt', 'reuse'] and self.relu_options == "same-slope":
            # Remove upper_d and upper_b to avoid confusion later
            upper_d = None
            upper_b = None

        return (upper_d, upper_b, lower_d, lower_b, lb_lower_d, ub_lower_d,
                lb_upper_d, ub_upper_d, lb_upper_b, ub_upper_b, alpha_lookup_idx)

    def _mask_bounds(self):

        # TODO: Figure out how to compute the masks


        # clip the alphas to their proper ranges


        pass

    def _get_optimal_SOL_bounds(self, x: torch.Tensor, use_tangent=False) -> None:

        verbosity = self.options.get('verbosity', 0)

        # init relaxation arrays
        self.init_linear_relaxation(x)

        # add code to initialize bisection algorithm
        self.linear_bounds = torch.zeros((*x.upper.shape, 2, 2), dtype=x.upper.dtype, device=x.upper.device)
        bounder = OptimalLinearBounder(
            self.numpy_general_function, self.numpy_general_grad_function, self.L1, self.L2, eps=1e-3,
            initial_npoints=10
        )
        Upper = x.upper.cpu().detach()
        Lower = x.lower.cpu().detach()
        Upper = Upper.unsqueeze(2)
        Lower = Lower.unsqueeze(2)
        boundaries = torch.cat((Lower, Upper), dim=2)
        boundaries = boundaries.squeeze().numpy()
        start = time()
        upper_branched_points = np.zeros(len(boundaries))
        lower_branched_points = np.zeros(len(boundaries))
        for i, boundary in enumerate(boundaries):
            boundary = np.expand_dims(boundary, axis=0)
            lower_bound, lower_branched_point, upper_bound, upper_branched_point = bounder.find_optimal_bounds(boundary,
                                                                                                               verbosity=verbosity,
                                                                                                               get_most_branched=not use_tangent)
            if not use_tangent:
                upper_branched_points[i] = upper_branched_point
                lower_branched_points[i] = lower_branched_point
            self.linear_bounds[0, i, 0, :] = torch.from_numpy(upper_bound).to(self.linear_bounds.device)
            self.linear_bounds[0, i, 1, :] = torch.from_numpy(lower_bound).to(self.linear_bounds.device)

        # this is for the upper bounds
        if use_tangent:
            ub_l_alphas_m, ub_l_alphas_b = self._get_tangent_line(self.torch_general_function,
                                                                  self.torch_general_grad_function, x.lower)
            ub_r_alphas_m, ub_r_alphas_b = self._get_tangent_line(self.torch_general_function,
                                                                  self.torch_general_grad_function, x.upper)
        else:
            ub_l_alphas_m = torch.zeros_like(x.lower)
            ub_l_alphas_b = torch.zeros_like(x.lower)
            ub_r_alphas_m = torch.zeros_like(x.lower)
            ub_r_alphas_b = torch.zeros_like(x.lower)
            for i, boundary in enumerate(boundaries):
                upper_branched_point = upper_branched_points[i]
                _, _, (ub_l_alphas_m[0, i], ub_l_alphas_b[0, i]), _ = bounder.find_optimal_bounds(
                    np.array([[boundary[0], upper_branched_point]]), verbosity=verbosity, get_most_branched=False,
                    lower=False)
                _, _, (ub_r_alphas_m[0, i], ub_r_alphas_b[0, i]), _ = bounder.find_optimal_bounds(
                    np.array([[upper_branched_point, boundary[1]]]), verbosity=verbosity, get_most_branched=False,
                    lower=False)


        self.ub_min_alphas = torch.min(ub_l_alphas_m, ub_r_alphas_m).squeeze()
        self.ub_max_alphas = torch.max(ub_l_alphas_m, ub_r_alphas_m).squeeze()
        self.ub_l_intersection_points = self._get_intersecting_points(self.linear_bounds[0, :, 0, 0], self.linear_bounds[0, :, 0, 1],
                                          ub_l_alphas_m.squeeze(), ub_l_alphas_b.squeeze()).permute(1, 0)
        self.ub_r_intersection_points = self._get_intersecting_points(self.linear_bounds[0, :, 1, 0], self.linear_bounds[0, :, 1, 1],
                                          ub_r_alphas_m.squeeze(), ub_r_alphas_b.squeeze()).permute(1, 0)

        # this is for the lower bounds
        if use_tangent:
            lb_l_alphas_m, lb_l_alphas_b = self._get_tangent_line(self.torch_general_function,
                                                                  self.torch_general_grad_function, x.lower)
            lb_r_alphas_m, lb_r_alphas_b = self._get_tangent_line(self.torch_general_function,
                                                                  self.torch_general_grad_function, x.upper)
        else:
            lb_l_alphas_m = torch.zeros_like(x.lower)
            lb_l_alphas_b = torch.zeros_like(x.lower)
            lb_r_alphas_m = torch.zeros_like(x.lower)
            lb_r_alphas_b = torch.zeros_like(x.lower)
            for i, boundary in enumerate(boundaries):
                lower_branched_point = lower_branched_points[i]
                (lb_l_alphas_m[0, i], lb_l_alphas_b[0, i]), _, _, _ = bounder.find_optimal_bounds(
                    np.array([[boundary[0], lower_branched_point]]), verbosity=verbosity, get_most_branched=False,
                    upper=False)
                (lb_r_alphas_m[0, i], lb_r_alphas_b[0, i]), _, _, _ = bounder.find_optimal_bounds(
                    np.array([[lower_branched_point, boundary[1]]]), verbosity=verbosity, get_most_branched=False,
                    upper=False)

        self.lb_min_alphas = torch.min(lb_l_alphas_m, lb_r_alphas_m).squeeze()
        self.lb_max_alphas = torch.max(lb_l_alphas_m, lb_r_alphas_m).squeeze()
        self.lb_l_intersection_points = self._get_intersecting_points(self.linear_bounds[0, :, 0, 0], self.linear_bounds[0, :, 0, 1],
                                          lb_l_alphas_m.squeeze(), lb_l_alphas_b.squeeze()).permute(1, 0)
        self.lb_r_intersection_points = self._get_intersecting_points(self.linear_bounds[0, :, 1, 0], self.linear_bounds[0, :, 1, 1],
                                          lb_r_alphas_m.squeeze(), lb_r_alphas_b.squeeze()).permute(1, 0)

        # check which alphas are optimizable
        self.alpha_is_optimizable = torch.zeros(self.alpha_size, *self.shape, dtype=torch.float)
        self.alpha_is_optimizable[0] = (ub_r_alphas_m < self.linear_bounds[0, :, 0, 0] < ub_l_alphas_m).to(dtype=torch.float)
        self.alpha_is_optimizable[2] = (lb_r_alphas_m < self.linear_bounds[0, :, 0, 0] < lb_l_alphas_m).to(dtype=torch.float)
        self.alpha_is_optimizable[1] = (ub_l_alphas_m < self.linear_bounds[0, :, 1, 0] < ub_r_alphas_m).to(dtype=torch.float)
        self.alpha_is_optimizable[3] = (lb_l_alphas_m < self.linear_bounds[0, :, 1, 0] < lb_r_alphas_m).to(dtype=torch.float)

        end = time()
        if verbosity: print(f"Took {end - start} seconds to curate {len(boundaries)} upper and lower bounds")

    def _get_relaxed_bounds(self, start_node: str) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the relaxed bounds using alphas
        :param start_node:
        :return:
        """
        node_alpha = self.alpha[start_node.name]
        ub_upper_alphas, ub_lower_alphas, lb_upper_alphas, lb_lower_alphas = node_alpha[0, ...], node_alpha[1, ...], \
        node_alpha[2, ...], node_alpha[3, ...]

        # get the limits on alpha with appropriate dimensions
        ub_min_alphas = self.ub_min_alphas.squeeze().expand_as(ub_upper_alphas)
        ub_max_alphas = self.ub_max_alphas.squeeze().expand_as(ub_upper_alphas)
        lb_min_alphas = self.lb_min_alphas.squeeze().expand_as(ub_upper_alphas)
        lb_max_alphas = self.lb_max_alphas.squeeze().expand_as(ub_upper_alphas)

        # get the slopes of the SOL bounds with appropriate dimensions
        upper_m = self.linear_bounds[0, :, 0, 0].squeeze().expand_as(ub_upper_alphas)
        lower_m = self.linear_bounds[0, :, 1, 0].squeeze().expand_as(ub_upper_alphas)

        db_pairs = []

        for upper_alphas, optimizable_elements in zip((ub_upper_alphas, lb_upper_alphas), (self.alpha_is_optimizable[0], self.alpha_is_optimizable[2])):

            # clip the alphas
            with (torch.no_grad()):
                upper_alphas.data = torch.clip(upper_alphas, min=ub_min_alphas, max=ub_max_alphas)

            # generate the masks
            left_line_mask = (upper_alphas.data <= upper_m).to(upper_alphas.dtype)
            right_line_mask = 1. - left_line_mask

            # retrieve upper_d
            upper_d = upper_alphas
            db_pairs.append(upper_d)

            # retrieve upper_b
            upper_b = (self.ub_l_intersection_points[:, 1].squeeze().expand_as(
                ub_upper_alphas) - upper_alphas * self.ub_l_intersection_points[:,
                                                  0].squeeze().expand_as(ub_upper_alphas)) * left_line_mask
            + (self.ub_r_intersection_points[:, 1].squeeze().expand_as(
                ub_upper_alphas) - upper_alphas * self.ub_r_intersection_points[:,
                                                  0].squeeze().expand_as(ub_upper_alphas)) * right_line_mask
            upper_b *= optimizable_elements
            db_pairs.append(upper_b)

        for lower_alphas, optimizable_elements in zip((ub_lower_alphas, lb_lower_alphas), (self.alpha_is_optimizable[1], self.alpha_is_optimizable[3])):

            # clip the alphas
            with (torch.no_grad()):
                lower_alphas.data = torch.clip(lower_alphas, min=lb_min_alphas, max=lb_max_alphas)

            # generate the masks
            right_line_mask = (lower_alphas.data <= lower_m).to(upper_alphas.dtype)
            left_line_mask = 1. - right_line_mask

            # retrieve lower_d
            lower_d = lower_alphas
            db_pairs.append(lower_d)

            # retrieve lower_b
            lower_b = (self.lb_l_intersection_points[:, 1].squeeze().expand_as(
                lb_upper_alphas) - lower_alphas * self.lb_l_intersection_points[:,
                                                  0].squeeze().expand_as(ub_upper_alphas)) * left_line_mask
            + (self.lb_r_intersection_points[:, 1].squeeze().expand_as(
                ub_upper_alphas) - lower_alphas * self.lb_r_intersection_points[:,
                                                  0].squeeze().expand_as(ub_upper_alphas)) * right_line_mask
            lower_b *= optimizable_elements
            db_pairs.append(lower_b)

        assert len(db_pairs) == 8, "Not enough db pairs to unpack"
        ub_upper_d, ub_upper_b, lb_upper_d, lb_upper_b, ub_lower_d, ub_lower_b, lb_lower_d, lb_lower_b = db_pairs
        return ub_upper_d, ub_upper_b, lb_upper_d, lb_upper_b, ub_lower_d, ub_lower_b, lb_lower_d, lb_lower_b

    def _get_tangent_line(self, activation, d1, x):
        """
        Returns the tangent line of activation.
        :param activation:
        :param d1:
        :param x:
        :return:
        """
        y = activation(x)
        m = d1(x).squeeze().expand_as(y)
        b = y - m * x
        return m, b

    def _get_intersecting_points(self, m1: torch.Tensor, b1: torch.Tensor, m2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """
        Returns the intersecting points between two lines.
        :param m1:
        :param b1:
        :param m2:
        :param b2:
        :return:
        """
        x = (b2 - b1) / (m1 - m2)
        y = m1 * (b2 - b1) / (m1 - m2) + b1
        return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)

class GeneralOptimizableActivationOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        return g.op('custom::GeneralOptimizableActivationOp', x)

    @staticmethod
    def forward(ctx, x):
        return general_function(x)

class GeneralOptimizableActivationFunction(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return GeneralOptimizableActivationOp.apply(x)

