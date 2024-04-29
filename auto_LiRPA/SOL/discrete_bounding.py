import gurobipy as gr
import numpy as np

from scipy.optimize import linprog

from .py_sdlp import solve_lp_2d, solve_lp_3d

'''
For all function here the "center of mass" of the region is assumed to be at x = 0.
All functions assume the natural Nd signature of the input tensors:
points: [n_points, coordinates]
values: [n_points]
eps   : []
'''

def bound_discrete_gurobi(points, values, eps):
    model = gr.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('FeasibilityTol', eps)
    model.setParam('OptimalityTol', eps)
    model.setParam('BarConvTol', eps)
    model.setParam('Method', 3)  # Concurrent
    model.setParam('Threads', 0)  # Use all available threads

    constraint_coeffs = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    vars = model.addMVar(constraint_coeffs.shape[1], lb=-gr.GRB.INFINITY, ub=gr.GRB.INFINITY, vtype=gr.GRB.CONTINUOUS)
    model.addMConstr(constraint_coeffs, vars, ">=", values)
    model.setObjective(vars[-1], gr.GRB.MINIMIZE)
    model.optimize()

    return np.array([v.x for v in vars])


def bound_discrete_scipy(points, values, eps):
    options = {
        'dual_feasibility_tolerance': eps,
        'primal_feasibility_tolerance': eps,
        'ipm_optimality_tolerance': eps,
    }
    constraint_coeffs = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    target_coeffs = np.zeros(constraint_coeffs.shape[1])
    target_coeffs[-1] = 1.
    result = linprog(target_coeffs, -constraint_coeffs, -values, bounds=[None, None], options=options)
    
    return result.x


def bound_discrete_bisect(points, values, eps):
    assert points.shape[1] == 1, 'Only 1d functions are supported'
    
    points = points[:, 0]
    left_mask = points < -1e-9
    right_mask = points > 1e-9
    left_points, left_values = points[left_mask], values[left_mask]
    right_points, right_values = points[right_mask], values[right_mask]

    assert left_points.shape[0] > 0 and right_points.shape[0] > 0, 'The volume is not lower-bounded'

    lower_bound = np.min(values)
    upper_bound = np.max(values)
    slope = 0.
    while upper_bound - lower_bound > eps:
        middle = (lower_bound + upper_bound) / 2

        left_min_slope = np.min((middle - left_values) / (-left_points))
        right_max_slope = np.max((right_values - middle) / right_points)

        if right_max_slope > left_min_slope:
            lower_bound = middle
        else:
            upper_bound = middle
            slope = (left_min_slope + right_max_slope) / 2

    return np.array([slope, upper_bound])


def bound_discrete_linear_LP(points, values, eps):
    assert points.shape[1] == 1, 'Only 1d functions are supported'

    points = np.stack([points[:, 0], values], axis=1)
    slope_lower_bound, slope_upper_bound = float('-inf'), float('+inf')

    n_points = points.shape[0]
    while True:
        half_n_points = n_points // 2
        leftover = points[half_n_points * 2:]

        pairs = np.stack([points[:half_n_points], points[half_n_points:half_n_points * 2]], axis=1)
        decision_values = (pairs[:, 1, 1] - pairs[:, 0, 1]) / (pairs[:, 1, 0] - pairs[:, 0, 0])

        pivot_index = np.random.randint(0, decision_values.shape[0])
        pivot = decision_values[pivot_index]
        left_mask = points[:, 0] <= 0
        right_mask = np.logical_not(left_mask)
        pivot_heights = points[:, 1] - pivot * points[:, 0]
        left_max = max(pivot_heights[left_mask])
        right_max = max(pivot_heights[right_mask])
        if abs(left_max - right_max) < eps:
            return np.array([pivot, max(left_max, right_max)])
        else:
            if left_max > right_max:
                slope_upper_bound = pivot
            else:
                slope_lower_bound = pivot

        reversed_pair = pairs[:, 0, 0] > pairs[:, 1, 0]
        greater_than_bound = decision_values > slope_upper_bound
        smaller_than_bound = decision_values < slope_lower_bound
        # Process pivot manually to make everything work when n_points ~ 1
        if left_max > right_max:
            greater_than_bound[pivot_index] = True
            smaller_than_bound[pivot_index] = False
        else:
            greater_than_bound[pivot_index] = False
            smaller_than_bound[pivot_index] = True

        pick_first = np.logical_or(
            np.logical_and(np.logical_not(reversed_pair), smaller_than_bound),
            np.logical_and(reversed_pair, greater_than_bound),
        )
        pick_second = np.logical_or(
            np.logical_and(reversed_pair, smaller_than_bound),
            np.logical_and(np.logical_not(reversed_pair), greater_than_bound),
        )
        leave_both = np.logical_not(np.logical_or(pick_first, pick_second))

        points = np.concatenate([
            pairs[pick_first][:, 0],
            pairs[pick_second][:, 1],
            pairs[leave_both].reshape([-1, 2]),
            leftover,
        ], axis=0)

        n_points = points.shape[0]


def bound_discrete_linear_LP_cpp(points, values, eps):
    assert points.shape[1] <= 2, 'Only d <= 2 functions are supported'

    # Make sure tensors are C-ordered in memory so that C++ bindings work correctly
    # TODO: enforce the correct order in the bounder to avoid copying here
    points = np.ascontiguousarray(points)
    values = np.ascontiguousarray(values)

    constraint_coeffs = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    target_coeffs = np.zeros(constraint_coeffs.shape[1])
    target_coeffs[-1] = 1.
    if points.shape[1] == 1:
        return solve_lp_2d(target_coeffs, -constraint_coeffs, -values)
    elif points.shape[1] == 2:
        return solve_lp_3d(target_coeffs, -constraint_coeffs, -values)
    else:
        raise NotImplementedError



def bound_discrete_two_sides(region_bound, points, values, method, eps=1e-7):
    center = region_bound.mean(axis=-1)
    points_shifted = points - center

    upper_bound = method(points_shifted, values, eps)
    upper_bound[-1] -= (upper_bound[:-1] * center).sum()

    lower_bound = -method(points_shifted, -values, eps)
    lower_bound[-1] -= (lower_bound[:-1] * center).sum()

    return upper_bound, lower_bound


BOUNDING_METHOD_NAME_TO_FUNCTION = {
    'gurobi': bound_discrete_gurobi,
    'scipy': bound_discrete_scipy,
    'bisect': bound_discrete_bisect,
    'linear': bound_discrete_linear_LP,
    'linear_cpp': bound_discrete_linear_LP_cpp,
}