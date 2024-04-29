import numpy as np


STANDARD_REGION = np.array([-2, 2])


def sample_subregions(n):
    region_width = STANDARD_REGION[1] - STANDARD_REGION[0]
    region_boundaries = STANDARD_REGION[0] + region_width * np.random.rand(n, 2)
    return np.sort(region_boundaries, axis=1)


def sample_points(region_boundaries, n_points):
    points = np.random.rand(region_boundaries.shape[0], n_points)
    scales = region_boundaries[:, 1] - region_boundaries[:, 0]
    points = points * np.expand_dims(scales, axis=1) + np.expand_dims(region_boundaries[:, 0], axis=1)
    return np.expand_dims(points, axis=-1)


def generate_dataset(num_instances, num_points, functions_list):
    datasets = []
    for function in functions_list:
        boundaries = sample_subregions(num_instances)
        points = sample_points(boundaries, num_points)
        datasets.append((boundaries, points, function(points[..., 0])))
    return datasets