import os

import numpy as np

from bezier import perform_arc_length_parametrization_bezier_curve
from utils import read_file_collect_points, read_nii_file, plot_sampling_with_shape
from math_utils import magnitude


def sample_ray(origin, direction_vector, shape):
    distance = 0
    # current_point = origin
    # in_boundary = True
    # while in_boundary:
    #     current_point += direction_vector
    #     x_coord, y_coord, z_coord = current_point[0], current_point[1], current_point[2]
    #     in_boundary = shape[x_coord][y_coord][z_coord] > 0
    #     if in_boundary:
    #         distance += magnitude(direction_vector)
    plot_sampling_with_shape(shape, None, None)
    return distance


def get_new_direction_vector(normal, previous_vector):
    pass


if __name__ == '__main__':
    skeletons_folder = '../skeletons/'
    for filename in os.listdir(skeletons_folder):
        points = read_file_collect_points(filename, skeletons_folder)
        object_points = read_nii_file('../extracted_data/', filename.replace('.csv', '.nii'))
        if points is None:
            continue
        _, arc = perform_arc_length_parametrization_bezier_curve(5, points, 10)
        if arc is not None:
            for i in range(1, len(arc) - 2):
                previous_point = arc[i - 1]
                current_point = arc[i]
                next_point = arc[i + 1]
                # normala
                a = next_point - current_point
                b = previous_point - current_point
                normal = np.cross(a, b)
                mag = magnitude(normal)
                normal = normal / mag
                sample_ray(current_point, normal, object_points)
