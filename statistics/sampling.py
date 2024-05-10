import os

import numpy as np

from bezier import perform_arc_length_parametrization_bezier_curve
from utils import read_file_collect_points, read_nii_file, plot_sampling_with_shape
from math_utils import magnitude, rotate_vector, distance_between_points

# todo: neki cudno sampla tudi izven BB-ja????????
def sample_rays(origin, direction_vector, shape, base_x, parametrized_points, skeleton):
    distances = []
    sampled_points = {}
    base_x = base_x / magnitude(base_x)
    counter = 0
    while counter < 360:
        distance, sampled_points_a = iterate_ray(origin, direction_vector, shape)
        distances.append(distance)
        sampled_points[counter] = sampled_points_a
        direction_vector = get_new_direction_vector(direction_vector, base_x)
        counter += 1
    plot_sampling_with_shape(shape, sampled_points, skeleton, parametrized_points)
    return distances


def iterate_ray(origin, direction_vector, shape):
    current_point = origin
    sampled_points_a = [current_point]
    previous_voxel, current_voxel = None, [int(origin[0]), int(origin[1]), int(origin[2])]
    while True:
        new_point = current_point + direction_vector
        x_coord, y_coord, z_coord = int(new_point[0]), int(new_point[1]), int(new_point[2])
        previous_voxel = current_voxel
        current_voxel = [x_coord, y_coord, z_coord]
        in_boundary = shape[x_coord][y_coord][z_coord] > 0
        # todo: improved collision detection and distance measurement
        if not in_boundary:
            # distance += magnitude(direction_vector)
            boundary_voxel = [previous_voxel[0], previous_voxel[1], previous_voxel[2]]
            distance = distance_between_points(origin, boundary_voxel)
            break
        sampled_points_a.append(new_point)
        # else:
        #     pass
        current_point = new_point
    return distance, sampled_points_a


def get_new_direction_vector(previous_vector, base_x):
    return rotate_vector(previous_vector, 1, base_x)


if __name__ == '__main__':
    skeletons_folder = '../skeletons/'
    num_of_points = 10
    for filename in os.listdir(skeletons_folder):
        distances = {}
        points = read_file_collect_points(filename, skeletons_folder)
        object_points = read_nii_file('../extracted_data/', filename.replace('.csv', '.nii'))
        if points is None:
            print('no points for file', filename)
            continue
        _, arc = perform_arc_length_parametrization_bezier_curve(5, points, num_of_points)
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
                distances[i] = sample_rays(current_point, normal, object_points, a, arc, points)
        print(distances)
        break
