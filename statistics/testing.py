import math

import utils
import trimesh
import skeletonization
import numpy as np
import bezier
import sampling
import math_utils
import os


def calculate_rmse_for_object(new_object_path, testing_directory):
    direction_vectors, direction_with_angles = sampling.sample_direction_vectors(1000, 3)
    mesh = trimesh.load(new_object_path)
    voxelized = mesh.voxelized(pitch=1.0)
    filled = voxelized.fill()

    # Get list of voxel coordinates (as numpy array)
    voxels = filled.points.astype(int)
    # utils.plot_3d(voxels)
    image = utils.create_3d_image(voxels)
    result = skeletonization.skeletonize_voxels(np.array(image))
    bezier_curve, arc, length = bezier.perform_arc_length_parametrization_bezier_curve(5, result, 15)
    skeleton_distances = {}
    for i in range(1, len(arc) - 2):
        previous_point = arc[i - 1]
        current_point = arc[i]
        next_point = arc[i + 1]
        vector_to_next_point = math_utils.normalize(next_point - current_point)
        vector_to_previous_point = math_utils.normalize(previous_point - current_point)
        normal = np.cross(vector_to_next_point, vector_to_previous_point)
        if np.any(np.isnan(normal)):
            return None, None, None
        normal = math_utils.normalize(normal)
        if math.isnan(normal[0]) or math.isnan(normal[1]) or math.isnan(normal[2]):
            normal = np.array([0, 1, 0])
        skeleton_distances[i], _ = sampling.sample_rays(current_point, normal, image, vector_to_next_point, image, 3)
    distances_start, _ = sampling.sample_at_ends(arc[0], arc[1], image, direction_vectors)
    distances_end, _ = sampling.sample_at_ends(arc[len(arc) - 1], arc[len(arc) - 2], image, direction_vectors)
    total_rmse, num_of_testing_intances = 0, 0
    for filename in os.listdir(testing_directory):
        print('processing', filename)
        total_rmse += calculate_rmse_between_objects(distances_start, distances_end, skeleton_distances, filename)
        num_of_testing_intances += 1
    return round(total_rmse / num_of_testing_intances, 3)


def calculate_rmse_between_objects(new_start, new_end, new_skeleton, test_object):
    _, start, end, skeleton, _, _, _ = utils.read_measurements_from_file('../measurements/testing/' + test_object)
    rmse, num_instances = 0, 0
    rmse_value, n = rmse_calculate_edges(new_start, start)
    num_instances += n
    rmse += rmse_value
    rmse_value, n = rmse_calculate_edges(new_end, end)
    num_instances += n
    rmse += rmse_value
    rmse_value, n = rmse_calculate_skeleton(new_skeleton, skeleton)
    num_instances += n
    rmse += rmse_value
    return math.sqrt(rmse / num_instances)


def rmse_calculate_edges(actual, testing):
    rmse = 0
    for key in actual:
        actual_value = actual[key]
        test_value = testing[key]
        rmse += (actual_value - test_value) ** 2
    return rmse, len(actual)


def rmse_calculate_skeleton(actual, testing):
    rmse, num_instances = 0, 0
    for key in actual:
        actual_values = actual[key]
        test_values = testing[key]
        for i, actual_value_and_angle in enumerate(actual_values):
            actual_value = actual_value_and_angle[0]
            test_value = test_values[i][0]
            rmse += (actual_value - test_value) ** 2
        num_instances += len(actual_values)
    return rmse, num_instances


# mito
rmse = calculate_rmse_for_object('../results/smooth_025_10_123.obj', '../measurements/testing/')
print('rmse for ...:', rmse)

