import argparse

import numpy as np
from matplotlib import pyplot as plt

import math_utils
import utils

import trimesh

class SigmaParameters:
    length = 0.2
    size = 0.2

    def __init__(self, length=0.2, size=0.2):
        self.length = length
        self.size = size

    def __str__(self):
        return 'length = {:.2f}, size = {:.2f}'.format(self.length, self.size)


def generate_mesh(points):
    index = 0
    vertices, faces = [], []
    for i, c in enumerate(points):
        cisterna_points = points[c]
        if i > 0:
            for j, point in enumerate(cisterna_points):
                vertices.append(point)
                if j > 0:
                    same_ring_previous = index - 1
                    previous_ring_same = index - len(cisterna_points)
                    previous_ring_previous = index - len(cisterna_points) - 1
                    faces.append([index, previous_ring_previous, previous_ring_same])
                    faces.append([index, same_ring_previous, previous_ring_previous])
                if j == len(cisterna_points) - 1:
                    same_ring_previous = index - len(cisterna_points) + 1
                    previous_ring_same = index - len(cisterna_points)
                    previous_ring_previous = index - 2 * len(cisterna_points) + 1
                    faces.append([index, previous_ring_same, previous_ring_previous])
                    faces.append([index, previous_ring_previous, same_ring_previous])
                index += 1
        if i == 0 or i == len(points) - 1:
            # ce smo na dnu ali na vrhu, generiraj se eno tocko v sredini in poveži v mesh
            center = np.mean(cisterna_points, axis=0)
            vertices.append(center)
            index += 1
            center_index = index
            for j, point in enumerate(cisterna_points):
                vertices.append(point)
                if j > 0:
                    if i == 0:
                        faces.append([index, index - 1, center_index])
                    else:
                        faces.append([index, center_index, index - 1])
                if j == len(cisterna_points) - 1:
                    if i == 0:
                        faces.append([index, center_index + 1, center_index])
                    else:
                        faces.append([index, center_index, center_index + 1])
                index += 1
    return vertices, faces

def populate_instances(distances, starting_point, num_of_direction_vectors):
    points = []
    direction_vectors = math_utils.generate_direction_vectors(num_of_direction_vectors)
    direction_vectors = np.array(direction_vectors)
    for i, direction_vector in enumerate(direction_vectors):
        distance = distances[i]
        points.append(direction_vector * distance + starting_point)
    # for i, distance in enumerate(distances):
    #     if i == 0:
    #         points.append([distance, 0, starting_point[2]])
    #     if i == 1:
    #         # points.append([0, distance, starting_point[2]])
    #         scaled_vector = [1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
    #         points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #     if i == 2:
    #         # points.append([-distance, 0, starting_point[2]])
    #         points.append([0, distance, starting_point[2]])
    #     if i == 3:
    #         # points.append([0, -distance, starting_point[2]])
    #         scaled_vector = [-1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
    #         points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #     if i == 4:
    #         # scaled_vector = [1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
    #         # points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #         points.append([-distance, 0, starting_point[2]])
    #     if i == 5:
    #         # scaled_vector = [1 / np.sqrt(2) * distance, -1 / np.sqrt(2) * distance, 0.0]
    #         # points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #         scaled_vector = [-1 / np.sqrt(2) * distance, -1 / np.sqrt(2) * distance, 0.0]
    #         points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #     if i == 6:
    #         # scaled_vector = [-1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
    #         # points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    #         points.append([0, -distance, starting_point[2]])
    #     if i == 7:
    #         scaled_vector = [1 / np.sqrt(2) * distance, -1 / np.sqrt(2) * distance, 0.0]
    #         points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    return np.array(points)


def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.legend()
    plt.show()


def calculate_new_distances_for_cisterna(data, sigma):
    distances = []
    for cisterna_distances in data:
        value = utils.retrieve_new_value_from_standard_derivation(sigma, cisterna_distances)[0]
        if value < 0:
            value = 0
        distances.append(value)
    return distances


def generate_average(num_of_cisternas, distances, num_of_direction_vectors, sigma):
    # figure out number of cisterna stacks
    # cisternae_sizes = np.array(cisternae_sizes)
    # num_of_cisternas = int(np.mean(cisternae_sizes))
    # generate landmark points at each step
    # first generate cisterna 1
    # average_distances_zero = math_utils.calculate_average_cisterna(distances[0])
    average_distances_zero = calculate_new_distances_for_cisterna(distances[0], sigma)
    # average_distances_max = math_utils.calculate_average_cisterna(distances[len(distances) - 1])
    average_distances_max = calculate_new_distances_for_cisterna(distances[len(distances) - 1], sigma)
    # zdruzi posamezne meritve med sabo
    num_of_remaining = num_of_cisternas - 2
    num_per_stack = int(len(distances) / num_of_remaining)
    final_object, final_object_dict = [], {}
    zero = populate_instances(average_distances_zero, [0, 0, 0], num_of_direction_vectors)
    final_object_dict[0] = zero
    final_object.extend(zero)
    multiplicator = 1
    for i in range(1, num_of_cisternas - 1):
        # zdruzi
        starting_index = (i - 1) * num_per_stack + 1
        ending_index = i * num_per_stack
        combined_data = [[] for _ in range(len(distances[0]))]
        for j in range(starting_index, ending_index + 1):
            if j >= len(distances):
                break
            for k in range(0, len(distances[0])):
                combined_data[k].extend(distances[j][k])
        combined_data = np.array(combined_data)
        # average_distances = math_utils.calculate_average_cisterna(combined_data)
        average_distances = calculate_new_distances_for_cisterna(combined_data, sigma)
        points = populate_instances(average_distances, [i * multiplicator, 0, 0], num_of_direction_vectors)
        final_object_dict[i] = points
        final_object.extend(points)
    final = populate_instances(average_distances_max, [(num_of_cisternas - 1) * multiplicator, 0, 0], num_of_direction_vectors)
    final_object_dict[num_of_cisternas - 1] = final
    final_object.extend(final)
    return np.array(final_object), final_object_dict


def create_parser():
    parser = argparse.ArgumentParser(description='Generate new GA shape')
    parser.add_argument('-c', '--cisternas', help='number of cisternas in stack')
    parser.add_argument('-l', '--length', help='value of sigma for length in each direction')
    return parser


if __name__ == '__main__':
    data = utils.read_measurements_from_file_ga('../measurements/measurements_ga.pkl')
    meta_data = data[0]
    data.pop(0)
    num_cisternas = int(np.mean(meta_data[0]))
    parser = create_parser()
    args = parser.parse_args()
    sigma = SigmaParameters()
    if args.cisternas:
        value = int(args.cisternas)
        num_cisternas = value
    if args.length:
        value = float(args.length)
        sigma.length = value
    num_of_direction_vectors = meta_data[1]
    num_cisternas = 45
    average_object_points, points_dict = generate_average(num_cisternas, data, num_of_direction_vectors, sigma.length)
    # plot_points(average_object_points)
    vertices, faces = generate_mesh(points_dict)
    utils.generate_obj_file(vertices, faces, f'ga.obj')
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    smooth = trimesh.smoothing.filter_humphrey(tri_mesh)
    utils.generate_obj_file(smooth.vertices, smooth.faces, f'smooth_ga.obj')
