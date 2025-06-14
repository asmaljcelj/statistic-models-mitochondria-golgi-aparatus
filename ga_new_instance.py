import argparse

import numpy as np
import trimesh

import math_utils
import utils


class SigmaParameters:
    length = [0.2]
    size = [0.2]
    smoothing_iterations = 0

    def __init__(self, length=[0.2], size=[0.2], shape=[1], smoothing_iterations=0):
        self.length = length
        self.size = size
        self.scale = shape
        self.smoothing_iterations = smoothing_iterations

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
            center_index = index
            index += 1
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
    direction_vectors = math_utils.generate_direction_vectors(np.array(
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]),
        num_of_direction_vectors)
    direction_vectors = np.array(direction_vectors)
    for i, direction_vector in enumerate(direction_vectors):
        distance = distances[i]
        points.append(direction_vector * distance + starting_point)
    return np.array(points)


def calculate_new_distances_for_cisterna(data, sigma_length, sigma_scale):
    distances = []
    for i, cisterna_distances in enumerate(data):
        value = utils.retrieve_new_value_from_standard_derivation_ga(sigma_length, sigma_scale, cisterna_distances)
        if value < 0:
            value = 0
        distances.append(value)
    return distances


def generate_instance(num_of_cisternas, distances, num_of_direction_vectors, sigma):
    # figure out number of cisterna stacks
    # generate landmark points at each step
    # first generate cisterna 1
    sequence_length = np.linspace(0, len(sigma.length) - 1, num_of_cisternas, dtype=int)
    sequence_scale = np.linspace(0, len(sigma.scale) - 1, num_of_cisternas, dtype=int)
    average_distances_zero = calculate_new_distances_for_cisterna(distances[0], sigma.length[0], sigma.scale[0])
    average_distances_max = calculate_new_distances_for_cisterna(distances[len(distances) - 1], sigma.length[len(sigma.length) - 1], sigma.scale[len(sigma.scale) - 1])
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
        average_distances = calculate_new_distances_for_cisterna(combined_data, sigma.length[sequence_length[i]], sigma.scale[sequence_scale[i]])
        points = populate_instances(average_distances, [0, 0, i * multiplicator], num_of_direction_vectors)
        final_object_dict[i] = points
        final_object.extend(points)
    # se zadnja cisterna
    final = populate_instances(average_distances_max, [0, 0, (num_of_cisternas - 1) * multiplicator], num_of_direction_vectors)
    final_object_dict[num_of_cisternas - 1] = final
    final_object.extend(final)
    return np.array(final_object), final_object_dict


def create_parser():
    parser = argparse.ArgumentParser(description='Generate new GA shape')
    parser.add_argument('-c', '--cisternas', help='number of cisternas in stack')
    parser.add_argument('-l', '--length', help='value of sigma for length in each direction', nargs='*')
    parser.add_argument('-s', '--seed', help='value of seed for random number generator')
    parser.add_argument('-sc', '--scale', help='value of sigma for scale in each direction', nargs='*')
    parser.add_argument('-it', '--smoothing-iterations', help='number of smoothing iterations')
    return parser


data = utils.read_measurements_from_file_ga('measurements_ga/learn/measurements_ga.pkl')
meta_data = data[0]
data.pop(0)
num_cisternae = None
parser = create_parser()
args = parser.parse_args()
sigma = SigmaParameters()
seed=123
num_of_direction_vectors = meta_data[1]
if args.cisternas:
    value = int(args.cisternas)
    num_cisternae = value
if args.length:
    values = list(args.length)
    if len(values) == 1:
        sigma.length = [float(values[0])]
    sigma.length = [float(s) for s in values]
if args.scale:
    values = list(args.scale)
    if len(values) == 1:
        sigma.scale = [float(values[0])]
    sigma.scale = [float(s) for s in values]
if args.seed:
    seed = int(args.seed)
if args.smoothing_iterations:
    sigma.smoothing_iterations = int(args.smoothing_iterations)
if num_cisternae is None:
    min_size = min(meta_data[0])
    max_size = max(meta_data[0])
    num_cisternae = round(np.random.uniform(min_size, max_size))
print('generating object with', num_cisternae, 'cisternae')
np.random.seed(seed)
object_points, points_dict = generate_instance(num_cisternae, data, num_of_direction_vectors, sigma)
vertices, faces = generate_mesh(points_dict)
tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
smooth = trimesh.smoothing.filter_humphrey(tri_mesh, iterations=sigma.smoothing_iterations)
utils.generate_obj_file(smooth.vertices, smooth.faces, f'mesh_ga.obj')