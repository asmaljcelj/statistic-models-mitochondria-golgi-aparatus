import argparse

import numpy as np
import trimesh

import math_utils
import utils


class Parameters:
    length = 0.2
    skeleton_points = 0.2
    start_points = 0.2
    end_points = 0.2
    curvature = 0.2
    torsion = 0.2
    smoothing_iterations = 0

    def __init__(self, length=0.2, skeleton=0.2, start=0.2, end=0.2, curvature=0.2, torsion=0.2, smoothing_iterations=0):
        self.length = length
        self.skeleton_points = skeleton
        self.start_points = start
        self.end_points = end
        self.curvature = curvature
        self.torsion = torsion
        self.smoothing_iterations = smoothing_iterations

    def __str__(self):
        return 'length = {:.2f}, skeleton = {:.2f}, start = {:.2f}, end = {:.2f}, curvature = {:.2f}, torsion = {:.2f}'.format(self.length, self.skeleton_points, self.start_points, self.end_points, self.curvature, self.torsion)


def extract_skeleton_points(skeleton_measurements, skeleton_length, sigma, curvature, torsions):
    skeleton_sampled_outside_points = {}
    # initialize starting skeleton point
    skeleton_points = np.array([[0, 0, 0]])
    # initialize T, N and B vectors
    T = np.array([0, 0, 1])
    N = np.array([1, 0, 0])
    B = np.array([0, 1, 0])
    for index, c in enumerate(curvature):
        # get curvature and torsion values for current skeleton point
        if len(curvature[c]) == 1:
            new_curvature = curvature[c][0]
        else:
            new_curvature = utils.retrieve_new_value_from_standard_derivation(sigma.curvature, curvature[c])[0]
        if len(torsions[c]) == 1:
            new_torsion = torsions[c][0]
        else:
            new_torsion = utils.retrieve_new_value_from_standard_derivation(sigma.torsion, torsions[c])[0]
        # new_torsion = 0
        print('using values', new_curvature, new_torsion, 'for curvature and torsion at index', index)
        # get new skeleton value
        solution = math_utils.calculate_next_skeleton_point(skeleton_points[-1], T, N, B, new_curvature, new_torsion,
                                                            skeleton_length / len(curvature))[1]
        # update T. N and B
        T = [solution[3], solution[4], solution[5]]
        N = [solution[6], solution[7], solution[8]]
        B = [solution[9], solution[10], solution[11]]
        new_skeleton_point = [solution[0], solution[1], solution[2]]
        skeleton_points = np.append(skeleton_points, [new_skeleton_point], axis=0)
        if index not in skeleton_measurements:
            continue
        # sample new points around that skeleton point
        for angle, distances in skeleton_measurements[index].items():
            if index not in skeleton_sampled_outside_points:
                skeleton_sampled_outside_points[index] = {}
            new_distance = utils.retrieve_new_value_from_standard_derivation(sigma.skeleton_points, distances)
            direction = math_utils.rotate_vector(np.array(N), angle, np.array(T))
            new_point = new_distance * np.array(direction) + new_skeleton_point
            skeleton_sampled_outside_points[index][angle % 360] = new_point
    return skeleton_sampled_outside_points, skeleton_points


def extract_edge_points(edge_distances, normal, skeleton_point, edge_direction_with_angles, start_points):
    points_dict = {}
    R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal)
    for direction, distances in edge_distances.items():
        distances = np.array(distances)
        theta, phi = edge_direction_with_angles[direction]
        new_distance = utils.retrieve_new_value_from_standard_derivation(parameters.end_points, distances)
        new_direction = np.dot(R, direction)
        if start_points:
            new_direction[2] *= -1
        new_point = new_distance * new_direction + skeleton_point
        new_point_key = (new_point[0], new_point[1], new_point[2])
        points_dict[new_point_key] = (theta, phi)
    return points_dict


def sample_new_points(skeleton_distances, start_distances, end_distances, curvature, direction_with_angles, lengths, torsions, random_seed=2000, parameters=Parameters()):
    np.random.seed(random_seed)
    # calculate skeleton points based on curvature
    print('finding new skeleton points')
    if len(lengths) == 1:
        total_skeleton_length = lengths[0]
    else:
        total_skeleton_length = utils.retrieve_new_value_from_standard_derivation(parameters.length, lengths)[0]
    print('using skeleton length of', total_skeleton_length)
    print('generating skeleton points')
    # calculate new skeleton points of mitochondria using Frenet-Serret equations
    skeleton_outside_points, skeleton_points = extract_skeleton_points(skeleton_distances, total_skeleton_length, parameters, curvature, torsions)
    print('generating end points')
    end_points_dict = extract_edge_points(end_distances, math_utils.normalize(skeleton_points[-1] - skeleton_points[-2]), skeleton_points[-1], direction_with_angles, False)
    print('generating starts points')
    start_points_dict = extract_edge_points(start_distances, math_utils.normalize(skeleton_points[0] - skeleton_points[1]) * -1, np.array([0, 0, 0]), direction_with_angles, True)
    vertices, faces = generate_mesh(skeleton_outside_points, start_points_dict, end_points_dict)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    smooth = trimesh.smoothing.filter_humphrey(tri_mesh, iterations=parameters.smoothing_iterations)
    utils.generate_obj_file(smooth.vertices, smooth.faces, f'mesh.obj')


def generate_mesh(skeleton_points, start_points, end_points):
    # seznam kotov za vzorčenje
    skeleton_angles = list(skeleton_points[1].keys())
    # inkrement med posameznimi koti
    angle_increment = skeleton_angles[1] - skeleton_angles[0]
    vertices, faces = [], []
    # skeleton points
    point_vertice_indexes = {}
    current_point_mesh_index = 0
    # start points
    print('start points')
    kroznica_start_points, current_point_mesh_index = sample_edge_points(start_points, vertices, faces, angle_increment, point_vertice_indexes, current_point_mesh_index, False)
    print('end points')
    kroznica_end_points, current_point_mesh_index = sample_edge_points(end_points, vertices, faces, angle_increment, point_vertice_indexes, current_point_mesh_index, True)
    print('skeleton points')
    sample_skeleton_points(skeleton_points, vertices, faces, kroznica_start_points, kroznica_end_points, point_vertice_indexes, current_point_mesh_index, angle_increment)
    return vertices, faces


def sample_edge_points(edge_sampled_points, vertices, faces, angle_increment, point_vertice_indexes, current_index, flip_orientation):
    grouped_data, top_point, kroznica_points = utils.group_edge_points_by_theta_extract_top_point(edge_sampled_points, angle_increment=angle_increment)
    top_point = top_point[0]
    for group_index, group in grouped_data.items():
        if group_index == len(grouped_data) - 1:
            vertices.append(top_point)
            key = (top_point[0], top_point[1], top_point[2])
            point_vertice_indexes[key] = current_index
            current_index += 1
        for index_inside_group, (point, angles) in enumerate(group):
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = current_index
            vertices.append(point)
            if index_inside_group == 0:
                # edge case, where first point can't connect to the previous one
                current_index += 1
                continue
            if group_index == 0:
                current_index += 1
                continue
            previous_point_same_ring = group[index_inside_group - 1][0]
            previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
            same_point_previous_ring = grouped_data[group_index - 1][index_inside_group][0]
            same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
            previous_point_previous_ring = grouped_data[group_index - 1][index_inside_group - 1][0]
            previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
            utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key], flip_orientation)
            utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key], flip_orientation)
            if group_index == len(grouped_data) - 1 and index_inside_group > 0:
                # connect all the points to the top point
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[top_point], flip_orientation)
                if index_inside_group == len(group) - 1:
                    # zadnje vozlisce povezi s prejsnjim na vrhu
                    previous_point_same_ring = group[0][0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[top_point], point_vertice_indexes[previous_point_same_ring_key], flip_orientation)
            if index_inside_group == len(group) - 1:
                # zadnjo vozlisce povezi s prvim
                previous_point_same_ring = group[0][0]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                same_point_previous_ring = grouped_data[group_index - 1][index_inside_group][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                previous_point_previous_ring = grouped_data[group_index - 1][0][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key], flip_orientation)
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key], flip_orientation)
            current_index += 1
    # also connect to kroznica points
    last_connected_index = 0
    for kroznica_index, p in enumerate(kroznica_points):
        i1 = kroznica_index
        p = p[0]
        vertices.append(p)
        key = utils.dict_key_from_point(p)
        point_vertice_indexes[key] = current_index
        if i1 > 0:
            skeleton_index = int(i1 * len(grouped_data[0]) / len(kroznica_points))
            utils.add_triangle_to_mesh(faces, current_index, current_index - 1, point_vertice_indexes[grouped_data[0][skeleton_index][0]], flip_orientation)
            if skeleton_index != last_connected_index:
                # make extra triangles
                for i in range(last_connected_index + 1, skeleton_index + 1):
                    utils.add_triangle_to_mesh(faces, current_index - 1, point_vertice_indexes[grouped_data[0][i - 1][0]], point_vertice_indexes[grouped_data[0][i][0]], flip_orientation)
                last_connected_index = skeleton_index
            if i1 == len(kroznica_points) - 1:
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[grouped_data[0][skeleton_index][0]], point_vertice_indexes[kroznica_points[0][0]], flip_orientation)
                utils.add_triangle_to_mesh(faces, point_vertice_indexes[kroznica_points[0][0]], point_vertice_indexes[grouped_data[0][skeleton_index][0]], point_vertice_indexes[grouped_data[0][0][0]], flip_orientation)
        current_index += 1
    return kroznica_points, current_index


def sample_skeleton_points(sampled_points, vertices, faces, kroznica_start_points, kroznica_end_points, point_vertice_indexes, current_index, angle_increment):
    for ring, points_at_angle in sampled_points.items():
        angles = list(points_at_angle.keys())
        angles.sort()
        for i, angle in enumerate(angles):
            point = points_at_angle[angle]
            vertices.append(point)
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = current_index
            if angle == 0:
                current_index += 1
                continue
            if 1 < ring:
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = sampled_points[ring - 1][angle - angle_increment]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = sampled_points[ring - 1][angle]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key])
                if i == len(angles) - 1:
                    # zadnjo vozlisce povezi s prvim
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    same_point_previous_ring = sampled_points[ring - 1][angle]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    previous_point_previous_ring = sampled_points[ring - 1][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key])
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
            elif ring == 1:
                # last ring connects to start points
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = kroznica_start_points[i - 1][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = kroznica_start_points[i][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key])
                if i == len(angles) - 1:
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    previous_point_previous_ring = kroznica_start_points[0][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    same_point_previous_ring = kroznica_start_points[i][0]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key])
            if ring == len(sampled_points):
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = kroznica_end_points[i - 1][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = kroznica_end_points[i][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key])
                utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
                if i == len(angles) - 1:
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    previous_point_previous_ring = kroznica_end_points[0][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    same_point_previous_ring = kroznica_end_points[i][0]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key])
                    utils.add_triangle_to_mesh(faces, current_index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key])
            current_index += 1


def create_parser():
    parser = argparse.ArgumentParser(description='Generate new mitochondria shape')
    parser.add_argument('-c', '--curvature', help='curvature of the generated shapes', nargs='*')
    parser.add_argument('-l', '--length', help='length of the generated shapes')
    parser.add_argument('-t', '--torsion', help='torsion of the generated shapes', nargs='*')
    parser.add_argument('-s', '--seed', help='seed of the random number generator')
    parser.add_argument('-sl', '--sigma-length', help='value of interval_width for length')
    parser.add_argument('-ss', '--sigma-skeleton', help='value of interval_width for skeleton points')
    parser.add_argument('-sst', '--sigma-start', help='value of interval_width for start points')
    parser.add_argument('-se', '--sigma-end', help='value of interval_width for end points')
    parser.add_argument('-sc', '--sigma-curvature', help='value of interval_width for curvature')
    parser.add_argument('-st', '--sigma-torsion', help='value of interval_width for torsion')
    parser.add_argument('-it', '--smoothing-iterations', help='number of smoothing iterations')
    return parser


def validate_sigma_parameter(sigma):
    if sigma < 0 or sigma > 1:
        raise ValueError('sigma value must be between 0 and 1')


curvature, start, end, skeleton, lengths, direction_with_angles, torsions = utils.read_measurements_from_file('measurements/learn/measurements.pkl')
parser = create_parser()
args = parser.parse_args()
seed = 123
if args.curvature:
    if len(curvature.keys()) != len(args.curvature):
        raise Exception('number of curvature values must match the number of characteristic points')
    for i, values in curvature.items():
        curvature[i] = [float(args.curvature[i])]
if args.seed:
    seed = args.seed
parameters = Parameters()
if args.torsion:
    if len(torsions.keys()) != len(args.torsion):
        raise Exception('number of torsion values must match the number of characteristic points')
    for i, values in curvature.items():
        torsions[i] = [float(args.torsion[i])]
if args.sigma_length:
    value = float(args.sigma_length)
    validate_sigma_parameter(value)
    parameters.length = value
if args.sigma_skeleton:
    value = float(args.sigma_skeleton)
    validate_sigma_parameter(value)
    parameters.skeleton_points = value
if args.sigma_start:
    value = float(args.sigma_start)
    validate_sigma_parameter(value)
    parameters.start_points = value
if args.sigma_end:
    value = float(args.sigma_end)
    validate_sigma_parameter(value)
    parameters.end_points = value
if args.sigma_curvature:
    value = float(args.sigma_curvature)
    validate_sigma_parameter(value)
    parameters.curvature = value
if args.sigma_torsion:
    value = float(args.sigma_torsion)
    validate_sigma_parameter(value)
    parameters.torsion = value
if args.length:
    lengths = [float(args.length)]
if args.smoothing_iterations:
    value = int(args.smoothing_iterations)
    parameters.smoothing_iterations = value
print('using sigma values:', parameters)
sample_new_points(skeleton, start, end, curvature, direction_with_angles, lengths, torsions, random_seed=seed, parameters=parameters)
