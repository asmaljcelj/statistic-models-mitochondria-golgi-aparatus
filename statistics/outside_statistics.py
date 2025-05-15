import argparse

import numpy as np
import trimesh

import math_utils
import utils


class SigmaParameters:
    length = 0.2
    skeleton_points = 0.2
    start_points = 0.2
    end_points = 0.2
    curvature = 0.2
    torsion = 0.2

    def __init__(self, length=0.2, skeleton=0.2, start=0.2, end=0.2, curvature=0.2, torsion=0.2):
        self.length = length
        self.skeleton_points = skeleton
        self.start_points = start
        self.end_points = end
        self.curvature = curvature
        self.torsion = torsion

    def __str__(self):
        return 'length = {:.2f}, skeleton = {:.2f}, start = {:.2f}, end = {:.2f}, curvature = {:.2f}, torsion = {:.2f}'.format(self.length, self.skeleton_points, self.start_points, self.end_points, self.curvature, self.torsion)


def sample_new_points(skeleton_distances, start_distances, end_distances, curvature, direction_with_angles, lengths, torsions, random_seed=2000, sigma=SigmaParameters()):
    np.random.seed(random_seed)
    # calculate skeleton points based on curvature
    print('finding new skeleton points')
    skeleton_points = []
    skeleton_outside_points, start_points_dict, end_points_dict, skeleton_lengths = {}, {}, {}, {}
    skeleton_points.append([0, 0, 0])
    if len(lengths) == 1:
        total_skeleton_length = lengths[0]
    else:
        total_skeleton_length = utils.retrieve_new_value_from_standard_derivation(sigma.length, lengths)[0]
    print('using skeleton length of', total_skeleton_length)
    print('generating skeleton points')
    # calculate new skeleton points of mitochondria using Frenet-Serret equations
    T = np.array([0, 0, 1])
    N = np.array([1, 0, 0])
    B = np.array([0, 1, 0])
    for index, c in enumerate(curvature):
        if len(curvature[c]) == 1:
            new_curvature = curvature[c][0]
        else:
            new_curvature = utils.retrieve_new_value_from_standard_derivation(sigma.curvature, curvature[c])[0]
        if len(torsions[c]) == 1:
            new_torsion = torsions[c][0]
        else:
            new_torsion = utils.retrieve_new_value_from_standard_derivation(sigma.torsion, torsions[c])[0]
        new_torsion = 0
        print('using values', new_curvature, new_torsion, 'for curvature and torsion at index', index)
        solution = math_utils.calculate_next_skeleton_point(skeleton_points[-1], T, N, B, new_curvature, new_torsion, total_skeleton_length / len(curvature))[1]
        old_T, old_N, old_B = T, N, B
        # update T. N and B
        T = [solution[3], solution[4], solution[5]]
        N = [solution[6], solution[7], solution[8]]
        B = [solution[9], solution[10], solution[11]]
        new_skeleton_point = [solution[0], solution[1], solution[2]]
        # if index == len(curvature) - 1:
        #     utils.plot_generated_skeleton_points(skeleton_points, T, N, B, new_skeleton_point, old_T, old_T, old_T)
        skeleton_points = np.append(skeleton_points, [new_skeleton_point], axis=0)
        if index not in skeleton_distances:
            continue
        for angle, distances in skeleton_distances[index].items():
            if index not in skeleton_outside_points:
                skeleton_outside_points[index] = {}
            new_distance = utils.retrieve_new_value_from_standard_derivation(sigma.skeleton_points, distances)
            direction = math_utils.rotate_vector(np.array(N), angle, np.array(T))
            new_point = new_distance * np.array(direction) + new_skeleton_point
            skeleton_outside_points[index][angle % 360] = new_point
    print('generating end points')
    for direction, distances in end_distances.items():
        distances = np.array(distances)
        theta, phi = direction_with_angles[direction]
        new_distance = utils.retrieve_new_value_from_standard_derivation(sigma.end_points, distances)
        normal_start = math_utils.normalize(skeleton_points[-1] - skeleton_points[-2])
        R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal_start)
        new_direction = np.dot(R, direction)
        new_point = new_distance * new_direction + skeleton_points[-1]
        new_point_key = (new_point[0], new_point[1], new_point[2])
        end_points_dict[new_point_key] = (theta, phi)
    print('generating starts points')
    for direction, distances in start_distances.items():
        distances = np.array(distances)
        new_distance = utils.retrieve_new_value_from_standard_derivation(sigma.start_points, distances)
        theta, phi = direction_with_angles[direction]
        normal_start = math_utils.normalize(skeleton_points[0] - skeleton_points[1]) * -1
        R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal_start)
        new_direction = np.dot(R, direction)
        new_direction[2] *= -1
        new_point = new_distance * new_direction + skeleton_points[0]
        new_point_key = (new_point[0], new_point[1], new_point[2])
        start_points_dict[new_point_key] = (theta, phi)
    vertices, faces = generate_mesh(skeleton_outside_points, start_points_dict, end_points_dict)
    utils.generate_obj_file(vertices, faces, f'test.obj')
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    smooth = trimesh.smoothing.filter_humphrey(tri_mesh, iterations=10)
    utils.generate_obj_file(smooth.vertices, smooth.faces, f'priblizek1.obj')


def generate_mesh(skeleton_points, start_points, end_points):
    skeleton_angles = list(skeleton_points[1].keys())
    angle_increment = skeleton_angles[1] - skeleton_angles[0]
    vertices, faces = [], []
    # skeleton points
    point_vertice_indexes = {}
    index = 0
    # start points
    print('start points')
    grouped_data_start, top_point_start, kroznica_start_points = utils.group_edge_points_by_theta_extract_top_point(start_points, angle_increment=angle_increment)
    top_point_start = top_point_start[0]
    for group_index, group in grouped_data_start.items():
        if group_index == len(grouped_data_start) - 1:
            # before processing last ring, add top point to vertices
            vertices.append(top_point_start)
            key = (top_point_start[0], top_point_start[1], top_point_start[2])
            point_vertice_indexes[key] = index
            index += 1
        for index_inside_group, (point, angles) in enumerate(group):
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = index
            vertices.append(point)
            if index_inside_group == 0:
                index += 1
                # edge case, where first point can't connect to the previous one
                continue
            if group_index == 0:
                index += 1
                continue
            previous_point_same_ring = group[index_inside_group - 1][0]
            previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
            same_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group][0]
            same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
            previous_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group - 1][0]
            previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
            triangle_1 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key]]
            faces.append(triangle_1)
            triangle_2 = [index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
            faces.append(triangle_2)
            if group_index == len(grouped_data_start) - 1 and index_inside_group > 0:
                # connect all the points to the top point
                triangle_4 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[top_point_start]]
                faces.append(triangle_4)
                if index_inside_group == len(group) - 1:
                    # zadnje vozlisce povezi s prejsnjim na vrhu
                    previous_point_same_ring = group[0][0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    triangle_5 = [index, point_vertice_indexes[top_point_start], point_vertice_indexes[previous_point_same_ring_key]]
                    faces.append(triangle_5)
            if index_inside_group == len(group) - 1:
                # zadnjo vozlisce povezi s prvim
                previous_point_same_ring = group[0][0]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                same_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                previous_point_previous_ring = grouped_data_start[group_index - 1][0][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                triangle_4 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_4)
                triangle_5 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_5)
            index += 1
    last_connected_index = 0
    for kroznica_index, p in enumerate(kroznica_start_points):
        # i1 = int(360 / angle_increment) - kroznica_index
        i1 = kroznica_index
        p = p[0]
        vertices.append(p)
        key = utils.dict_key_from_point(p)
        point_vertice_indexes[key] = index
        # if i1 < int(360 / angle_increment):
        if i1 > 0:
            skeleton_index = int(i1 * len(grouped_data_start[0]) / len(kroznica_start_points))
            triangle = [index, index - 1, point_vertice_indexes[grouped_data_start[0][skeleton_index][0]]]
            faces.append(triangle)
            if skeleton_index != last_connected_index:
                # make extra triangles
                for i in range(last_connected_index + 1, skeleton_index + 1):
                    triangle_1 = [index - 1, point_vertice_indexes[grouped_data_start[0][i - 1][0]], point_vertice_indexes[grouped_data_start[0][i][0]]]
                    faces.append(triangle_1)
                last_connected_index = skeleton_index
            if i1 == len(kroznica_start_points) - 1:
                triangle_1 = [index, point_vertice_indexes[grouped_data_start[0][skeleton_index][0]], point_vertice_indexes[kroznica_start_points[0][0]]]
                faces.append(triangle_1)
                triangle_2 = [point_vertice_indexes[kroznica_start_points[0][0]], point_vertice_indexes[grouped_data_start[0][skeleton_index][0]], point_vertice_indexes[grouped_data_start[0][0][0]]]
                faces.append(triangle_2)
        index += 1
    print('end points')
    grouped_data_end, top_point, kroznica_points = utils.group_edge_points_by_theta_extract_top_point(end_points, angle_increment=angle_increment)
    top_point = top_point[0]
    for group_index, group in grouped_data_end.items():
        if group_index == len(grouped_data_end) - 1:
            # before processing last ring, add top point to vertices
            vertices.append(top_point)
            key = (top_point[0], top_point[1], top_point[2])
            point_vertice_indexes[key] = index
            index += 1
        for index_inside_group, (point, angles) in enumerate(group):
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = index
            vertices.append(point)
            if index_inside_group == 0:
                index += 1
                # edge case, where first point can't connect to the previous one
                continue
            if group_index == 0:
                index += 1
                continue
            previous_point_same_ring = group[index_inside_group - 1][0]
            previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
            same_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group][0]
            same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
            previous_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group - 1][0]
            previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
            triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
            faces.append(triangle_1)
            triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
            faces.append(triangle_2)
            if group_index == len(grouped_data_end) - 1 and index_inside_group > 0:
                # connect all the points to the top point
                triangle_4 = [index, point_vertice_indexes[top_point], point_vertice_indexes[previous_point_same_ring_key]]
                faces.append(triangle_4)
                if index_inside_group == len(group) - 1:
                    # zadnje vozlisce povezi s prejsnjim na vrhu
                    previous_point_same_ring = group[0][0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    triangle_5 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[top_point]]
                    faces.append(triangle_5)
            if index_inside_group == len(group) - 1:
                # zadnjo vozlisce povezi s prvim
                previous_point_same_ring = group[0][0]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                same_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                previous_point_previous_ring = grouped_data_end[group_index - 1][0][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                triangle_4 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key]]
                faces.append(triangle_4)
                triangle_5 = [index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_5)
            index += 1
    last_connected_index = 0
    for kroznica_index, p in enumerate(kroznica_points):
        i1 = kroznica_index
        p = p[0]
        vertices.append(p)
        key = utils.dict_key_from_point(p)
        point_vertice_indexes[key] = index
        if i1 > 0:
            skeleton_index = int(i1 * len(grouped_data_end[0]) / len(kroznica_points))
            triangle = [index, point_vertice_indexes[grouped_data_end[0][skeleton_index][0]], index - 1]
            faces.append(triangle)
            if skeleton_index != last_connected_index:
                # make extra triangles
                for i in range(last_connected_index + 1, skeleton_index + 1):
                    triangle_1 = [index - 1, point_vertice_indexes[grouped_data_end[0][i][0]], point_vertice_indexes[grouped_data_end[0][i - 1][0]]]
                    faces.append(triangle_1)
                last_connected_index = skeleton_index
            if i1 == len(kroznica_points) - 1:
                triangle_1 = [index, point_vertice_indexes[kroznica_points[0][0]], point_vertice_indexes[grouped_data_end[0][skeleton_index][0]]]
                faces.append(triangle_1)
                triangle_2 = [point_vertice_indexes[kroznica_points[0][0]], point_vertice_indexes[grouped_data_end[0][0][0]], point_vertice_indexes[grouped_data_end[0][skeleton_index][0]]]
                faces.append(triangle_2)
        index += 1
    print('skeleton points')
    for ring, points_at_angle in skeleton_points.items():
        angles = list(points_at_angle.keys())
        angles.sort()
        for i, angle in enumerate(angles):
            point = points_at_angle[angle]
            vertices.append(point)
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = index
            if angle == 0:
                index += 1
                continue
            if 1 < ring:
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = skeleton_points[ring - 1][angle - angle_increment]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = skeleton_points[ring - 1][angle]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_1)
                triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_2)
                if i == len(angles) - 1:
                    # zadnjo vozlisce povezi s prvim
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    same_point_previous_ring = skeleton_points[ring - 1][angle]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    previous_point_previous_ring = skeleton_points[ring - 1][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    triangle_4 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key]]
                    faces.append(triangle_4)
                    triangle_5 = [index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                    faces.append(triangle_5)
            elif ring == 1:
                # last ring connects to start points
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = kroznica_start_points[i - 1][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = kroznica_start_points[i][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_1)
                triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_2)
                if i == len(angles) - 1:
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    previous_point_previous_ring = kroznica_start_points[0][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    same_point_previous_ring = kroznica_start_points[i][0]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                    faces.append(triangle_1)
                    triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                    faces.append(triangle_2)
            if ring == len(skeleton_points):
                # first ring connects to end points
                previous_point_same_ring = points_at_angle[angle - angle_increment]
                previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                previous_point_previous_ring = kroznica_points[i - 1][0]
                previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                same_point_previous_ring = kroznica_points[i][0]
                same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                triangle_1 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[previous_point_same_ring_key]]
                faces.append(triangle_1)
                triangle_2 = [index, point_vertice_indexes[same_point_previous_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_2)
                if i == len(angles) - 1:
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = utils.dict_key_from_point(previous_point_same_ring)
                    previous_point_previous_ring = kroznica_points[0][0]
                    previous_point_previous_ring_key = utils.dict_key_from_point(previous_point_previous_ring)
                    same_point_previous_ring = kroznica_points[i][0]
                    same_point_previous_ring_key = utils.dict_key_from_point(same_point_previous_ring)
                    triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                    faces.append(triangle_1)
                    triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                    faces.append(triangle_2)
            index += 1
    return vertices, faces


def create_parser():
    parser = argparse.ArgumentParser(description='Generate new mitochondria shape')
    parser.add_argument('-c', '--curvature', help='curvature of the generated shapes', nargs='*')
    parser.add_argument('-l', '--length', help='length of the generated shapes')
    parser.add_argument('-t', '--torsion', help='torsion of the generated shapes', nargs='*')
    parser.add_argument('-sl', '--sigma-length', help='value of sigma for length')
    parser.add_argument('-ss', '--sigma-skeleton', help='value of sigma for skeleton points')
    parser.add_argument('-sst', '--sigma-start', help='value of sigma for start points')
    parser.add_argument('-se', '--sigma-end', help='value of sigma for end points')
    parser.add_argument('-sc', '--sigma-curvature', help='value of sigma for curvature')
    parser.add_argument('-st', '--sigma-torsion', help='value of sigma for torsion')
    return parser


def validate_sigma_parameter(sigma):
    if sigma < 0 or sigma > 1:
        raise ValueError('sigma value must be between 0 and 1')

# todo: dodaj seed v argumente
if __name__ == '__main__':
    curvature, start, end, skeleton, lengths, direction_with_angles, torsions = utils.read_measurements_from_file('../measurements/learn/measurements.pkl')
    parser = create_parser()
    args = parser.parse_args()
    if args.curvature:
        if len(curvature.keys()) != len(args.curvature):
            raise Exception('number of curvature values must match the number of characteristic points')
        for i, values in curvature.items():
            curvature[i] = [float(args.curvature[i])]
    sigma = SigmaParameters()
    if args.torsion:
        if len(torsions.keys()) != len(args.torsion):
            raise Exception('number of torsion values must match the number of characteristic points')
        for i, values in curvature.items():
            torsions[i] = [float(args.torsion[i])]
    if args.sigma_length:
        value = float(args.sigma_length)
        validate_sigma_parameter(value)
        sigma.length = value
    if args.sigma_skeleton:
        value = float(args.sigma_skeleton)
        validate_sigma_parameter(value)
        sigma.skeleton_points = value
    if args.sigma_start:
        value = float(args.sigma_start)
        validate_sigma_parameter(value)
        sigma.start_points = value
    if args.sigma_end:
        value = float(args.sigma_end)
        validate_sigma_parameter(value)
        sigma.end_points = value
    if args.sigma_curvature:
        value = float(args.sigma_curvature)
        validate_sigma_parameter(value)
        sigma.curvature = value
    if args.sigma_torsion:
        value = float(args.sigma_torsion)
        validate_sigma_parameter(value)
        sigma.torsion = value
    if args.length:
        lengths = [float(args.length)]
    print('using sigma values:', sigma)
    # utils.plot_histograms_for_data(skeleton, start, end, curvature, torsions, lengths)
    # for i, values in torsions.items():
    #     torsions[i] = [0]
    sample_new_points(skeleton, start, end, curvature, direction_with_angles, lengths, torsions, random_seed=123, sigma=sigma)

