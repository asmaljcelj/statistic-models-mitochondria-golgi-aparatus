import math

import numpy as np
import trimesh
from scipy.stats import gaussian_kde

import math_utils
import utils


def calculate_new_skeleton_point(previous_point, curvature_value, distance_between_points):
    radius_of_circle = abs(distance_between_points / curvature_value)
    center_of_circle = np.copy(previous_point)
    if curvature_value < 0:
        center_of_circle[1] -= radius_of_circle
        return find_new_skeleton_point(center_of_circle, previous_point, radius_of_circle, distance_between_points, 1)
    elif curvature_value > 0:
        center_of_circle[1] += radius_of_circle
        return find_new_skeleton_point(center_of_circle, previous_point, radius_of_circle, distance_between_points, -1)
    else:
        # curvature value is 0 -> take only straight point
        return [0, previous_point[1], previous_point[2] + distance_between_points]


def find_new_skeleton_point(center, current_point, radius, distance_between_points, direction):
    theta = distance_between_points / radius
    phi = np.pi / 2 * utils.get_sign_of_number(current_point[1] - center[1])
    z1 = center[2] + radius * math.cos(phi + theta)
    if direction == -1:
        z1 = center[2] + radius * math.cos(phi - theta)
    y1 = center[1] + radius * math.sin(phi + theta)
    return [0, y1, z1]


def sample_new_points(skeleton_distances, start_distances, end_distances, curvature, num_files, direction_with_angles):
    # calculate skeleton points based on curvature
    print('finding new skeleton points')
    new_points, total_skeleton_points, edge_points = [], [], []
    start_edge_points, end_edge_points, skeleton_points = [], [], []
    skeleton_points_dict = {}
    start_points_dict, end_points_dict = {}, {}
    for i in range(num_files):
        new_points.append([])
        start_edge_points.append([])
        end_edge_points.append([])
        skeleton_points.append([])
        total_skeleton_points.append(np.array([[0, 0, 0]], dtype=float))
        edge_points.append([])
        skeleton_points_dict[i] = {}
        start_points_dict[i] = {}
        end_points_dict[i] = {}
    for c in curvature:
        data = curvature[c]
        kde = gaussian_kde(data)
        new_curvatures = kde.resample(num_files)
        for i, sample in enumerate(new_curvatures):
            new_curvature = sample[0]
            total_skeleton_points[i] = np.append(total_skeleton_points[i], [calculate_new_skeleton_point(total_skeleton_points[i][-1], new_curvature, 1)], axis=0)
    # generate points on skeleton
    print('generating skeleton points')
    for point, distances_around in skeleton_distances.items():
        for angle, distances in distances_around.items():
            distances = np.array(distances)
            kde = gaussian_kde(distances)
            new_distances = kde.resample(num_files)
            for i, sample in enumerate(new_distances):
                if point not in skeleton_points_dict[i]:
                    skeleton_points_dict[i][point] = {}
                new_distance = sample[0]
                # calculate new boundary point in 3D space
                direction = math_utils.rotate_vector(np.array([1, 0, 0]), angle, np.array([0, 0, 1]))
                new_point = new_distance * np.array(direction) + total_skeleton_points[i][point]
                skeleton_points_dict[i][point][angle % 360] = new_point
    # start
    print('generating start points')
    for direction, distances in start_distances.items():
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        new_distances = kde.resample(num_files)
        theta, phi = direction_with_angles[direction]
        for i, sample in enumerate(new_distances):
            new_distance = sample[0]
            # new_distance *= 10
            # calculate new boundary point in 3D space
            normal_start = total_skeleton_points[i][-1] - total_skeleton_points[i][-2]
            R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal_start)
            new_direction = np.dot(R, direction)
            # new_point = new_distance * (new_direction * np.array(-1)) + total_skeleton_points[i][-1]
            new_point = new_distance * new_direction + total_skeleton_points[i][-1]
            new_point_key = (new_point[0], new_point[1], new_point[2])
            # edge_points[i].append(new_point)
            # new_points[i].extend(get_points_between_2_points(np.array(([0, 0, 0])), new_point, math.ceil(new_distance)))
            # new_points[i].append(new_point)
            # start_edge_points[i].append(new_point)
            start_points_dict[i][new_point_key] = (theta, phi)
        # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
        # new_points.append(new_point)
    # # end
    print('generating end points')
    for direction, distances in end_distances.items():
        distances.sort()
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        new_distances = kde.resample(num_files)
        theta, phi = direction_with_angles[direction]
        for i, sample in enumerate(new_distances):
            # new_distance = np.interp(sample, cdf_estimation, range_distances)
            new_distance = sample[0]
            # new_distance *= 10
            # calculate new boundary point in 3D space
            normal_start = total_skeleton_points[i][1] - total_skeleton_points[i][0]
            R = math_utils.get_rotation_matrix(np.array([0, 0, 1]), normal_start)
            new_direction = np.dot(R, direction)
            new_point = new_distance * (new_direction * np.array(-1))
            new_point_key = (new_point[0], new_point[1], new_point[2])
            end_points_dict[i][new_point_key] = (theta, phi)
            # edge_points[i].append(new_point)
            # new_points[i].extend(
            #     get_points_between_2_points(total_skeleton_points[i][-1], new_point, math.ceil(new_distance)))
            # new_points[i].append(new_point)
            # end_edge_points[i].append(new_point)
            # new_point = direction + last_point
            # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
            # new_points.append(new_point)
    for i in range(num_files):
        vertices, faces = generate_mesh(skeleton_points_dict[i], start_points_dict[i], end_points_dict[i])
        utils.generate_obj_file(vertices, faces, f'test_{i}.obj')
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        smooth = trimesh.smoothing.filter_humphrey(tri_mesh)
        utils.generate_obj_file(smooth.vertices, smooth.faces, f'smooth_{i}.obj')
    # save_as_nii_layers(start_edge_points, end_edge_points, skeleton_points)
    # save_as_nii(new_points)
    # save_as_normal_file(new_points, 1)


def generate_mesh(skeleton_points, start_points, end_points):
    vertices, faces = [], []
    # skeleton points
    point_vertice_indexes = {}
    index = 0
    points_on_ring = -1
    for ring, points_at_angle in skeleton_points.items():
        angles = list(points_at_angle.keys())
        angles.sort()
        points_on_ring = len(angles)
        for angle in angles:
            point = points_at_angle[angle]
            vertices.append(point)
            key = (point[0], point[1], point[2])
            point_vertice_indexes[key] = index
            # todo: ali handlem zadnji primer, da zadnji kot povezem s prvim????
            if ring > 1 and angle > 0:
                # triangle_1 = [point, points_at_angle[angle - 1], skeleton_points[ring - 1][angle - 1]]
                previous_point_same_ring = points_at_angle[angle - 1]
                previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                previous_point_previous_ring = skeleton_points[ring - 1][angle - 1]
                previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
                same_point_previous_ring = skeleton_points[ring - 1][angle]
                same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
                triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_1)
                # triangle_2 = [point, skeleton_points[ring - 1][angle - 1], skeleton_points[ring - 1][angle]]
                triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_2)
                if angle == len(angles) - 1:
                    # zadnjo vozlisce povezi s prvim
                    previous_point_same_ring = points_at_angle[0]
                    previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                    same_point_previous_ring = skeleton_points[ring - 1][angle]
                    same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
                    previous_point_previous_ring = skeleton_points[ring - 1][0]
                    previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
                    triangle_4 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                    faces.append(triangle_4)
                    triangle_5 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                    faces.append(triangle_5)
            index += 1
    # start points
    print('start points')
    grouped_data_start, top_point_start = utils.group_edge_points_by_theta_extract_top_point(start_points)
    top_point_start = top_point_start[0]
    for group_index, group in grouped_data_start.items():
        interval = points_on_ring / len(group)
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
                # edge case, where first point in a layer can't connect to the previous one
                continue
            if group_index == 0:
                # todo: connect with skeleton points
                previous_point_same_ring = group[index_inside_group - 1][0]
                previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                skeleton_angle_same_point = round(interval * index_inside_group)
                same_point_on_skeleton = skeleton_points[list(skeleton_points)[-1]][skeleton_angle_same_point]
                same_point_on_skeleton_key = (same_point_on_skeleton[0], same_point_on_skeleton[1], same_point_on_skeleton[2])
                skeleton_angle_previous_point = round(interval * (index_inside_group - 1))
                previous_point_on_skeleton = skeleton_points[list(skeleton_points)[-1]][skeleton_angle_previous_point]
                previous_point_on_skeleton_key = (previous_point_on_skeleton[0], previous_point_on_skeleton[1], previous_point_on_skeleton[2])
                triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_on_skeleton_key]]
                faces.append(triangle_1)
                triangle_2 = [index, point_vertice_indexes[previous_point_on_skeleton_key], point_vertice_indexes[same_point_on_skeleton_key]]
                faces.append(triangle_2)
                index += 1
                continue
            # todo: handle so last one connects to first one
            previous_point_same_ring = group[index_inside_group - 1][0]
            previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
            same_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group][0]
            same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
            previous_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group - 1][0]
            previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
            triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
            faces.append(triangle_1)
            triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
            faces.append(triangle_2)
            if group_index == len(grouped_data_start) - 1 and index_inside_group > 0:
                # connect all the points to the top point
                triangle_3 = [index, point_vertice_indexes[top_point_start], point_vertice_indexes[previous_point_same_ring_key]]
                faces.append(triangle_3)
            if index_inside_group == len(group) - 1:
                # zadnjo vozlisce povezi s prvim
                previous_point_same_ring = group[0][0]
                previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                same_point_previous_ring = grouped_data_start[group_index - 1][index_inside_group][0]
                same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
                previous_point_previous_ring = grouped_data_start[group_index - 1][0][0]
                previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
                triangle_4 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_4)
                triangle_5 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_5)
            index += 1
    print('end points')
    grouped_data_end, top_point = utils.group_edge_points_by_theta_extract_top_point(end_points)
    top_point = top_point[0]
    for group_index, group in grouped_data_end.items():
        interval = points_on_ring / len(group)
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
                previous_point_same_ring = group[index_inside_group - 1][0]
                previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                skeleton_angle_same_point = round(interval * index_inside_group)
                same_point_on_skeleton = skeleton_points[1][skeleton_angle_same_point]
                same_point_on_skeleton_key = (same_point_on_skeleton[0], same_point_on_skeleton[1], same_point_on_skeleton[2])
                skeleton_angle_previous_point = round(interval * (index_inside_group - 1))
                previous_point_on_skeleton = skeleton_points[1][skeleton_angle_previous_point]
                previous_point_on_skeleton_key = (previous_point_on_skeleton[0], previous_point_on_skeleton[1], previous_point_on_skeleton[2])
                triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_on_skeleton_key]]
                faces.append(triangle_1)
                triangle_2 = [index, point_vertice_indexes[previous_point_on_skeleton_key], point_vertice_indexes[same_point_on_skeleton_key]]
                faces.append(triangle_2)
                index += 1
                continue
            # todo: handle so last one connects to first one
            previous_point_same_ring = group[index_inside_group - 1][0]
            previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
            same_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group][0]
            same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
            previous_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group - 1][0]
            previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
            triangle_1 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
            faces.append(triangle_1)
            triangle_2 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
            faces.append(triangle_2)
            if group_index == len(grouped_data_end) - 1 and index_inside_group > 0:
                # connect all the points to the top point
                triangle_3 = [index, point_vertice_indexes[top_point], point_vertice_indexes[previous_point_same_ring_key]]
                faces.append(triangle_3)
            if index_inside_group == len(group) - 1:
                # zadnjo vozlisce povezi s prvim
                previous_point_same_ring = group[0][0]
                previous_point_same_ring_key = (previous_point_same_ring[0], previous_point_same_ring[1], previous_point_same_ring[2])
                same_point_previous_ring = grouped_data_end[group_index - 1][index_inside_group][0]
                same_point_previous_ring_key = (same_point_previous_ring[0], same_point_previous_ring[1], same_point_previous_ring[2])
                previous_point_previous_ring = grouped_data_end[group_index - 1][0][0]
                previous_point_previous_ring_key = (previous_point_previous_ring[0], previous_point_previous_ring[1], previous_point_previous_ring[2])
                triangle_4 = [index, point_vertice_indexes[previous_point_same_ring_key], point_vertice_indexes[previous_point_previous_ring_key]]
                faces.append(triangle_4)
                triangle_5 = [index, point_vertice_indexes[previous_point_previous_ring_key], point_vertice_indexes[same_point_previous_ring_key]]
                faces.append(triangle_5)
            index += 1

    # end points
    return vertices, faces
