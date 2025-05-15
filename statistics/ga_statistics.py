import logging
import math
import os
from typing import final

import numpy as np
from scipy import ndimage
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from scipy.spatial import cKDTree
import matplotlib
from shapely.geometry import Polygon, LineString

import math_utils
import utils

gradient = [
 [0.0, 0.0, 1.0, 1.0],
 [0.0706, 0.0, 0.9255, 1.0],
 [0.1412, 0.0, 0.8549, 1.0],
 [0.2118, 0.0, 0.7843, 1.0],
 [0.2824, 0.0, 0.7137, 1.0],
 [0.3569, 0.0, 0.6392, 1.0],
 [0.4275, 0.0, 0.5686, 1.0],
 [0.4980, 0.0, 0.4980, 1.0],
 [0.5686, 0.0, 0.4275, 1.0],
 [0.6392, 0.0, 0.3569, 1.0],
 [0.7137, 0.0, 0.2824, 1.0],
 [0.7843, 0.0, 0.2118, 1.0],
 [0.8549, 0.0, 0.1412, 1.0],
 [0.9255, 0.0, 0.0706, 1.0],
 [1.0, 0.0, 0.0, 1.0]
]

data_directory = '../ga_instances'

def plot_instance(ga_instance):
    # matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, d in enumerate(ga_instance):
        cisternae = ga_instance[d]
        ax.scatter(cisternae[:, 0], cisternae[:, 1], cisternae[:, 2], c=[[gradient[i % len(gradient)]]], label=str(i))
    # ax.legend()
        plt.show()

def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.legend()
    plt.show()


def calculate_statistics_on_instance(ga_instance):
    data = {}
    for d in ga_instance:
        statistical_data = calculate_statistics(ga_instance[d])

def calculate_statistics(cistanae):
    # todo: thickness, shape
    # https: // www.cad - journal.net / files / vol_13 / CAD_13(2)_2016_199 - 207.pdf
    pass


def extract_edge_from_cisternae(cisternae):
    edge_points_dict = {}
    for c in cisternae:
        cisterna = cisternae[c]
        edge_p = extract_edge(cisterna)
        edge_points_dict[c] = edge_p
    return edge_points_dict


def check_if_point_is_object(point, cisterna):
    for c_point in cisterna:
        x, y, z = int(c_point[0]), int(c_point[1]), int(c_point[2])
        if point[0] == x and point[1] == y and point[2] == z:
            return True
    return False


def extract_edge(cisterna):
    cisterna_edge_points = []
    for c_point in cisterna:
        x, y, z = int(c_point[0]), int(c_point[1]), int(c_point[2])
        top = [x, y, z + 1]
        if check_if_point_is_object(top, cisterna):
            continue
        bot = [x, y, z - 1]
        if check_if_point_is_object(bot, cisterna):
            continue
        right = [x + 1, y, z]
        if check_if_point_is_object(right, cisterna):
            continue
        left = [x - 1, y, z]
        if check_if_point_is_object(left, cisterna):
            continue
        front = [x, y - 1, z]
        if check_if_point_is_object(front, cisterna):
            continue
        back = [x, y + 1, z]
        if check_if_point_is_object(back, cisterna):
            continue
        cisterna_edge_points.append(c_point)
        # top_right = [x + 1, y, z + 1]
        # top_left = [x - 1, y, z + 1]
        # top_front = [x, y - 1, z + 1]
        # top_back = [x, y + 1, z + 1]
        # top_right_front = [x + 1, y - 1, z + 1]
        # top_right_back = [x + 1, y + 1, z + 1]
        # top_left_front = [x - 1, y - 1, z + 1]
        # top_left_back = [x - 1, y + 1, z + 1]
        # front_left = [x - 1, y - 1, z]
        # front_right = [x + 1, y - 1, z]
        # back_left = [x - 1, y + 1, z]
        # back_right = [x + 1, y + 1, z]
        # bot_right = [x + 1, y, z - 1]
        # bot_left = [x - 1, y, z - 1]
        # bot_front = [x, y - 1, z - 1]
        # bot_back = [x, y + 1, z - 1]
        # bot_right_front = [x + 1, y - 1, z - 1]
        # bot_right_back = [x + 1, y + 1, z - 1]
        # bot_left_front = [x - 1, y - 1, z - 1]
        # bot_left_back = [x - 1, y + 1, z - 1]
    return np.array(cisterna_edge_points)


def get_furthest_point_in_direction(line_vector, points, origin, threshold=1.0):
    line_vector = np.array(line_vector)
    max_distance = -float('inf')
    furthest_point = None
    for point in points:
        t = np.dot(np.array([point[0], point[1], point[2]]), line_vector)
        closest_point = t * line_vector
        distance_to_line = np.linalg.norm(np.array([point[0], point[1], point[2]]) - closest_point)
        if distance_to_line <= threshold and t > 0:
            distance_to_origin = np.linalg.norm(point - origin)
            if distance_to_origin > max_distance:
                furthest_point = point
                max_distance = distance_to_origin
    # if furthest_point is None:
    #     print('no furthest point detected in direction', line_vector)
    #     utils.plot_new_points(points)
    # print('furthest point in direction', line_vector, 'is', furthest_point)
    return max_distance
    # line_vector = np.array(line_vector, dtype=np.float64)
    # tol = 1
    # max_distance = -1
    # for point in points:
    #     distance_from_line = math_utils.distance_between_point_and_line(point, line_vector)
    #     if abs(distance_from_line) < tol:
    #         if mode == 0 and point[0] < 0:
    #             continue
    #         if mode == 1 and point[1] < 0:
    #             continue
    #         if mode == 2 and point[0] > 0:
    #             continue
    #         if mode == 3 and point[1] > 0:
    #             continue
    #         if mode == 4 and (point[0] < 0 or point[1] < 0):
    #             continue
    #         if mode == 5 and (point[0] < 0 or point[1] > 0):
    #             continue
    #         if mode == 6 and (point[0] > 0 or point[1] < 0):
    #             continue
    #         if mode == 7 and (point[0] > 0 or point[1] > 0):
    #             continue
    #         distance = np.sqrt(np.square(point[0]) + np.square(point[1]) + np.square(point[2]))
    #         if distance > max_distance:
    #             max_distance = distance
    # return max_distance

def transform(coords):
    grid_size = np.max(coords, axis=0) + 1  # Add 1 to include max index

    # Initialize a voxel grid with zeros
    voxel_grid = np.zeros(grid_size, dtype=int)

    # Set the specified coordinates to 1
    voxel_grid[tuple(coords.T)] = 1
    return voxel_grid


# def calculate_distances_to_landmark_points(group, vectors, num_of_direction_vectors=8):
def calculate_distances_to_landmark_points(group, direction_vectors, testing=False):
    # direction_vectors = math_utils.generate_direction_vectors(vectors, num_of_direction_vectors)
    all_measurments = []
    # matplotlib.use('TkAgg')
    for origin in group:
        max_distance_points = []
        try:
            points = group[origin]
            # if testing:
            #     origin = (0, 0)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="3d")

            # Plot defining corner points
            # ax.plot(points.T[0], points.T[1], points.T[2], "ko")
            hull = ConvexHull(points)
            hull_points = []
            for vertex in hull.vertices:
                hull_points.append(hull.points[vertex])
            hull = Polygon(hull_points)
            # list = []
            # for face in hull.simplices:
            #     list.append(np.array([face[0], face[1], 0]))
            # list = np.array(list)
            # mesh = trimesh.Trimesh(vertices=hull.vertices, faces=list)
            # for s in hull.simplices:
            #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            #     ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")
            # ax.plot([origin[0]], [origin[1]], [origin[2]], "ko")
            # ax.quiver(*origin, *direction_vectors[0], color='black')
            # l, _, _ = mesh.ray.intersects_location(
            #     ray_origins=[origin], ray_directions=[direction_vectors[0]]
            # )
            # ax.scatter([l[0][0]], [l[0][1]], [l[0][2]], color='orange')
            # plt.show()

            for direction_vector in direction_vectors:
                ray_end = origin + direction_vector * 1000000
                ray = LineString([origin, ray_end])
                intersection = ray.intersection(hull.boundary)
                # origin = [origin[0], origin[1], 0]
                # direction_vector = [direction_vector[0], direction_vector[1], 0]
                # locations, _, _ = mesh.ray.intersects_location(
                #     ray_origins=[origin], ray_directions=[direction_vector]
                # )
                # if locations.shape[0] > 0:
                distances = np.linalg.norm(np.array(intersection.xy) - origin)
                # max_distance = get_furthest_point_in_direction(direction_vector, points, origin)
                # if max_distance == -float('inf'):
                #     max_distance = 0
                # if math.isnan(max_distance):
                #     print()
                #     if distances <= 0:
                #         print()
                max_distance_points.append(distances)
                # else:
                #     print()
            all_measurments.append(max_distance_points)
        except QhullError:
            # return [0 for _ in direction_vectors]
            logging.exception("message")
            # all_measurments.append([0 for _ in direction_vectors])
            print('cisterna not added')
    return all_measurments
    # distance in the x axis
    # line_vector = [1.0, 0.0, 0.0]
    # max_distance_x = get_furthest_point_in_direction(line_vector, points, 0)
    # # distance in the y axis
    # line_vector = [0.0, 1.0, 0.0]
    # max_distance_y = get_furthest_point_in_direction(line_vector, points, 1)
    # # distance in the -x axis
    # # distance in the -x axis
    # line_vector = [-1.0, 0.0, 0.0]
    # line_vector = [-1.0, 0.0, 0.0]
    # max_distance_minus_x = get_furthest_point_in_direction(line_vector, points, 2)
    # # distance in the -y axis
    # line_vector = [0.0, -1.0, 0.0]
    # max_distance_minus_y = get_furthest_point_in_direction(line_vector, points, 3)
    # # distance in the 1st quadrant
    # line_vector = [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]
    # max_distance_first = get_furthest_point_in_direction(line_vector, points, 4)
    # # distance in the 2nd quadrant
    # line_vector = [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]
    # max_distance_second = get_furthest_point_in_direction(line_vector, points, 5)
    # # distance in the 3rd quadrant
    # line_vector = [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]
    # max_distance_third = get_furthest_point_in_direction(line_vector, points, 6)
    # # distance in the 4th quadrant
    # line_vector = [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]
    # max_distance_fourth = get_furthest_point_in_direction(line_vector, points, 7)
    # return max_distance_x, max_distance_y, max_distance_minus_x, max_distance_minus_y, max_distance_first, max_distance_second, max_distance_third, max_distance_fourth


if __name__ == '__main__':
    split_percentage = 82
    num_split = int(len(os.listdir(data_directory)) / 2 * split_percentage / 100)
    length = []
    # ugotovi najvecje stevilo cistern
    added = 0
    for filename in os.listdir(data_directory):
        if '_ev' in filename:
            continue
        if added >= num_split:
            continue
        data = np.load(data_directory + '/' + filename, allow_pickle=True)
        length.append(len(data))
        added += 1
    distances = []
    max_cisternas = max(length)
    for i in range(max_cisternas):
        distances.append([])

    num_of_distance_vectors = 10
    j = 0
    for filename in os.listdir(data_directory):
        if '_ev' in filename:
            continue
        data = np.load(data_directory + '/' + filename, allow_pickle=True)
        ev_filename = filename.replace('.npz', '_ev.npz')
        ev_data = np.load(data_directory + '/' + ev_filename, allow_pickle=True)
        ev_data = np.array([ev_data[key] for key in ev_data])
        print('processing file', filename)
        # plot_instance(data)
        # edge_points = extract_edge_from_cisternae(data)
        # plot_instance(edge_points)
        max_cisternas_of_instance = len(data)
        index = 0
        # utils.plot_3d(data[cisterna_points])
        # direction_vectors = math_utils.generate_direction_vectors(ev_data, num_of_distance_vectors)
        direction_vectors = math_utils.generate_direction_vectors(
            np.array([[1, 0],
                [0, 1]]), num_of_distance_vectors)
        # collect cisteernas iin on llist, than get them in the combinied daaata astructure
        all_points = []
        # utils.plot_3_base_vectors_and_direction_vectors(ev_data[0], ev_data[1], ev_data[2], direction_vectors)
        print('processing', len(data), 'cisternas')
        for i, cisterna_points in enumerate(data):
            print('processing cisterna', i)
            cis = data[cisterna_points]
            cis = np.array([np.array(lst).astype(float) for lst in cis], dtype=float)
            # utils.plot_vectors_and_points_vector_rotation(direction_vectors, cis)
            # izracunaj distanco do "landmark" tock za vsako cisterno
            # x, y, minus_x, minus_y, first, second, third, fourth = calculate_distances_to_landmark_points(cis)
            # print('processing cisterna')
            centers = utils.cisterna_volume_extraction(cis)
            # if len(centers) > 1:
            #     utils.plot_grouped_points(centers)
            # if len(centers) >= 10:
            #     utils.plot_grouped_points(centers)
            print('found', len(centers), 'centers')
            # todo: za vsak center izracunaj Convex hull??? in naredi meritev
            # utils.plot_3d(cis)
            # grid = transform(cis)
            # new_image = ndimage.binary_erosion(grid).astype(grid.dtype)
            # utils.plot_2_sets_of_points(cis, new_image)
            # mean = np.mean(cis, axis=0)
            # measurements = calculate_distances_to_landmark_points(centers, ev_data, num_of_distance_vectors)
            measurements = calculate_distances_to_landmark_points(centers, direction_vectors)
            if len(measurements) == 0:
                print('no measurements, skipping this cisterna')
                continue
            #     print()

            # distances_index = int(len(distances) / max_cisternas_of_instance * index)
            # distances_index = np.linspace(0, len(distances) - 1, max_cisternas_of_instance).astype(int)[index]
            # print('inserting at index', distances_index)
            # distances[distances_index].append([x, first, y, second, minus_x, third, minus_y, fourth])
            # distances[distances_index] += measurements
            all_points.append(measurements)
            # index += 1
        if j < num_split:
            distances_index = np.linspace(0, max_cisternas - 1, len(all_points)).astype(int)

            for k, measurement in enumerate(all_points):
                distances[distances_index[k]] += measurement
        else:
            # test data
            utils.save_ga_measurements_to_file('../measurements_ga/testing/ga_' + str(j) + '.pkl', all_points)
        j += 1
    # pogrupiraj distance med sabo
    final_list = []
    final_list.append([length, num_of_distance_vectors])
    for i, cisterna_distances in enumerate(distances):
        transposed = list(zip(*cisterna_distances))
        result = [list(t) for t in transposed]
        final_list.append([])
        for r in result:
            final_list[i + 1].append(r)
    print('done')
    # utils.plot_histograms_for_ga_data(length, final_list[1][0], final_list[max_cisternas][0])
    utils.save_ga_measurements_to_file('../measurements_ga/learn/measurements_ga.pkl', final_list)
    # average_object_points, points_dict = generate_average(length, distances)
    # plot_points(average_object_points)
    # todo: daj objekt v mesh; zgeneriraj še več referenčnih točk; dodaj kak parameter (npr. št. cistern, standardni odklon)
    # vertices, faces = generate_mesh(points_dict)
    # utils.generate_obj_file(vertices, faces, f'ga_average.obj')

