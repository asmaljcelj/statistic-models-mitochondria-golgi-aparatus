import os

import utils
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import alphashape

import math_utils

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


def get_furthest_point_in_direction(line_vector, points, mode):
    line_vector = np.array(line_vector, dtype=np.float64)
    tol = 1
    max_distance = -1
    for point in points:
        distance_from_line = math_utils.distance_between_point_and_line(point, line_vector)
        if abs(distance_from_line) < tol:
            if mode == 0 and point[0] < 0:
                continue
            if mode == 1 and point[1] < 0:
                continue
            if mode == 2 and point[0] > 0:
                continue
            if mode == 3 and point[1] > 0:
                continue
            if mode == 4 and (point[0] < 0 or point[1] < 0):
                continue
            if mode == 5 and (point[0] < 0 or point[1] > 0):
                continue
            if mode == 6 and (point[0] > 0 or point[1] < 0):
                continue
            if mode == 7 and (point[0] > 0 or point[1] > 0):
                continue
            distance = np.sqrt(np.square(point[0]) + np.square(point[1]) + np.square(point[2]))
            if distance > max_distance:
                max_distance = distance
    return max_distance



def calculate_distances_to_landmark_points(points):
    # distance in the x axis
    line_vector = [1.0, 0.0, 0.0]
    max_distance_x = get_furthest_point_in_direction(line_vector, points, 0)
    # distance in the y axis
    line_vector = [0.0, 1.0, 0.0]
    max_distance_y = get_furthest_point_in_direction(line_vector, points, 1)
    # distance in the -x axis
    line_vector = [-1.0, 0.0, 0.0]
    max_distance_minus_x = get_furthest_point_in_direction(line_vector, points, 2)
    # distance in the -y axis
    line_vector = [0.0, -1.0, 0.0]
    max_distance_minus_y = get_furthest_point_in_direction(line_vector, points, 3)
    # distance in the 1st quadrant
    line_vector = [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]
    max_distance_first = get_furthest_point_in_direction(line_vector, points, 4)
    # distance in the 2nd quadrant
    line_vector = [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]
    max_distance_second = get_furthest_point_in_direction(line_vector, points, 5)
    # distance in the 3rd quadrant
    line_vector = [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]
    max_distance_third = get_furthest_point_in_direction(line_vector, points, 6)
    # distance in the 4th quadrant
    line_vector = [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]
    max_distance_fourth = get_furthest_point_in_direction(line_vector, points, 7)
    return max_distance_x, max_distance_y, max_distance_minus_x, max_distance_minus_y, max_distance_first, max_distance_second, max_distance_third, max_distance_fourth

def populate_instances(distances, starting_point):
    points = []
    for i, distance in enumerate(distances):
        if i == 0:
            points.append([distance, 0, starting_point[2]])
        if i == 1:
            points.append([0, distance, starting_point[2]])
        if i == 2:
            points.append([-distance, 0, starting_point[2]])
        if i == 3:
            points.append([0, -distance, starting_point[2]])
        if i == 4:
            scaled_vector = [1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
            points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
        if i == 5:
            scaled_vector = [1 / np.sqrt(2) * distance, -1 / np.sqrt(2) * distance, 0.0]
            points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
        if i == 6:
            scaled_vector = [-1 / np.sqrt(2) * distance, 1 / np.sqrt(2) * distance, 0.0]
            points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
        if i == 7:
            scaled_vector = [-1 / np.sqrt(2) * distance, -1 / np.sqrt(2) * distance, 0.0]
            points.append([scaled_vector[0], scaled_vector[1], starting_point[2]])
    return np.array(points)


def generate_average(cisternae_sizes, distances):
    # figure out number of cisterna stacks
    cisternae_sizes = np.array(cisternae_sizes)
    num_of_cisternas = int(np.mean(cisternae_sizes))
    # generate landmark points at each step
    # first generate cisterna 1
    average_distances_zero = np.mean(distances[0], axis=0)
    average_distances_max = np.mean(distances[len(distances) - 1], axis=0)
    # zdruzi posamezne meritve med sabo
    num_of_remaining = num_of_cisternas - 2
    num_per_stack = int(len(distances) / num_of_remaining)
    final_object, final_object_dict = [], {}
    zero = populate_instances(average_distances_zero, [0, 0, 0])
    final_object_dict[0] = zero
    final_object.extend(zero)
    for i in range(1, num_of_cisternas - 1):
        # zdruzi
        starting_index = (i - 1) * num_per_stack + 1
        ending_index = i * num_per_stack
        combined_data = []
        for j in range(starting_index, ending_index + 1):
            combined_data.extend(distances[j])
        combined_data = np.array(combined_data)
        average_distances = np.mean(combined_data, axis=0)
        points = populate_instances(average_distances, [0, 0, i])
        final_object_dict[i] = points
        final_object.extend(points)
    final = populate_instances(average_distances_max, [0, 0, num_of_cisternas - 1])
    final_object_dict[num_of_cisternas - 1] = final
    final_object.extend(final)
    return np.array(final_object), final_object_dict


def generate_mesh(points):
    index = 0
    vertices, faces = [], []
    for i, c in enumerate(points):
        cisterna_points = points[c]
        if i == 0 or i == len(points) - 1:
            # ce smo na dnu ali na vrhu, generiraj se eno tocko v sredini in poveži v mesh
            print()
            center = np.mean(cisterna_points, axis=0)
            vertices.append(center)
            center_index = index
            index += 1
            for j, point in enumerate(cisterna_points):
                vertices.append(point)
                if j > 0:
                    faces.append([index, center_index, index - 1])
                index += 1
            if i == 0:
                continue
        for j, point in enumerate(cisterna_points):
            vertices.append(point)
            if j > 0:
                same_ring_previous = index - 1
                previous_ring_same = index - len(cisterna_points)
                previous_ring_previous = index - len(cisterna_points) - 1
                faces.append([index, previous_ring_same, previous_ring_previous])
                faces.append([index, same_ring_previous, previous_ring_previous])
            index += 1
    return vertices, faces


length = []
# ugotovi najvecje stevilo cistern
for filename in os.listdir(data_directory):
    data = np.load(data_directory + '/' + filename, allow_pickle=True)
    length.append(len(data))

distances = []
max_cisternas = max(length)
for i in range(max_cisternas):
    distances.append([])


for filename in os.listdir(data_directory):
    data = np.load(data_directory + '/' + filename, allow_pickle=True)
    print('processing file', filename)
    # plot_instance(data)
    # edge_points = extract_edge_from_cisternae(data)
    # plot_instance(edge_points)
    max_cisternas_of_instance = len(data)
    index = 0
    for cisterna_points in data:
        cis = data[cisterna_points]
        # izracunaj distanco do "landmark" tock za vsako cisterno
        x, y, minus_x, minus_y, first, second, third, fourth = calculate_distances_to_landmark_points(cis)
        distances_index = int(len(distances) / max_cisternas_of_instance * index)
        distances[distances_index].append([x, first, y, second, minus_x, third, minus_y, fourth])
        index += 1
average_object_points, points_dict = generate_average(length, distances)
# plot_points(average_object_points)
# todo: daj objekt v mesh; zgeneriraj še več referenčnih točk; dodaj kak parameter (npr. št. cistern, standardni odklon)
vertices, faces = generate_mesh(points_dict)
utils.generate_obj_file(vertices, faces, f'ga_average.obj')

