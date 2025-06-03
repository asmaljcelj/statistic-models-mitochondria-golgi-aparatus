import logging
import os

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError
from shapely.geometry import Polygon, LineString

import math_utils
import utils


def get_all_stack_size_of_learning_dataset(data_directory, num_split):
    # ugotovi najvecje stevilo cistern
    length = []
    added = 0
    for filename in os.listdir(data_directory):
        if '_ev' in filename:
            continue
        if added >= num_split:
            continue
        data = np.load(data_directory + '/' + filename, allow_pickle=True)
        length.append(len(data))
        added += 1
    return length, max(length)


def calculate_distances_to_landmark_points(group, direction_vectors):
    all_measurments = []
    for origin in group:
        max_distance_points = []
        try:
            points = group[origin]
            hull = ConvexHull(points)
            hull_points = []
            for vertex in hull.vertices:
                hull_points.append(hull.points[vertex])
            hull = Polygon(hull_points)
            for direction_vector in direction_vectors:
                ray_end = origin + direction_vector * 1000000
                ray = LineString([origin, ray_end])
                intersection = ray.intersection(hull.boundary)
                distances = np.linalg.norm(np.array(intersection.xy) - origin)
                max_distance_points.append(distances)
            all_measurments.append(max_distance_points)
        except QhullError:
            logging.exception("message")
            print('cisterna not added')
    return all_measurments

data_directory = 'ga_instances'

split_percentage, num_of_distance_vectors, instance_index = 82, 10, 0
num_split = int(len(os.listdir(data_directory)) / 2 * split_percentage / 100)
length, max_stack_size = get_all_stack_size_of_learning_dataset(data_directory, num_split)
distances = utils.initialize_list_of_lists(max_stack_size)
for filename in os.listdir(data_directory):
    # _ev v imenu datoteke vsebuje lastne vrednosti za ta primerek, tako da ga ne obdeluj posebej
    if '_ev' in filename:
        continue
    # nalozi obe datoteki
    data = np.load(data_directory + '/' + filename, allow_pickle=True)
    ev_filename = filename.replace('.npz', '_ev.npz')
    ev_data = np.load(data_directory + '/' + ev_filename, allow_pickle=True)
    ev_data = np.array([ev_data[key] for key in ev_data])
    print('processing file', filename)
    max_cisternas_of_instance = len(data)
    index = 0
    direction_vectors = math_utils.generate_direction_vectors(
        np.array([[1, 0],
            [0, 1]]), num_of_distance_vectors)
    # collect cisternae in list, then get them in the combined data structure
    all_points = []
    print('processing', len(data), 'cisternas')
    for i, cisterna_points in enumerate(data):
        print('processing cisterna', i)
        cis = data[cisterna_points]
        cis = np.array([np.array(lst).astype(float) for lst in cis], dtype=float)
        centers = utils.cisterna_volume_extraction(cis)
        print('found', len(centers), 'centers')
        measurements = calculate_distances_to_landmark_points(centers, direction_vectors)
        if len(measurements) == 0:
            print('no measurements, skipping this cisterna')
            continue
        all_points.append(measurements)
    # split the data
    if instance_index < num_split:
        distances_index = np.linspace(0, max_stack_size - 1, len(all_points)).astype(int)
        for k, measurement in enumerate(all_points):
            distances[distances_index[k]] += measurement
    else:
        # test data
        utils.save_ga_measurements_to_file('measurements_ga/testing/ga_' + str(instance_index) + '.pkl', all_points)
    instance_index += 1
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
utils.save_ga_measurements_to_file('measurements_ga/learn/measurements_ga.pkl', final_list)
