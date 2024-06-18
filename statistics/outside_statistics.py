import math

import numpy as np
from scipy.stats import gaussian_kde
from utils import plot_kde, plot_new_points, save_as_nii
from math_utils import rotate_vector, get_points_between_2_points

def calculate_average(skeleton_distances, start_distances, end_distances):
    average_skeleton_distances = {}
    # calculate average skeleton distances
    for skeleton_distance in skeleton_distances:
        distances = skeleton_distances[skeleton_distance]
        for i, distances_on_point in distances.items():
            if i not in average_skeleton_distances:
                average_skeleton_distances[i] = {}
            for j, distance in enumerate(distances_on_point):
                if j not in average_skeleton_distances[i]:
                    average_skeleton_distances[i][j] = [0, 0]
                average_skeleton_distances[i][j][0] += distance
                average_skeleton_distances[i][j][1] += 1
    # calculate averages
    for i in average_skeleton_distances:
        for j in average_skeleton_distances[i]:
            average_skeleton_distances[i][j][0] /= average_skeleton_distances[i][j][1]
    print(average_skeleton_distances)
    # calculate average start distances
    average_start_distances = {}
    for start_distance in start_distances:
        distances = start_distances[start_distance]
        for i, distances_on_point in distances.items():
            if i not in average_start_distances:
                average_start_distances[i] = [0, 0]
            average_start_distances[i][0] += distances_on_point
            average_start_distances[i][1] += 1
    for i in average_start_distances:
        average_start_distances[i][0] /= average_start_distances[i][1]
    print(average_start_distances)
    # calculate average end distances
    average_end_distances = {}
    for end_distance in end_distances:
        distances = end_distances[end_distance]
        for i, distances_on_point in distances.items():
            if i not in average_end_distances:
                average_end_distances[i] = [0, 0]
            average_end_distances[i][0] += distances_on_point
            average_end_distances[i][1] += 1
    for i in average_end_distances:
        average_end_distances[i][0] /= average_end_distances[i][1]
    print(average_end_distances)
    return average_skeleton_distances, average_start_distances, average_end_distances


def group_distances(skeleton_distances, start_distances, end_distances):
    print('grouping distances')
    skeleton = group_skeleton_data(skeleton_distances)
    start = group_both_ends_data(start_distances)
    end = group_both_ends_data(end_distances)
    return skeleton, start, end


def group_skeleton_data(data):
    grouped_data = {}
    for skeleton_distance in data:
        distances = data[skeleton_distance]
        for i, distances_on_point in distances.items():
            if i not in grouped_data:
                grouped_data[i] = {}
            for j, distance in enumerate(distances_on_point):
                if j not in grouped_data[i]:
                    grouped_data[i][j] = []
                grouped_data[i][j].append(distance)
    return grouped_data


def group_both_ends_data(data):
    grouped_data = {}
    for skeleton_distance in data:
        distances = data[skeleton_distance]
        for i, distance in distances.items():
            if i not in grouped_data:
                grouped_data[i] = []
            grouped_data[i].append(distance)
    return grouped_data


def sample_new_points(skeleton_distances, start_distances, end_distances, num_files):
    skeleton_points = np.column_stack((np.zeros(10), np.zeros(10), np.linspace(1, 10, 10, endpoint=True)))
    new_points = []
    for _ in range(num_files):
        new_points.append([])
    # start
    print('generating start points')
    for direction, distances in start_distances.items():
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        range_distances = np.linspace(min(distances), max(distances), 10000)
        pdf_estimation = kde(range_distances)
        # plot_kde(range, pdf_estimation)
        cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
        uniform_samples = np.random.uniform(0, 1, num_files)
        for i, sample in enumerate(uniform_samples):
            new_distance = np.interp(sample, cdf_estimation, range_distances)
            # calculate new boundary point in 3D space
            new_point = new_distance * (direction * np.array(-1))
            new_points[i].extend(get_points_between_2_points(np.array(([0, 0, 0])), new_point, math.ceil(new_distance)))
        # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
        # new_points.append(new_point)
    # end
    print('generating end points')
    last_point = skeleton_points[-1]
    for direction, distances in end_distances.items():
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        range_distances = np.linspace(min(distances), max(distances), 10000)
        pdf_estimation = kde(range_distances)
        # plot_kde(range, pdf_estimation)
        cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
        uniform_samples = np.random.uniform(0, 1, num_files)
        for i, sample in enumerate(uniform_samples):
            new_distance = np.interp(sample, cdf_estimation, range_distances)
            # calculate new boundary point in 3D space
            new_point = (new_distance * np.array(direction)) + last_point
            new_points[i].extend(get_points_between_2_points(last_point, new_point, math.ceil(new_distance)))
            # new_point = direction + last_point
            # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
            # new_points.append(new_point)
    # generate points on skeleton
    for point, distances_around in skeleton_distances.items():
        skeleton_point = skeleton_points[point]
        for angle, distances in distances_around.items():
            distances = np.array(distances)
            kde = gaussian_kde(distances)
            range_distances = np.linspace(min(distances), max(distances), 10000)
            pdf_estimation = kde(range_distances)
            # plot_kde(range, pdf_estimation)
            cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
            uniform_samples = np.random.uniform(0, 1, num_files)
            for i, sample in enumerate(uniform_samples):
                new_distance = np.interp(sample, cdf_estimation, range_distances)
                # calculate new boundary point in 3D space
                direction = rotate_vector(np.array([1, 0, 0]), angle, np.array([0, 0, 1]))
                new_point = new_distance * np.array(direction) + skeleton_point
                # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
                new_points[i].extend(get_points_between_2_points(skeleton_point, new_point, math.ceil(new_distance)))
                # new_points.append(new_point)

    for i in range(num_files):
        new_points[i] = np.array(new_points[i])
    # plot_new_points(new_points)
    save_as_nii(new_points)


