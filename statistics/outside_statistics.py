import numpy as np
from scipy.stats import gaussian_kde
from utils import plot_kde

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


def sample_new_points(skeleton_distances, start_distances, end_distances):
    for direction, distances in start_distances.items():
        estimation = estimate_distribution(distances)
        print(estimation)



def estimate_distribution(data):
    data = np.array(data)
    kde = gaussian_kde(data)
    range = np.linspace(min(data), max(data), 10000)
    pdf_estimation = kde(range)
    plot_kde(range, pdf_estimation)
    return pdf_estimation

