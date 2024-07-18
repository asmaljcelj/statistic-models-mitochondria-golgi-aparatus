import csv
import datetime
import math
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
from sympy.integrals.intpoly import point_sort
from collections import Counter


def print_log(message):
    ct = datetime.datetime.now()
    print(ct, ":", message)


def plot_save_result(num_of_points, bezier_curve, original_points, arc_length_approx, number_of_plots, filename):
    # plot the result and save it
    if num_of_points == number_of_plots:
        time.sleep(15)
        num_of_points = 0
    bezier_curve = np.array(bezier_curve)
    arc_length_approx = np.array(arc_length_approx)
    num_of_points += 1
    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        bezier_curve[:, 0],  # x-coordinates.
        bezier_curve[:, 1],  # y-coordinates.
        bezier_curve[:, 2],  # y-coordinates.
        'o:',
        label='Bezier curve'
    )
    ax.plot(
        arc_length_approx[:, 0],  # x-coordinates.
        arc_length_approx[:, 1],  # y-coordinates.
        arc_length_approx[:, 2],  # y-coordinates.
        'ro:',  # Styling (red, circles, dotted).
        label='Arc length parametrization'
    )
    # new calculated points
    # ax.plot(
    #     original_points[:, 0],  # x-coordinates.
    #     original_points[:, 1],  # y-coordinates.
    #     original_points[:, 2],  # y-coordinates.
    #     'yo:',  # Styling (yellow, circles, dotted).
    #     label='Original skeleton'
    # )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.view_init(50, 20)
    plt.title(filename)
    plt.savefig('../plots/' + filename + '.png')
    plt.close()


def plot_sampling_with_shape(shape, sampled_points, skeleton, parametrized_points):
    matplotlib.use('TkAgg')
    colors = [
        (0.0, 1.0, 0.0),
         (0.1111111111111111, 0.8861111111111111, 0.0),
         (0.2222222222222222, 0.7722222222222221, 0.0),
         (0.3333333333333333, 0.6583333333333333, 0.0),
         (0.4444444444444444, 0.5444444444444444, 0.0),
         (0.5555555555555556, 0.4305555555555556, 0.0),
         (0.6666666666666666, 0.31666666666666665, 0.0),
         (0.7777777777777777, 0.20277777777777775, 0.0),
         (0.8888888888888888, 0.0888888888888889, 0.0),
         (1.0, 0.0, 0.0)
    ]
    count = 0
    # todo: verificiraj, ali je okej samplano!!!
    # sampled_points = np.array(sampled_points)
    skeleton = np.array(skeleton)
    parametrized_points = np.array(parametrized_points)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.voxels(shape, facecolors=[0, 0, 1, 0.2])
    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], 'yo:')
    ax.plot(parametrized_points[:, 0], parametrized_points[:, 1], parametrized_points[:, 2], 'gx:')
    for i in sampled_points:
        list = np.array(sampled_points[i])
        # ax.plot(list[:, 0], list[:, 1], list[:, 2], 'o:', color=colors[count])
        ax.plot(list[:, 0], list[:, 1], list[:, 2], 'co:')
        count += 1
    ax.view_init(azim=-125, elev=-40)
    plt.show()


def read_file_collect_points(filename, base_folder):
    if filename.endswith('.nii'):
        return None
    file_path = base_folder + filename
    points = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            points.append([int(row[0]), int(row[1]), int(row[2])])
        points = np.array(points)
    return points


def read_nii_file(base_folder, filename):
    nib_image = nib.load(base_folder + filename)
    image_data = nib_image.get_fdata()
    return np.array(image_data)


def plot_kde(range, data):
    plt.plot(range, data)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.show()


def plot_new_points(new_points):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(new_points[:, 0], new_points[:, 1], new_points[:, 2], 'yo')
    ax.view_init(azim=-125, elev=-40)
    plt.show()


def save_as_nii(set_of_points):
    for i, points in enumerate(set_of_points):
        affine = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        z_points = [p[2] for p in points]
        min_x = math.trunc(min(x_points))
        max_x = math.trunc(max(x_points))
        min_y = math.trunc(min(y_points))
        max_y = math.trunc(max(y_points))
        min_z = math.trunc(min(z_points))
        max_z = math.trunc(max(z_points))
        final_instance_object = np.zeros((max_x - min_x + 10, max_y - min_y + 10, max_z - min_z + 10))
        for point in points:
            x = math.trunc(point[0])
            y = math.trunc(point[1])
            z = math.trunc(point[2])
            final_instance_object[x - min_x + 5][y - min_y + 5][z - min_z + 5] = 255
        new_image = nib.Nifti1Image(final_instance_object, affine)
        new_filename = '../results/generated_shape_' + str(i)
        nib.save(new_image, new_filename)


def save_as_nii_layers(start, end, skeleton):
    for i, start_points in enumerate(start):
        end_points = end[i]
        skeleton_points = skeleton[i]
        affine = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        x_points = [p[0] for p in start_points] + [p[0] for p in end_points] + [p[0] for p in skeleton_points]
        y_points = [p[1] for p in start_points] + [p[1] for p in end_points] + [p[1] for p in skeleton_points]
        z_points = [p[2] for p in start_points] + [p[2] for p in end_points] + [p[2] for p in skeleton_points]
        min_x = math.trunc(min(x_points))
        max_x = math.trunc(max(x_points))
        min_y = math.trunc(min(y_points))
        max_y = math.trunc(max(y_points))
        min_z = math.trunc(min(z_points))
        max_z = math.trunc(max(z_points))
        final_instance_object = np.zeros((max_x - min_x + 10, max_y - min_y + 10, max_z - min_z + 10))
        for point in start_points:
            x = math.trunc(point[0])
            y = math.trunc(point[1])
            z = math.trunc(point[2])
            final_instance_object[x - min_x + 5][y - min_y + 5][z - min_z + 5] = 85
        for point in end_points:
            x = math.trunc(point[0])
            y = math.trunc(point[1])
            z = math.trunc(point[2])
            final_instance_object[x - min_x + 5][y - min_y + 5][z - min_z + 5] = 175
        for point in skeleton_points:
            x = math.trunc(point[0])
            y = math.trunc(point[1])
            z = math.trunc(point[2])
            final_instance_object[x - min_x + 5][y - min_y + 5][z - min_z + 5] = 255
        new_image = nib.Nifti1Image(final_instance_object, affine)
        # new_filename = '../results/a_generated_shape_' + str(i)
        new_filename = '../results/generated_shape_35'
        nib.save(new_image, new_filename)


def save_as_normal_file(points, suffix):
    f = open("../results_" + str(suffix) + ".txt", "a")
    for point in points:
        f.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")
    f.close()


def get_sign_of_number(number):
    if number >= 0:
        return 1
    return -1


def construct_3d_volume_array(points, edge_points):
    # return points
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    z_points = [p[2] for p in points]
    max_x, min_x = int(max(x_points)), int(min(x_points))
    max_y, min_y = int(max(y_points)), int(min(y_points))
    max_z, min_z = int(max(z_points)), int(min(z_points))
    size_x = max_x - min_x
    if get_sign_of_number(max_x) != get_sign_of_number(min_x):
        size_x += 1
    size_y = max_y - min_y
    if get_sign_of_number(max_y) != get_sign_of_number(min_y):
        size_y += 1
    size_z = max_z - min_z
    if get_sign_of_number(max_z) != get_sign_of_number(min_z):
        size_z += 1
    volume_array = np.zeros((size_x, size_y, size_z))
    mask = np.zeros((size_x, size_y, size_z))
    for point in points:
        x, y, z = int(point[0] - min_x), int(point[1] - min_y), int(point[2] - min_z)
        volume_array[x][y][z] = 1
    for point in edge_points:
        x, y, z = int(point[0] - min_x), int(point[1] - min_y), int(point[2] - min_z)
        mask[x][y][z] = 1
    return volume_array, np.array(mask, dtype='bool')


def generate_obj_file(vertices, faces, filename):
    with open('../results/' + filename, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')
        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]} \n')


def plot_distribution(data):
    for i, distances_at_point in data.items():
        for j, distances in distances_at_point.items():
            distances.sort()
            distance_counts = Counter(distances)
            sorted_distances = sorted(distance_counts.items())
            x_values, y_values = zip(*sorted_distances)
            plt.figure(figsize=(10, 6))
            plt.bar(x_values, y_values, width=0.4, color='blue', alpha=0.7)

            # Label the plot
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.title(f'Frequency of Distances for distance {i} and angle {j}')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.show()


def plot_distribution_end_points(data):
    for i, distances in data.items():
        distances.sort()
        distance_counts = Counter(distances)
        sorted_distances = sorted(distance_counts.items())
        x_values, y_values = zip(*sorted_distances)
        plt.figure(figsize=(10, 6))
        plt.bar(x_values, y_values, width=0.4, color='blue', alpha=0.7)
        # Label the plot
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Distances for distance {i}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


def group_distances(skeleton_distances, start_distances, end_distances, curvatures):
    print_log('grouping distances')
    skeleton = group_skeleton_data(skeleton_distances)
    start = group_both_ends_data(start_distances)
    end = group_both_ends_data(end_distances)
    curvature = group_curvatures_data(curvatures)
    return skeleton, start, end, curvature


def group_skeleton_data(data):
    grouped_data = {}
    for skeleton_distance in data:
        distances = data[skeleton_distance]
        for i, distances_on_point in distances.items():
            if i not in grouped_data:
                grouped_data[i] = {}
            for distance in distances_on_point:
                if distance[1] not in grouped_data[i]:
                    grouped_data[i][distance[1]] = []
                grouped_data[i][distance[1]].append(distance[0])
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


def group_curvatures_data(data):
    grouped_data = {}
    for skeleton_curvatures in data:
        curvatures = data[skeleton_curvatures]
        for i, curvature in enumerate(curvatures):
            if i not in grouped_data:
                grouped_data[i] = []
            grouped_data[i].append(curvature)
    return grouped_data


def group_edge_points_by_theta_extract_top_point(data, num_of_groups=5):
    top_point = data.popitem()
    if len(data) % num_of_groups != 0:
        raise Exception('Number of groups must be divisible by number of data')
    sorted_data = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
    # theta_values = [v[0] for v in data.values()]
    # min_value = min(theta_values)
    # max_value = max(theta_values)
    # interval = (max_value - min_value) / num_of_groups
    group_size = len(sorted_data) // num_of_groups
    grouped_data = {}
    for i in range(num_of_groups):
        grouped_data[i] = []
    # for point, angles in data.items():
    for i, (point, angles) in enumerate(sorted_data):
        # theta = angles[0]
        # group_index = (theta - min_value) // interval
        group_index = i // group_size
        if group_index >= num_of_groups:
            group_index = num_of_groups - 1
        grouped_data[group_index].append((point, angles))
    for i, points in grouped_data.items():
        grouped_data[i] = sorted(grouped_data[i], key=lambda x: x[1][1], reverse=True)
    return grouped_data, top_point
