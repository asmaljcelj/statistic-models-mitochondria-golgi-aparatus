import csv
import math
import random
import time
from collections import Counter
from tabnanny import check

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.distance_measures import center
from scipy.spatial import cKDTree
from trimesh.graph import neighbors

import math_utils

import pickle





def plot_vectors_and_points(point1, point2, point3, vector1, vector2, vector3, object_points):
    vector1 = math_utils.normalize(vector1) * 2
    vector2 = math_utils.normalize(vector2) * 2
    vector3 = math_utils.normalize(vector3) * 2
    voxels_plane = math_utils.get_voxel_on_plane(object_points, point1, vector2, epsilon=0.5)
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(voxels_plane[:, 0], voxels_plane[:, 1], voxels_plane[:, 2], alpha=0.5, c='pink')
    ax.scatter(point1[0], point1[1], point1[2], c='black')
    ax.scatter(point2[0], point2[1], point2[2], c='yellow')
    ax.scatter(point3[0], point3[1], point3[2], c='red')
    ax.quiver(*point1, *vector1, color='yellow', arrow_length_ratio=0.1)
    ax.quiver(*point1, *vector2, color='red', arrow_length_ratio=0.1)
    ax.quiver(*point1, *vector3, color='blue', arrow_length_ratio=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.grid(b=None)
    plt.show()


def plot_vectors_and_points_vector_rotation(direction_vectors, cis):
    # original_vector = math_utils.normalize(original_vector) * 5
    # rotated_vector = math_utils.normalize(rotated_vector) * 5
    # normal = math_utils.normalize(normal) * 5
    # voxels_plane = math_utils.get_voxel_on_plane(object_points, point1, normal, epsilon=0.5)
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cis[:, 0], cis[:, 1], cis[:, 2], alpha=0.5, c='pink')
    # ax.scatter(point1[0], point1[1], point1[2], c='black')
    # ax.quiver(*point1, *normal, color='yellow', arrow_length_ratio=0.1)
    # ax.quiver(*point1, *original_vector, color='red', arrow_length_ratio=0.1)
    for v in direction_vectors:
        v *= 5
        ax.quiver(*[0, 0, 0], *v, color='blue', arrow_length_ratio=0.1)
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # plt.axis('off')
    # plt.grid(b=None)
    plt.show()


def plot_3_base_vectors_and_direction_vectors(base1, base2, base3, direction_vectors):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(*[0, 0, 0], *base1, color='red', arrow_length_ratio=0.1)
    ax.quiver(*[0, 0, 0], *base2, color='green', arrow_length_ratio=0.1)
    ax.quiver(*[0, 0, 0], *base3, color='yellow', arrow_length_ratio=0.1)
    for v in direction_vectors:
        ax.quiver(*[0, 0, 0], *v, color='blue', arrow_length_ratio=0.1, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.axis('off')
    plt.grid(b=None)
    plt.rcParams['axes.grid'] = False
    plt.show()


def plot_save_result(num_of_points, bezier_curve, original_points, arc_length_approx, number_of_plots, filename):
    # plot the result and save it
    if num_of_points == number_of_plots:
        time.sleep(15)
        num_of_points = 0
    matplotlib.use('TkAgg')
    bezier_curve = np.array(bezier_curve)
    arc_length_approx = np.array(arc_length_approx)
    num_of_points += 1
    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(
        bezier_curve[:, 0],  # x-coordinates.
        bezier_curve[:, 1],  # y-coordinates.
        bezier_curve[:, 2],  # y-coordinates.
        'o:'
        # label='Bezier curve'
    )
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.plot(
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
    #     'ro:',  # Styling (yellow, circles, dotted).
    #     label='Original skeleton',
    #     alpha=0.3
    # )
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.legend()
    # ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(70, 150)
    ax1.view_init(70, 150)
    plt.rcParams['axes.grid'] = False
    # plt.title(filename)
    # plt.savefig('../plots/' + filename + '.png')
    # plt.close()
    plt.show()


def plot_bezier_curve(curve):
    matplotlib.use('TkAgg')
    curve = np.array(curve)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        curve[:, 0],  # x-coordinates.
        curve[:, 1],  # y-coordinates.
        curve[:, 2],  # y-coordinates.
        'o:',
        label='Bezier curve'
    )
    plt.show()


def plot_3d(points):
    matplotlib.use('TkAgg')
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.grid(False)
    # plt.axis('off')
    # plt.grid(b=None)
    plt.show()


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
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2],  alpha=0.3)
    # ax.view_init(azim=-125, elev=-40)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.grid(b=None)
    plt.show()


def plot_2_sets_of_points(old_points, new_points):
    # matplotlib.use('TkAgg')
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(old_points[:, 0], old_points[:, 1], old_points[:, 2], 'go')
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], 'yo')
    # ax.view_init(azim=-125, elev=-40)
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


def save_as_nii_2(set_of_points):
    affine = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    x_points = [p[0] for p in set_of_points]
    y_points = [p[1] for p in set_of_points]
    z_points = [p[2] for p in set_of_points]
    min_x = math.trunc(min(x_points))
    max_x = math.trunc(max(x_points))
    min_y = math.trunc(min(y_points))
    max_y = math.trunc(max(y_points))
    min_z = math.trunc(min(z_points))
    max_z = math.trunc(max(z_points))
    final_instance_object = np.zeros((max_x - min_x + 10, max_y - min_y + 10, max_z - min_z + 10))
    for point in set_of_points:
        x = math.trunc(point[0])
        y = math.trunc(point[1])
        z = math.trunc(point[2])
        final_instance_object[x - min_x + 5][y - min_y + 5][z - min_z + 5] = 255
    new_image = nib.Nifti1Image(final_instance_object, affine)
    new_filename = '../results/generated_shape_0'
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
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1} \n')


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


def group_distances(skeleton_distances, start_distances, end_distances, curvatures, torsions):
    print('grouping distances')
    skeleton = group_skeleton_data(skeleton_distances)
    start = group_both_ends_data(start_distances)
    end = group_both_ends_data(end_distances)
    curvature = group_curvatures_data(curvatures)
    t = group_curvatures_data(torsions)
    return skeleton, start, end, curvature, t


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
            # key = float(i[0]), float(i[1]), float(i[2])
            # if key not in grouped_data:
            #     grouped_data[key] = []
            # grouped_data[key].append(distance)
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


def group_edge_points_by_theta_extract_top_point(data, num_of_groups=5, angle_increment=1):
    num_of_points_on_kroznica = int(360 / angle_increment)
    kroznica_points = list(data.items())[:num_of_points_on_kroznica]
    data = dict(list(data.items())[num_of_points_on_kroznica:])
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
        grouped_data[i] = sorted(grouped_data[i], key=lambda x: x[1][1])
    return grouped_data, top_point, kroznica_points


def dict_key_from_point(point):
    return point[0], point[1], point[2]


def find_nearest_point_from_point(point, points):
    min_distance, min_point, index = math.inf, None, -1
    for i, p in enumerate(points):
        distance = math_utils.distance_between_points(p, point)
        if distance < min_distance:
            min_distance = distance
            min_point = p
            index = i
    return min_point, index


def save_measurements_to_file(filename, skeleton, start, end, curvature, lengths, direction_with_angles, torsions):
    combined = {'skeleton': skeleton, 'start': start, 'end': end, 'curvature': curvature, 'lengths': lengths, 'direction_with_angles': direction_with_angles, 'torsions': torsions}
    with open(filename, 'wb') as file:
        pickle.dump(combined, file)


def save_ga_measurements_to_file(filename, calculated_distances):
    with open(filename, 'wb') as file:
        pickle.dump(calculated_distances, file)


def group_calculations_ga(distances):
    x, first, y, second, minus_x, third, minus_y, fourth = [], [], [], [], [], [], [], []
    for distance in distances:
        x.append(distance[0])
        first.append(distance[1])
        y.append(distance[2])
        second.append(distance[3])
        minus_x.append(distance[4])
        third.append(distance[5])
        minus_y.append(distance[6])
        fourth.append(distance[7])
    return x, first, y, second, minus_x, third, minus_y, fourth


def read_measurements_from_file(filename):
    print('reading result')
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data['curvature'], data['start'], data['end'], data['skeleton'], data['lengths'], data['direction_with_angles'], data['torsions']


def read_measurements_from_file_ga(filename):
    print('reading result')
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data


def retrieve_new_value_from_standard_derivation(sigma, data):
    average, standard_deviation = math_utils.calculate_average_and_standard_deviation(data)
    sample = np.random.normal(0.5, sigma, 1)
    whole_std_interval = 2 * np.array(standard_deviation)
    return (average - standard_deviation) + sample * whole_std_interval


def plot_histograms_for_data(skeletons, start, end, curvature, torsion, lengths):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))
    ax[0, 0].set_title('Skeleton', fontsize='30')
    ax[0, 0].hist(skeletons[1][3], bins=10, color='skyblue', edgecolor='black')
    random_key = random.choice(list(start.keys()))
    ax[0, 1].set_title('Začetek', fontsize='30')
    ax[0, 1].hist(start[random_key], bins=10, color='skyblue', edgecolor='black')
    ax[0, 2].set_title('Konec', fontsize='30')
    ax[0, 2].hist(end[random_key], bins=10, color='skyblue', edgecolor='black')
    ax[1, 0].set_title('Ukrivljenost', fontsize='30')
    ax[1, 0].hist(curvature[11], bins=10, color='skyblue', edgecolor='black')
    ax[1, 1].set_title('Torzija', fontsize='30')
    ax[1, 1].hist(torsion[11], bins=10, color='skyblue', edgecolor='black')
    ax[1, 2].set_title('Dolžina skeletona', fontsize='30')
    ax[1, 2].hist(lengths, bins=10, color='skyblue', edgecolor='black')
    plt.show()

def plot_histograms_for_ga_data(length, first_cisterna, last_cisterna):
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 15))
    ax[0].set_title('Dolžine', fontsize='30')
    ax[0].hist(length, bins=5, color='skyblue', edgecolor='black')
    ax[1].set_title('1. cisterna', fontsize='30')
    ax[1].hist(first_cisterna, bins=5, color='skyblue', edgecolor='black')
    ax[2].set_title('Zadnja cisterna', fontsize='30')
    ax[2].hist(last_cisterna, bins=5, color='skyblue', edgecolor='black')
    plt.show()


def plot_generated_skeleton_points(previous_points, T, N, B, new_point, new_T, new_N, new_B):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    previous_points = np.array(previous_points)
    T = np.array(T)
    N = np.array(N) * 0.05
    B = np.array(B) * 0.05
    ax.plot(previous_points[-2:, 0], previous_points[-2:, 1], previous_points[-2:, 2], c='pink')
    ax.scatter(previous_points[-2:, 0], previous_points[-2:, 1], previous_points[-2:, 2], c='pink')
    ax.scatter(new_point[0], new_point[1], new_point[2], c='black')
    # ax.quiver(*point1, *normal, color='yellow', arrow_length_ratio=0.1)
    ax.quiver(*previous_points[-1], *T, color='red', arrow_length_ratio=0.005)
    ax.quiver(*previous_points[-1], *N, color='blue', arrow_length_ratio=0.005)
    ax.quiver(*previous_points[-1], *B, color='green', arrow_length_ratio=0.005)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.axis('off')
    plt.grid(b=None)
    plt.show()
    print()


def cisterna_volume_extraction(cisterna_points):
    # plot_3d(cisterna_points)
    current_volume_points = cisterna_points.copy()
    centers = []
    grouped_points = {}
    while len(current_volume_points) > 0:
        new_volume_points = []
        tree = cKDTree(current_volume_points)
        for point in current_volume_points:
            neighbors = tree.query_ball_point(point, 1.8)
            # if points has no neighbors, we found the center
            if len(neighbors) == 1:
                centers.append(point)
            # if point has 3 neighbors, that point is not on the edge, so preserve it (also check if it has neighbors in opposite directions)
            # if len(neighbors) >= 3 and check_if_voxel_inside(neighbors, current_volume_points, point):
            if check_if_voxel_inside(neighbors, current_volume_points, point):
                new_volume_points.append(point)
        current_volume_points = np.array(new_volume_points)
    if len(centers) == 0:
        mean = np.mean(cisterna_points, axis=0)
        grouped_points[tuple(mean)] = cisterna_points
    else:
        # for every center, calculate points that belong to it
        grouped_points = calculate_points_around_centers(centers, cisterna_points)
    # if len(grouped_points.keys()) > 1:
    #     print()
    #     plot_grouped_points(grouped_points)
    #     print()
    return grouped_points


def check_if_voxel_inside(neighbors, points, current_point):
    x, y, z = current_point
    neighbors_points = [list(points[i]) for i  in neighbors if not np.array_equal(points[i], current_point)]
    if [x - 1, y, z] in neighbors_points and [x + 1, y, z] in neighbors_points:
        return True
    if [x, y - 1, z] in neighbors_points and [x, y + 1, z] in neighbors_points:
        return True
    if [x, y, z - 1] in neighbors_points and [x, y, z + 1] in neighbors_points:
        return True
    # 4 diagonale
    if [x - 1, y - 1, z + 1] in neighbors_points and [x + 1, y + 1, z - 1] in neighbors_points:
        return True
    if [x - 1, y - 1, z - 1] in neighbors_points and [x + 1, y + 1, z + 1] in neighbors_points:
        return True
    if [x + 1, y - 1, z - 1] in neighbors_points and [x - 1, y + 1, z + 1] in neighbors_points:
        return True
    if [x + 1, y - 1, z + 1] in neighbors_points and [x - 1, y + 1, z - 1] in neighbors_points:
        return True
    # rob
    if [x, y - 1, z - 1] in neighbors_points and [x, y + 1, z + 1] in neighbors_points:
        return True
    if [x + 1, y, z - 1] in neighbors_points and [x - 1, y, z + 1] in neighbors_points:
        return True
    if [x, y + 1, z - 1] in neighbors_points and [x, y - 1, z + 1] in neighbors_points:
        return True
    if [x - 1, y, z - 1] in neighbors_points and [x + 1, y, z + 1] in neighbors_points:
        return True
    # rob diagonala
    if [x - 1, y - 1, z] in neighbors_points and [x + 1, y + 1, z] in neighbors_points:
        return True
    if [x + 1, y - 1, z] in neighbors_points and [x - 1, y + 1, z] in neighbors_points:
        return True
    return False


def calculate_points_around_centers(centers, cisterna_points):
    tree = cKDTree(cisterna_points)
    points_to_examine = {}
    points_assigned_to_center = {}
    # Determine array dimensions (max coordinate values)
    max_x, max_y, max_z = np.max(cisterna_points, axis=0)
    checked = np.zeros((max_x + 1, max_y + 1, max_z + 1), dtype=bool)
    for center in centers:
        points_to_examine[tuple(center)] = [center]
        points_assigned_to_center[tuple(center)] = []
    while all(len(points_to_examine[c]) != 0 for c in points_to_examine):
        for c in points_to_examine:
            new_points_to_examine = []
            for point in points_to_examine[c]:
                neighbors = tree.query_ball_point(point, 1.8)
                for neighbor in neighbors:
                    p = cisterna_points[neighbor]
                    # if not any(np.array_equal(p, arr) for group in points_assigned_to_center.values() for arr in group):
                    if not checked[p[0]][p[1]][p[2]]:
                        points_assigned_to_center[c].append(p)
                        new_points_to_examine.append(p)
                        checked[p[0]][p[1]][p[2]] = True
            points_to_examine[c] = new_points_to_examine
    # calculate new centers
    final_points = {}
    for c in points_assigned_to_center:
        points = points_assigned_to_center[c]
        mean = np.mean(points, axis=0)
        final_points[tuple(mean)] = points
    return final_points


def plot_grouped_points(grouped_points):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.jet(np.linspace(0, 1, len(grouped_points)))

    for (center, points), color in zip(grouped_points.items(), colors):
        center = np.array(center)
        points = np.array(points)

        # Plot center with a larger marker
        ax.scatter(*center, color=color, marker='o', s=100, edgecolor='black', label=f'Center {center}')

        # Plot points with the same color
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, marker='o', alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def plot_points_and_vector(object_points, eigenvectors, direction_vectors):
    vector1 = math_utils.normalize(vector1) * 2
    vector2 = math_utils.normalize(vector2) * 2
    vector3 = math_utils.normalize(vector3) * 2
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], alpha=0.5, c='pink')
    ax.quiver(*point1, *vector1, color='yellow', arrow_length_ratio=0.1)
    ax.quiver(*point1, *vector2, color='red', arrow_length_ratio=0.1)
    ax.quiver(*point1, *vector3, color='blue', arrow_length_ratio=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.grid(b=None)
    plt.show()


def create_3d_image(voxels):
    min_coords = voxels.min(axis=0)
    shifted_voxels = voxels - min_coords
    shape = np.array((128, 128, 128))
    volume = np.zeros(shape, dtype=np.uint8)
    volume[tuple(shifted_voxels.T)] = 1
    return volume
