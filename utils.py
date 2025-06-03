import csv
import pickle

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

import math_utils


def plot_3d(points):
    matplotlib.use('TkAgg')
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.axis('off')
    plt.grid(b=None)
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


def generate_obj_file(vertices, faces, filename):
    with open('results/' + filename, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} \n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1} \n')


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
    # grupira točke na polsferi glede na kot theta, na vrhu polsfere naredi točko in generiraj krožico točk
    num_of_points_on_kroznica = int(360 / angle_increment)
    kroznica_points = list(data.items())[:num_of_points_on_kroznica]
    data = dict(list(data.items())[num_of_points_on_kroznica:])
    top_point = data.popitem()
    if len(data) % num_of_groups != 0:
        raise Exception('Number of groups must be divisible by number of data')
    sorted_data = sorted(data.items(), key=lambda x: x[1][0], reverse=True)
    group_size = len(sorted_data) // num_of_groups
    grouped_data = {}
    for i in range(num_of_groups):
        grouped_data[i] = []
    for i, (point, angles) in enumerate(sorted_data):
        group_index = i // group_size
        if group_index >= num_of_groups:
            group_index = num_of_groups - 1
        grouped_data[group_index].append((point, angles))
    for i, points in grouped_data.items():
        grouped_data[i] = sorted(grouped_data[i], key=lambda x: x[1][1])
    return grouped_data, top_point, kroznica_points


def dict_key_from_point(point):
    return point[0], point[1], point[2]


def save_measurements_to_file(filename, skeleton, start, end, curvature, lengths, direction_with_angles, torsions):
    combined = {'skeleton': skeleton, 'start': start, 'end': end, 'curvature': curvature, 'lengths': lengths, 'direction_with_angles': direction_with_angles, 'torsions': torsions}
    with open(filename, 'wb') as file:
        pickle.dump(combined, file)


def save_ga_measurements_to_file(filename, calculated_distances):
    with open(filename, 'wb') as file:
        pickle.dump(calculated_distances, file)


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


def retrieve_new_value_from_standard_derivation_ga(sigma, sigma_scale, data):
    data = np.array(data)
    X = data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma)
    kde.fit(X)
    sample = np.abs(kde.sample()[0][0])
    sample = sample * sigma_scale
    return sample


def cisterna_volume_extraction(cisterna_points):
    db = DBSCAN(eps=1.8, min_samples=3).fit(cisterna_points)
    labels = db.labels_
    labeled_array = {}
    for i, label in enumerate(labels):
        if label == -1:
            print('voxel detected as noise, ignoring')
            continue
        if label not in labeled_array:
            labeled_array[label] = []
        point = [cisterna_points[i][0], cisterna_points[i][1]]
        labeled_array[label].append(np.array(point))
    print('found', len(labeled_array), 'centers')
    grouped_points = {}
    for key in labeled_array:
        if len(labeled_array[key]) < 4:
            print('to little number of elements', len(labeled_array[key]), '-ignoring')
            continue
        mean = np.mean(labeled_array[key], axis=0)
        grouped_points[tuple(mean)] = labeled_array[key]
    return grouped_points


def create_3d_image(voxels):
    min_coords = voxels.min(axis=0)
    shifted_voxels = voxels - min_coords
    shape = np.array((128, 128, 128))
    volume = np.zeros(shape, dtype=np.uint8)
    volume[tuple(shifted_voxels.T)] = 1
    return volume


def read_nii_data(file_path):
    # read .nii file at specified location and return its data including image's metainfo
    nib_image = nib.load(file_path)
    return nib_image.get_fdata(), nib_image


def save_new_object(data, filename, directory_path, image):
    new_image = nib.Nifti1Image(data, image.affine)
    nib.save(new_image, directory_path + '/' + filename)


def plot_ga_object(voxels):
    # vizualizacija primerke Goglijevega aparata
    cloud = pv.PolyData(voxels)
    volume = cloud.delaunay_3d(alpha=1.5)
    surface = volume.extract_geometry()
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color="lightgray", smooth_shading=False, show_edges=False)
    plotter.add_axes()
    plotter.set_background("white")
    plotter.show()


def initialize_list_of_lists(size):
    l = []
    for i in range(size):
        l.append([])
    return l


def add_triangle_to_mesh(faces, index1, index2, index3, flip_orientation=False):
    triangle = [index1, index2, index3]
    if flip_orientation:
        triangle = [index1, index3, index2]
    faces.append(triangle)
