import math

import numpy as np
import pyvista as pv
import scipy.spatial
from scipy.stats import gaussian_kde
from utils import plot_kde, plot_new_points, save_as_nii, save_as_normal_file, save_as_nii_layers, get_sign_of_number, construct_3d_volume_array, generate_obj_file
from math_utils import rotate_vector, get_points_between_2_points
import trimesh
from skimage import measure
from mayavi import mlab
import open3d as o3d


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


def group_distances(skeleton_distances, start_distances, end_distances, curvatures):
    print('grouping distances')
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


def group_curvatures_data(data):
    grouped_data = {}
    for skeleton_curvatures in data:
        curvatures = data[skeleton_curvatures]
        for i, curvature in enumerate(curvatures):
            if i not in grouped_data:
                grouped_data[i] = []
            grouped_data[i].append(curvature)
    return grouped_data


def calculate_new_skeleton_point(previous_point, curvature_value):
    radius_of_circle = abs(1 / curvature_value)
    center_of_circle = np.copy(previous_point)
    if curvature_value < 0:
        center_of_circle[1] -= radius_of_circle
        return find_new_skeleton_point(center_of_circle, previous_point, radius_of_circle, 1)
    elif curvature_value > 0:
        center_of_circle[1] += radius_of_circle
        return find_new_skeleton_point(center_of_circle, previous_point, radius_of_circle, -1)
    else:
        # curvature value is 0 -> take only straight point
        return [0, previous_point[1], previous_point[2] + 1]


def find_new_skeleton_point(center, current_point, radius, direction):
    theta = 1 / radius
    phi = np.pi / 2 * get_sign_of_number(current_point[1] - center[1])
    z1 = center[2] + radius * math.cos(phi + theta)
    if direction == -1:
        z1 = center[2] + radius * math.cos(phi - theta)
    y1 = center[1] + radius * math.sin(phi + theta)
    return [0, y1, z1]


def sample_new_points(skeleton_distances, start_distances, end_distances, curvature, num_files):
    # calculate skeleton points based on curvature
    print('finding new skeleton points')
    new_points, total_skeleton_points = [], []
    start_edge_points, end_edge_points, skeleton_points = [], [], []
    for _ in range(num_files):
        new_points.append([])
        start_edge_points.append([])
        end_edge_points.append([])
        skeleton_points.append([])
        total_skeleton_points.append(np.array([[0, 0, 0]], dtype=float))
    for c in curvature:
        data = curvature[c]
        kde = gaussian_kde(data)
        range_distances = np.linspace(min(data), max(data), 10000)
        pdf_estimation = kde(range_distances)
        cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
        samples = np.random.normal(0, 1, num_files)
        for i, sample in enumerate(samples):
            new_curvature = np.interp(sample, cdf_estimation, range_distances)
            # skeleton_points = np.append(skeleton_points, [calculate_new_skeleton_point(skeleton_points[-1], new_curvature)], axis=0)
            total_skeleton_points[i] = np.append(total_skeleton_points[i], [calculate_new_skeleton_point(total_skeleton_points[i][-1], new_curvature)], axis=0)
    # skeleton_points = np.column_stack((np.zeros(10), np.zeros(10), np.linspace(1, 10, 10, endpoint=True)))
    # skeleton_points = np.array(skeleton_points)
    # plot_new_points(total_skeleton_points[0])
    faces = []
    # start
    print('generating start points')
    for direction, distances in start_distances.items():
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        range_distances = np.linspace(min(distances), max(distances), 10000)
        pdf_estimation = kde(range_distances)
        # plot_kde(range, pdf_estimation)
        cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
        samples = np.random.normal(0, 1, num_files)
        for i, sample in enumerate(samples):
            new_distance = np.interp(sample, cdf_estimation, range_distances)
            # new_distance *= 10
            # calculate new boundary point in 3D space
            new_point = new_distance * (direction * np.array(-1)) + total_skeleton_points[i][-1]
            # new_points[i].extend(get_points_between_2_points(np.array(([0, 0, 0])), new_point, math.ceil(new_distance)))
            new_points[i].append(new_point)
            start_edge_points[i].append(new_point)
        # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
        # new_points.append(new_point)
    # end
    print('generating end points')
    for direction, distances in end_distances.items():
        distances = np.array(distances)
        kde = gaussian_kde(distances)
        range_distances = np.linspace(min(distances), max(distances), 10000)
        pdf_estimation = kde(range_distances)
        # plot_kde(range, pdf_estimation)
        cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
        samples = np.random.normal(0, 1, num_files)
        for i, sample in enumerate(samples):
            new_distance = np.interp(sample, cdf_estimation, range_distances)
            # new_distance *= 10
            # calculate new boundary point in 3D space
            new_point = new_distance * np.array(direction)
            # new_points[i].extend(get_points_between_2_points(total_skeleton_points[i][-1], new_point, math.ceil(new_distance)))
            new_points[i].append(new_point)
            end_edge_points[i].append(new_point)
            # new_point = direction + last_point
            # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
            # new_points.append(new_point)
    # generate points on skeleton
    print('generating skeleton points')
    for point, distances_around in skeleton_distances.items():
        for angle, distances in distances_around.items():
            distances = np.array(distances)
            kde = gaussian_kde(distances)
            range_distances = np.linspace(min(distances), max(distances), 10000)
            pdf_estimation = kde(range_distances)
            # plot_kde(range, pdf_estimation)
            cdf_estimation = np.cumsum(pdf_estimation) / np.sum(pdf_estimation)
            samples = np.random.normal(0, 1, num_files)
            for i, sample in enumerate(samples):
                new_distance = np.interp(sample, cdf_estimation, range_distances)
                # new_distance *= 10
                # calculate new boundary point in 3D space
                direction = rotate_vector(np.array([1, 0, 0]), angle, np.array([0, 0, 1]))
                new_point = new_distance * np.array(direction) + total_skeleton_points[i][point]
                # new_point = [math.trunc(new_point[0]), math.trunc(new_point[1]), math.trunc(new_point[2])]
                # new_points[i].extend(get_points_between_2_points(total_skeleton_points[i][point], new_point, math.ceil(new_distance)))
                new_points[i].append(new_point)
                skeleton_points[i].append(new_point)
                # new_points.append(new_point)
    for i in range(num_files):
        new_points[i] = np.array(new_points[i])
        generate_mesh(new_points[i])
    # save_as_nii_layers(start_edge_points, end_edge_points, skeleton_points)
    # save_as_normal_file(new_points, 1)


def generate_mesh(points):
    print('Generating mesh')
    # todo: tu moram definirati se ploskve oz. kako se oglišča povezujejo med sabo
    volume = construct_3d_volume_array(points)
    vertices, faces, normals, values = measure.marching_cubes(volume)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.visualization.draw_geometries([mesh])
    # mlab.triangular_mesh([vert[0] for vert in vertices],
    #                      [vert[1] for vert in vertices],
    #                      [vert[2] for vert in vertices],
    #                      faces)
    # mlab.show()
    # tri = scipy.spatial.Delaunay(points, incremental=True)
    # print('Gotovo')
    # tri_mesh = trimesh.Trimesh(vertices=points, faces=tri.simplices)
    # smooth = trimesh.smoothing.filter_humphrey(tri_mesh)
    # generate_obj_file(smooth.vertices, smooth.faces)
    # smooth_vertices = smooth.vertices

    # cloud = pv.PolyData(points)
    # # cloud.plot()
    #
    # # print('Generating geometry')
    # volume = cloud.reconstruct_surface(progress_bar=True)
    # shell = volume.extract_geometry()
    # shell.plot()
