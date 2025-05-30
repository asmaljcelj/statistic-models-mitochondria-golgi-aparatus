import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.bipartite.basic import color
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import scipy.ndimage as ndi

def has_all_elements(array1, array2):
    remaining = len(array1)
    for i, e in enumerate(array1):
        if e not in array2:
            print('not present', e)
            return False
        remaining -= 1
    print('remaining:', remaining)
    return True

def plot_dataset_and_pca(dataset, vectors):
    matplotlib.use('TkAgg')
    x = [p[0] for p in dataset]
    y = [p[1] for p in dataset]
    z = [p[2] for p in dataset]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(x, y, z)
    origin = np.array([0, 0, 0])
    colors = ['r', 'g', 'b']
    for i, vector in enumerate(vectors):
        vector = vector * 10
        ax.quiver(*origin, *vector, color=colors[i], arrow_length_ratio=0.1)
    plt.show()

def plot_dataset_moved_and_pca(dataset, aligned_dataset, vectors, mean):
    matplotlib.use('TkAgg')
    # x = [p[0] for p in dataset]
    # y = [p[1] for p in dataset]
    # z = [p[2] for p in dataset]
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='orange')
    x_a = np.array([p[0] for p in aligned_dataset])
    y_a = np.array([p[1] for p in aligned_dataset])
    z_a = np.array([p[2] for p in aligned_dataset])
    # ax1 = fig.add_subplot(122, projection='3d')
    ax.scatter(x_a, y_a, z_a, c='blue', depthshade=True)
    # surface = ax.plot_surface(x_a, y_a, z_a, cmap='viridis', edgecolor='none', shade=True)

    # fig.colorbar(surface)
    ax.view_init(43, -140)
    # ax1.view_init(43, -140)
    # x_m = [p[0] for p in mean]
    # y_m = [p[1] for p in mean]
    # z_m = [p[2] for p in mean]
    # ax.plot(x_m, y_m, z_m)
    # origin = np.array([0, 0, 0])
    # colors = ['r', 'g', 'b']
    # for i, vector in enumerate(vectors):
    #     vector = vector * 100
    #     ax.quiver(*origin, *vector, color=colors[i], arrow_length_ratio=0.1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # # ax1.grid(False)
    # # ax1.set_xticks([])
    # # ax1.set_yticks([])
    # # ax1.set_zticks([])
    plt.axis('off')
    plt.grid(b=None)
    plt.show()
    # print()

    # cubes = [pv.Cube(center=voxel, x_length=1, y_length=1, z_length=1) for voxel in aligned_dataset]

    # === Step 3: Combine cubes into a single surface ===
    # combined = pv.MultiBlock(cubes).combine()

    # Optional: Smooth normals to improve shading
    # combined.compute_normals(inplace=True, auto_orient_normals=True)

    # === Step 4: Plot with shading and lighting ===
    # plotter = pv.Plotter()
    # plotter.add_mesh(combined, color="lightblue", smooth_shading=True, show_edges=False)
    # plotter.add_axes()
    # plotter.show()
    print()


def plot(voxels):
    # === Step 2: Create PyVista point cloud ===
    cloud = pv.PolyData(voxels)

    # === Step 3: Surface reconstruction using Delaunay 3D ===
    # This creates a volume from which we can extract an outer surface
    volume = cloud.delaunay_3d(alpha=1.5)
    surface = volume.extract_geometry()

    # === Step 4: Plot with edge emphasis ===
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color="lightgray", smooth_shading=False, show_edges=False)
    plotter.add_axes()
    plotter.set_background("white")
    plotter.show()
    print()


def plot_points(points):
    points = np.array(points)
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


def plot_original_and_aligned_cisterna(original, aligned):
    original = np.array(original)
    aligned = np.array(aligned)
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='original')
    ax.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='red', label='aligned')
    ax.legend()
    plt.show()


def check_points_in_bounds(point):
    return 0 <= int(point[0]) <= 255 and 0 <= int(point[1]) <= 255 and 0 <= int(point[2]) <= 255


def plot_ga_instances(points, vector, vector2, center, lowest_point, cistarnae):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='3D Points')
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, c in enumerate(cistarnae):
        if len(c) != 0:
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=colors[i % len(colors)], label='cistarnae' + str(i))

    # t = np.linspace(0, 10, 100)
    # x = t * vector[0] # X coordinates
    # y = vector[1] * t  # Y coordinates
    # z = vector[2] * t  # Z coordinates
    # x2 = t * vector2[0]  # X coordinates
    # y2 = vector2[1] * t  # Y coordinates
    # z2 = vector2[2] * t  # Z coordinates
    #
    # # Plot the vector (originating from the origin (0, 0, 0))
    # ax.plot(x, y, z, label='3D Line 1 ', color='r')
    # ax.plot(x2, y2, z2, label='3D Line 2', color='g')
    #
    # ax.plot([0, center[0]], [0, center[1]], [0, center[2]], label='center', color='y')
    # ax.plot([0, lowest_point[0]], [0, lowest_point[1]], [0, lowest_point[2]], label='center', color='c')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the legend
    # ax.legend()

    # Show the plot
    plt.show()
    print()


data_directory = '../data_ga/approximate'
extracted_data_directory = '../extracted_ga_data'


def check_if_voxel_is_ga(volume, point, points_to_process, instance_voxels, covered_volume, instantiated_volume):
    if point[0] < 0 or point[0] > 255 or point[1] < 0 or point[1] > 255 or point[2] < 0 or point[2] > 255:
        # invalid range
        return points_to_process, instance_voxels, instantiated_volume
    if volume[point[0]][point[1]][point[2]] != 0 and covered_volume[point[0]][point[1]][point[2]] == 0 and instantiated_volume[point[0]][point[1]][point[2]] == 0:
        points_to_process.append(point)
        instance_voxels.append(point)
        instantiated_volume[point[0]][point[1]][point[2]] = 1
    return points_to_process, instance_voxels, instantiated_volume


def get_entire_instance_2(start_point, volume, covered_volume, instantiated_volume):
    points_to_process = [start_point]
    instance_voxels = [start_point]
    while points_to_process:
        current_point = points_to_process.pop(0)
        if covered_volume[current_point[0]][current_point[1]][current_point[2]] != 0:
            continue
        covered_volume[current_point[0]][current_point[1]][current_point[2]] = 1
        left = [current_point[0] - 1, current_point[1], current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right = [current_point[0] + 1, current_point[1], current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        top = [current_point[0], current_point[1], current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, top, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        bot = [current_point[0], current_point[1], current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, bot, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        front = [current_point[0], current_point[1] - 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, front, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        back = [current_point[0], current_point[1] + 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, back, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_up_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_up_front, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_front = [current_point[0] - 1, current_point[1] - 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_front, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_down_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_down_front, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_up = [current_point[0] - 1, current_point[1], current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_down = [current_point[0] - 1, current_point[1], current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_back_up = [current_point[0] - 1, current_point[1] + 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_back_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_back_down = [current_point[0] - 1, current_point[1] + 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_back_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        left_back = [current_point[0] - 1, current_point[1] + 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, left_back, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        front_up = [current_point[0], current_point[1] - 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, front_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        front_down = [current_point[0], current_point[1] - 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, front_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        back_up = [current_point[0], current_point[1] + 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, back_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        back_down = [current_point[0], current_point[1] + 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, back_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_front_up = [current_point[0] + 1, current_point[1] - 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_front_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_front = [current_point[0] + 1, current_point[1] - 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_front, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_front_down = [current_point[0] + 1, current_point[1] - 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_front_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_up = [current_point[0] + 1, current_point[1], current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_down = [current_point[0] + 1, current_point[1], current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_back_up = [current_point[0] + 1, current_point[1] + 1, current_point[2] + 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_back_up, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_back = [current_point[0] + 1, current_point[1] + 1, current_point[2]]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_back, points_to_process, instance_voxels, covered_volume, instantiated_volume)
        right_back_down = [current_point[0] + 1, current_point[1] + 1, current_point[2] - 1]
        points_to_process, instance_voxels, instantiated_volume = check_if_voxel_is_ga(volume, right_back_down, points_to_process, instance_voxels, covered_volume, instantiated_volume)
    return instance_voxels, covered_volume, instantiated_volume


def check_if_boundary_point(current_point, instance_voxels):
    number_of_instance_neighbors = 0
    left = [current_point[0] - 1, current_point[1], current_point[2]]
    if instance_voxels[left[0], left[1], left[2]] == 1:
        number_of_instance_neighbors += 1
    right = [current_point[0] + 1, current_point[1], current_point[2]]
    if instance_voxels[right[0], right[1], right[2]] == 1:
        number_of_instance_neighbors += 1
    top = [current_point[0], current_point[1], current_point[2] + 1]
    if instance_voxels[top[0], top[1], top[2]] == 1:
        number_of_instance_neighbors += 1
    bot = [current_point[0], current_point[1], current_point[2] - 1]
    if instance_voxels[bot[0], bot[1], bot[2]] == 1:
        number_of_instance_neighbors += 1
    front = [current_point[0], current_point[1] - 1, current_point[2]]
    if instance_voxels[front[0], front[1], front[2]] == 1:
        number_of_instance_neighbors += 1
    back = [current_point[0], current_point[1] + 1, current_point[2]]
    if instance_voxels[back[0], back[1], back[2]] == 1:
        number_of_instance_neighbors += 1
    return number_of_instance_neighbors <= 5


def get_point_cloud(instance_points, instance_voxels):
    point_cloud = []
    for current_point in instance_points:
        if check_if_boundary_point(current_point, instance_voxels):
            point_cloud.append(current_point)
    return point_cloud


def extract_ga_instances(volume):
    labeled, num_stacks = ndi.label(volume)
    # instance_volume hrani vrednosti, kateremu GA pripada voksel, ce 0 -> ne pripada nobenemu
    # covered_volume: vrednost 0 -> voksel se ni bil obdelan, 1 -> je ze bil obdelan
    # instance_volume, covered_volume, instantiated_volume = np.zeros(volume.shape), np.zeros(volume.shape), np.zeros(volume.shape)
    # current_instance = 1
    ga_instances = {}
    # for x in range(len(instance_volume)):
    #     for y in range(len(instance_volume[x])):
    #         for z in range(len(instance_volume[x][y])):
    #             if volume[x][y][z] == 1 and covered_volume[x][y][z] == 0:
    #                 print('found new instance at', x, '-', y, '-', z)
    #                 ga_instances[current_instance], covered_volume, instantiated_volume = get_entire_instance_2([x, y, z], volume, covered_volume, instantiated_volume)
    #                 current_instance += 1
    # print('found', current_instance - 1, 'instances in this volume')
    print('found', num_stacks, 'instances in this volume')
    ga_point_clouds = {}
    # for index in ga_instances:
    #     print('instance', index, 'has', len(ga_instances[index]), 'voxels')
        # point_cloud = get_point_cloud(ga_instances[index], instance_volume)
        # ga_point_clouds[index] = point_cloud
        # print('instance', index, 'has', len(ga_point_clouds[index]), 'edge voxels')
        # print()
    for i in range(num_stacks):
        ga_instances[(i + 1)] = np.array(np.where(labeled == (i + 1))).T
    return ga_instances

instances_folder = '../ga_instances'


def save_files(image_data, data, og_filename):
    print('saving files')
    counter = 1
    # pca = PCA(n_components=2)
    for key in data:
        new_filename = og_filename[:og_filename.find('.')] + '_' + str(counter)
        counter += 1
        voxels = data[key]
        final_instance_object = np.zeros(image_data.shape)
        for voxel in voxels:
            final_instance_object[voxel[0], voxel[1], voxel[2]] = 1
        np.savetxt('../ga_instances/' + new_filename + '.csv', voxels, delimiter=',', fmt='%-0d')
        # dataset = pd.read_csv(instances_folder + '/' + new_filename + '.csv')
        # pca.fit(dataset)
        # eig_vec = pca.components_
        # plot_ga_instances(data[key], eig_vec)
        # first_unit = eig_vec[0]
        # # for t in range(1, 15):
        # #     point = (t * first_unit[0], t * first_unit[1], t * first_unit[2])
        # #     final_instance_object[int(point[0])][int(point[1])][int(point[2])] = 120
        #
        new_image = nib.Nifti1Image(final_instance_object, image_data.affine)
        nib.save(new_image, extracted_data_directory + '/' + new_filename)


def read_files(instance_volume, filename, dataset):
    # filename = filename.replace('.nii.gz', '')
    # all_cistarnae_points = {}
    # for i in range(1, 10000):
        # filepath = instances_folder + '/' + filename + '_' + str(i) + '.csv'
        # if not os.path.exists(filepath):
        #     break
    # pca = PCA(n_components=2)
    # dataset = pd.read_csv(filepath)
    dataset = np.array(dataset)
    # center = np.mean(dataset, axis=0)
    # pca.fit(dataset)
    # eig_vec = pca.components_
    # plot_dataset_and_pca(dataset, eig_vec)
    dataset_aligned, eigenvectors, mean = align_cisterna(dataset)
    centered_dataset = dataset - mean
    # plot_dataset_moved_and_pca(dataset, dataset_aligned, eigenvectors, mean)
    plot(dataset)
    min_z_index = np.argmin(dataset_aligned[:, 2])
    lowest_point = dataset_aligned[min_z_index]
    max_z_index = np.argmax(dataset_aligned[:, 2])
    highest_point = dataset_aligned[max_z_index]
    current_z_value = lowest_point[2]
    final_list = {}
    while current_z_value <= highest_point[2]:
        # zberi piksle v posamezno cisterno
        filtered_points = dataset_aligned[np.floor(dataset_aligned[:, 2]) == np.floor(current_z_value)]
        # final_points = []
        # for filtered_point in filtered_points:
        #     original_point = eigenvectors @ filtered_point + mean
        #     final_points.append(original_point)
        if len(filtered_points) < 4:
            print()
        final_list[current_z_value] = np.array(filtered_points)
        current_z_value += 1
    bank = []
    final_list1 = []
    for i, key in enumerate(final_list):
        values = final_list[key]
        if len(values) + len(bank) < 10 and i < len(final_list) - 1:
            bank += list(values)
            continue
        if len(bank) > 0:
            values = np.concatenate((values, np.array(bank)))
            bank = []
        final_list1.append(values)
    # flattened = [inner for outer in final_list for inner in outer]
    # plot_dataset_moved_and_pca(centered_dataset, dataset_aligned, None, None)
    # height = eig_vec[0]
    # length = eig_vec[1]
    # find lowest point
    # lowest_point = np.copy(center)
    # point_in_volume = []
    # points_to_cover = []
    # while check_points_in_bounds(lowest_point) and instance_volume[int(lowest_point[0])][int(lowest_point[1])][int(lowest_point[2])] == 1:
    # while check_points_in_bounds(lowest_point):
        # point_in_volume = np.copy([int(lowest_point[0]), int(lowest_point[1]), int(lowest_point[2])])
        # points_to_cover.append([int(lowest_point[0]), int(lowest_point[1]), int(lowest_point[2])])
        # lowest_point -= height
    # postavi najnizjo tocko nazaj v predmet
    # lowest_point += height
    # new_point = np.copy(lowest_point)
    # zberi vse točke
    # while check_points_in_bounds(new_point) and instance_volume[int(new_point[0])][int(new_point[1])][int(new_point[2])] == 1:
    # while check_points_in_bounds(new_point):
    #     points_to_cover.append([new_point[0], new_point[1], new_point[2]])
    #     new_point += height
    # združi točke, ki so v enakem vokslu
    # points_to_cover_final = []
    # for p in points_to_cover:
    #     if len(points_to_cover_final) > 0:
    #         last_point = points_to_cover_final[-1]
    #         if int(last_point[0]) == int(p[0]) and int(last_point[1]) == int(p[1]) and int(last_point[2]) == int(p[2]):
    #             continue
    #     points_to_cover_final.append([p[0], p[1], p[2]])
    # print('original size:', len(points_to_cover), '; new size:', len(points_to_cover_final))
    # points_to_cover = points_to_cover_final
    # plot_points(points_to_cover)
    # print('found', len(points_to_cover), 'points')
    # final_list = []
    # for p in points_to_cover:
    #     if p not in final_list:
    #         final_list.append(p)
    # print('working with', len(final_list), 'points')
    # if len(final_list) == 0:
    #     return []
    # all_points = []
    # counter = 0
    # remaining_points = np.copy(dataset)
    # plane_equations = []
    # for p in final_list:
    #     print('processing point', counter, 'of', len(final_list))
        # print('remaining points:', len(remaining_points))
        # width = np.cross(length, height)
        # all_points.append([])
        # a, b, c, k = get_plane_equation_parameters(height, p)
        # cistarnae_points, remaining_points = get_all_cisternae_points([a, b, c, k], remaining_points)
        # all_cistarnae_points[counter] = cistarnae_points
        # plane_equations.append([a, b, c, k])
        # counter += 1
    # for point in remaining_points:
    #     closest_index = get_all_cisternae_points_from_all(point, plane_equations)
        # if closest_index not in all_cistarnae_points:
        #     all_cistarnae_points[closest_index] = []
        # all_points[closest_index].append(point)
    # print('END remaining points:', len(remaining_points))
    # plot_ga_instances(dataset, 0, 0, 0, 0, final_list)
    for i in range(len(final_list1)):
        final_list1[i] = np.array(final_list1[i], dtype=object)
    return final_list1, eigenvectors
    # for i in final_list1:
    #     final_list1[i] = np.array(final_list1[i], dtype=object)
    # return final_list1, eigenvectors


def get_plane_equation_parameters(normal, point_on_plane):
    k = -(normal[0] * int(point_on_plane[0]) + normal[1] * int(point_on_plane[1]) + normal[2] * int(point_on_plane[2]))
    return normal[0], normal[1], normal[2], k


def get_all_cisternae_points(plane_equation, instance_volume):
    cisternae = []
    for point in instance_volume:
        denominator = np.sqrt(plane_equation[0] ** 2 + plane_equation[1] ** 2 + plane_equation[2] ** 2)
        solution = (plane_equation[0] * point[0] + plane_equation[1] * point[1] + plane_equation[2] * point[2] + plane_equation[3]) / denominator
        if np.abs(solution) < 1:
            cisternae.append(point)
            index = np.where((instance_volume == point).all(axis=1))[0]
            instance_volume = np.delete(instance_volume, index, axis=0)
    return np.array(cisternae), instance_volume


# return index of nearest cisternae center
def get_all_cisternae_points_from_all(point, plane_equations):
    index, min_distance = -1, -1
    for i, plane_equation in enumerate(plane_equations):
        denominator = np.sqrt(plane_equation[0] ** 2 + plane_equation[1] ** 2 + plane_equation[2] ** 2)
        distance = (plane_equation[0] * point[0] + plane_equation[1] * point[1] + plane_equation[2] * point[2] +
                    plane_equation[3]) / denominator
        distance = np.abs(distance)
        if distance == 0:
            return i
        if min_distance == -1 or distance < min_distance:
            min_distance = distance
            index = i
    return index

# method from https://stackoverflow.com/questions/67108932/align-pca-component-with-cartesian-axis-with-rotation
def align_cisternae_to_axis(cisterna):
    if cisterna.scalle[0] < 3:
        print('cisterna too small, ignore it')
        return None
    mean = cisterna.mean(axis=0)
    # mean = np.array([int(mean[0]), int(mean[1]), int(mean[2])])
    centered_points = cisterna - mean
    centered_points = np.array(centered_points, dtype='float64')
    # pca = PCA(n_components=3)
    # pca.fit(centered_points)
    # covariance_matrix = pca.components_
    # values, vectors = np.linalg.eigh(covariance_matrix)
    # rotation_matrix = vectors.T
    # result = np.dot(centered_points, covariance_matrix)
    U, S, Vt = np.linalg.svd(centered_points)
    result = centered_points @ Vt.T
    # plot_original_and_aligned_cisterna(cisterna, result)
    return result


def align_cisterna(ga_object):
    ga_object = np.array(ga_object)
    mean = ga_object.mean(axis=0)
    # mean = np.array([int(mean[0]), int(mean[1]), int(mean[2])])
    centered_points = ga_object - mean
    centered_points = np.array(centered_points, dtype='float64')
    # U, S, Vt = np.linalg.svd(centered_points)
    pca = PCA(n_components=3)
    pca.fit_transform(centered_points)
    matrix = pca.components_
    result = centered_points @ matrix
    return result, matrix, mean


# todo: razdeli v learning in test dataset
if __name__ == '__main__':
    for filename in os.listdir(data_directory):
        relative_file_path = data_directory + '/' + filename
        print('processing file:', filename)
        nib_image = nib.load(relative_file_path)
        image_data = nib_image.get_fdata()
        ga_instances = extract_ga_instances(image_data)
        all_ga_data, all_eigenvectors = {}, {}
        for instance in ga_instances:
            print('processing instance ', instance)
            cisternae, eigenvectors = read_files(image_data, filename, ga_instances[instance])
            if len(cisternae) <= 3:
                print('ignoring instance', instance, '; length = ', len(cisternae))
                continue
            all_ga_data[instance] = cisternae
            all_eigenvectors[instance] = eigenvectors
            print('done with cisternae extraction')
            print('found', len(cisternae), 'cisternas')
            # plot_ga_instances(None, None, None, None, None, cisternae)
            # new_list = []
            # for c in cisternae:
            #     aligned = align_cisternae_to_axis(c)
            #     if aligned is not None:
            #         new_list.append(aligned)
            # all_ga_data[instance] = new_list
        print('done with processing, saving files')
        new_filename = filename.replace('.nii.gz', '')
        # with open('../ga_instances/' + new_filename, 'w') as outfile:
        #     writer = csv.writer(outfile)
        #     for data in all_ga_data:
        #         writer.writerow(all_ga_data[data])
        #         writer.writerow({})
        for d in all_ga_data:
            # with open('../ga_instances/' + new_filename, 'w') as outfile:
            #     yaml.dump(all_ga_data[d], outfile, default_flow_style=False)
            # data = [all_ga_data[d]]
            filename = '../ga_instances/' + new_filename + '_' + str(d)
            np.savez(filename, *all_ga_data[d])
            filename += '_ev'
            np.savez(filename, *all_eigenvectors[d])
        # save_files(nib_image, ga_instances, filename)
