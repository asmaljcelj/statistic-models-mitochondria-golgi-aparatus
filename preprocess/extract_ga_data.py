import os

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from networkx.algorithms.threshold import eigenvalues
from sklearn.decomposition import PCA

def check_points_in_bounds(point):
    return 0 <= int(point[0]) <= 255 and 0 <= int(point[1]) <= 255 and 0 <= int(point[2]) <= 255


def plot_ga_instances(points, vector, vector2, center, lowest_point, cistarnae):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='3D Points')
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, c in enumerate(cistarnae):
        ax.scatter(cistarnae[c][:, 0], cistarnae[c][:, 1], cistarnae[c][:, 2], c=colors[i % len(colors)], label='cistarnae' + str(i))

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
    ax.legend()

    # Show the plot
    plt.show()


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


# def check_if_boundary_point(current_point, instance_voxels):
#     number_of_instance_neighbors = 0
#     left = [current_point[0] - 1, current_point[1], current_point[2]]
#     if instance_voxels[left[0], left[1], left[2]] == 1:
#         number_of_instance_neighbors += 1
#     right = [current_point[0] + 1, current_point[1], current_point[2]]
#     if instance_voxels[right[0], right[1], right[2]] == 1:
#         number_of_instance_neighbors += 1
#     top = [current_point[0], current_point[1], current_point[2] + 1]
#     if instance_voxels[top[0], top[1], top[2]] == 1:
#         number_of_instance_neighbors += 1
#     bot = [current_point[0], current_point[1], current_point[2] - 1]
#     if instance_voxels[bot[0], bot[1], bot[2]] == 1:
#         number_of_instance_neighbors += 1
#     front = [current_point[0], current_point[1] - 1, current_point[2]]
#     if instance_voxels[front[0], front[1], front[2]] == 1:
#         number_of_instance_neighbors += 1
#     back = [current_point[0], current_point[1] + 1, current_point[2]]
#     if instance_voxels[back[0], back[1], back[2]] == 1:
#         number_of_instance_neighbors += 1
#     return number_of_instance_neighbors <= 5


# def get_point_cloud(instance_points, instance_voxels):
#     point_cloud = []
#     for current_point in instance_points:
#         if check_if_boundary_point(current_point, instance_voxels):
#             point_cloud.append(current_point)
#     return point_cloud



def extract_ga_instances(volume):
    # instance_volume hrani vrednosti, kateremu GA pripada voksel, ce 0 -> ne pripada nobenemu
    # covered_volume: vrednost 0 -> voksel se ni bil obdelan, 1 -> je ze bil obdelan
    instance_volume, covered_volume, instantiated_volume = np.zeros(volume.shape), np.zeros(volume.shape), np.zeros(volume.shape)
    current_instance = 1
    ga_instances = {}
    for x in range(len(instance_volume)):
        for y in range(len(instance_volume[x])):
            for z in range(len(instance_volume[x][y])):
                if volume[x][y][z] == 1 and covered_volume[x][y][z] == 0:
                    print('found new instance at', x, '-', y, '-', z)
                    ga_instances[current_instance], covered_volume, instantiated_volume = get_entire_instance_2([x, y, z], volume, covered_volume, instantiated_volume)
                    current_instance += 1
    print('found', current_instance - 1, 'instances in this volume')
    for index in ga_instances:
        print('instance', index, 'has', len(ga_instances[index]), 'voxels')
        # point_cloud = get_point_cloud(ga_instances[index], instance_volume)
        # print()
    return ga_instances

instances_folder = '../ga_instances'

# def save_files(image_data, data, og_filename):
#     print('saving files')
#     counter = 1
#     pca = PCA(n_components=2)
#     for key in data:
#         new_filename = og_filename[:og_filename.find('.')] + '_' + str(counter)
#         # counter += 1
#         voxels = data[key]
#         # final_instance_object = np.zeros(image_data.shape)
#         # for voxel in voxels:
#         #     final_instance_object[voxel[0], voxel[1], voxel[2]] = 1
#         np.savetxt('../ga_instances/' + new_filename + '.csv', voxels, delimiter=',', fmt='%-0d')
#         dataset = pd.read_csv(instances_folder + '/' + new_filename + '.csv')
#         pca.fit(dataset)
#         eig_vec = pca.components_
#         plot_ga_instances(data[key], eig_vec)
#         # first_unit = eig_vec[0]
#         # # for t in range(1, 15):
#         # #     point = (t * first_unit[0], t * first_unit[1], t * first_unit[2])
#         # #     final_instance_object[int(point[0])][int(point[1])][int(point[2])] = 120
#         #
#         # new_image = nib.Nifti1Image(final_instance_object, image_data.affine)
#         # nib.save(new_image, extracted_data_directory + '/' + new_filename)

def read_files(instance_volume, filename, dataset):
    # filename = filename.replace('.nii.gz', '')
    all_cistarnae_points = {}
    # for i in range(1, 10000):
        # filepath = instances_folder + '/' + filename + '_' + str(i) + '.csv'
        # if not os.path.exists(filepath):
        #     break
    pca = PCA(n_components=2)
    # dataset = pd.read_csv(filepath)
    # dataset = np.array(dataset)
    center = np.mean(dataset, axis=0)
    pca.fit(dataset)
    eig_vec = pca.components_
    height = eig_vec[0]
    # length = eig_vec[1]
    # find lowest point
    lowest_point = np.copy(center)
    # point_in_volume = []
    points_to_cover = []
    while check_points_in_bounds(lowest_point) and instance_volume[int(lowest_point[0])][int(lowest_point[1])][int(lowest_point[2])] == 1:
        # point_in_volume = np.copy([int(lowest_point[0]), int(lowest_point[1]), int(lowest_point[2])])
        # points_to_cover.append([int(lowest_point[0]), int(lowest_point[1]), int(lowest_point[2])])
        lowest_point -= height
    lowest_point += height
    new_point = np.copy(lowest_point)
    while check_points_in_bounds(new_point) and instance_volume[int(new_point[0])][int(new_point[1])][int(new_point[2])] == 1:
        points_to_cover.append([int(new_point[0]), int(new_point[1]), int(new_point[2])])
        new_point += height
    print('found', len(points_to_cover), 'points')
    final_list = []
    for p in points_to_cover:
        if p not in final_list:
            final_list.append(p)
    print('working with', len(final_list), 'points')
    if len(final_list) == 0:
        return []
    all_points = []
    counter = 0
    remaining_points = np.copy(dataset)
    plane_equations = []
    for p in final_list:
        print('processing point', counter, 'of', len(final_list))
        # print('remaining points:', len(remaining_points))
        # todo: poberi posamezne cisterne
        # width = np.cross(length, height)
        all_points.append([])
        a, b, c, k = get_plane_equation_parameters(height, p)
        # cistarnae_points, remaining_points = get_all_cisternae_points([a, b, c, k], remaining_points)
        # all_cistarnae_points[counter] = cistarnae_points
        plane_equations.append([a, b, c, k])
        # todo: statistika za posamezno cisterno
        # todo: statistični model???
        counter += 1
    for point in remaining_points:
        closest_index = get_all_cisternae_points_from_all(point, plane_equations)
        # if closest_index not in all_cistarnae_points:
        #     all_cistarnae_points[closest_index] = []
        all_points[closest_index].append(point)
    # print('END remaining points:', len(remaining_points))
    # plot_ga_instances(dataset, height, eig_vec[1], center, point_in_volume, all_cistarnae_points)
    for i in range(len(final_list)):
        all_points[i] = np.array(all_points[i], dtype=object)
    return all_points


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


def align_cisternae_to_axis(cisterna):
    if cisterna.shape[0] < 3:
        print('cisterna too small, ignore it')
        return None
    mean = np.mean(cisterna, axis=0)
    centered_points = cisterna - mean
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    covariance_matrix = pca.components_
    values, vectors = np.linalg.eigh(covariance_matrix)
    rotation_matrix = vectors.T
    return np.dot(centered_points, rotation_matrix)


for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()
    ga_instances = extract_ga_instances(image_data)
    all_ga_data = {}
    for instance in ga_instances:
        print('processing instance ', instance)
        cistarnae = read_files(image_data, filename, ga_instances[instance])
        new_list = []
        for c in cistarnae:
            aligned = align_cisternae_to_axis(c)
            if aligned is not None:
                new_list.append(aligned)
        all_ga_data[instance] = new_list
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
    # save_files(nib_image, ga_instances, filename)



