import os
import random

import nibabel as nib
import numpy as np

import pandas as pd
from sklearn.decomposition import PCA

data_directory = '../data_ga/approximate'
extracted_data_directory = '../extracted_ga_data'


def check_if_voxel_is_ga(volume, point, points_to_process, instance_voxels, covered_volume):
    if point[0] < 0 or point[0] > 255 or point[1] < 0 or point[1] > 255 or point[2] < 0 or point[2] > 255:
        # invalid range
        return
    if volume[point[0]][point[1]][point[2]] != 0 and covered_volume[point[0]][point[1]][point[2]] == 0:
        points_to_process.append(point)
        instance_voxels.append(point)


def get_entire_instance_2(start_point, volume, covered_volume):
    points_to_process = [start_point]
    instance_voxels = [start_point]
    while points_to_process:
        current_point = points_to_process.pop(0)
        if covered_volume[current_point[0]][current_point[1]][current_point[2]] != 0:
            continue
        covered_volume[current_point[0]][current_point[1]][current_point[2]] = 1
        left = [current_point[0] - 1, current_point[1], current_point[2]]
        check_if_voxel_is_ga(volume, left, points_to_process, instance_voxels, covered_volume)
        right = [current_point[0] + 1, current_point[1], current_point[2]]
        check_if_voxel_is_ga(volume, right, points_to_process, instance_voxels, covered_volume)
        top = [current_point[0], current_point[1], current_point[2] + 1]
        check_if_voxel_is_ga(volume, top, points_to_process, instance_voxels, covered_volume)
        bot = [current_point[0], current_point[1], current_point[2] - 1]
        check_if_voxel_is_ga(volume, bot, points_to_process, instance_voxels, covered_volume)
        front = [current_point[0], current_point[1] - 1, current_point[2]]
        check_if_voxel_is_ga(volume, front, points_to_process, instance_voxels, covered_volume)
        back = [current_point[0], current_point[1] + 1, current_point[2]]
        check_if_voxel_is_ga(volume, back, points_to_process, instance_voxels, covered_volume)
        left_up_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, left_up_front, points_to_process, instance_voxels, covered_volume)
        left_front = [current_point[0] - 1, current_point[1] - 1, current_point[2]]
        check_if_voxel_is_ga(volume, left_front, points_to_process, instance_voxels, covered_volume)
        left_down_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, left_down_front, points_to_process, instance_voxels, covered_volume)
        left_up = [current_point[0] - 1, current_point[1], current_point[2] + 1]
        check_if_voxel_is_ga(volume, left_up, points_to_process, instance_voxels, covered_volume)
        left_down = [current_point[0] - 1, current_point[1], current_point[2] - 1]
        check_if_voxel_is_ga(volume, left_down, points_to_process, instance_voxels, covered_volume)
        left_back_up = [current_point[0] - 1, current_point[1] + 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, left_back_up, points_to_process, instance_voxels, covered_volume)
        left_back_down = [current_point[0] - 1, current_point[1] + 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, left_back_down, points_to_process, instance_voxels, covered_volume)
        left_back = [current_point[0] - 1, current_point[1] + 1, current_point[2]]
        check_if_voxel_is_ga(volume, left_back, points_to_process, instance_voxels, covered_volume)
        front_up = [current_point[0], current_point[1] - 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, front_up, points_to_process, instance_voxels, covered_volume)
        front_down = [current_point[0], current_point[1] - 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, front_down, points_to_process, instance_voxels, covered_volume)
        back_up = [current_point[0], current_point[1] + 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, back_up, points_to_process, instance_voxels, covered_volume)
        back_down = [current_point[0], current_point[1] + 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, back_down, points_to_process, instance_voxels, covered_volume)
        right_front_up = [current_point[0] + 1, current_point[1] - 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, right_front_up, points_to_process, instance_voxels, covered_volume)
        right_front = [current_point[0] + 1, current_point[1] - 1, current_point[2]]
        check_if_voxel_is_ga(volume, right_front, points_to_process, instance_voxels, covered_volume)
        right_front_down = [current_point[0] + 1, current_point[1] - 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, right_front_down, points_to_process, instance_voxels, covered_volume)
        right_up = [current_point[0] + 1, current_point[1], current_point[2] + 1]
        check_if_voxel_is_ga(volume, right_up, points_to_process, instance_voxels, covered_volume)
        right_down = [current_point[0] + 1, current_point[1], current_point[2] - 1]
        check_if_voxel_is_ga(volume, right_down, points_to_process, instance_voxels, covered_volume)
        right_back_up = [current_point[0] + 1, current_point[1] + 1, current_point[2] + 1]
        check_if_voxel_is_ga(volume, right_back_up, points_to_process, instance_voxels, covered_volume)
        right_back = [current_point[0] + 1, current_point[1] + 1, current_point[2]]
        check_if_voxel_is_ga(volume, right_back, points_to_process, instance_voxels, covered_volume)
        right_back_down = [current_point[0] + 1, current_point[1] + 1, current_point[2] - 1]
        check_if_voxel_is_ga(volume, right_back_down, points_to_process, instance_voxels, covered_volume)
    return instance_voxels


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
    left_up_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] + 1]
    if instance_voxels[left_up_front[0], left_up_front[1], left_up_front[2]] == 1:
        number_of_instance_neighbors += 1
    left_front = [current_point[0] - 1, current_point[1] - 1, current_point[2]]
    if instance_voxels[left_front[0], left_front[1], left_front[2]] == 1:
        number_of_instance_neighbors += 1
    left_down_front = [current_point[0] - 1, current_point[1] - 1, current_point[2] - 1]
    if instance_voxels[left_down_front[0], left_down_front[1], left_down_front[2]] == 1:
        number_of_instance_neighbors += 1
    left_up = [current_point[0] - 1, current_point[1], current_point[2] + 1]
    if instance_voxels[left_up[0], left_up[1], left_up[2]] == 1:
        number_of_instance_neighbors += 1
    left_down = [current_point[0] - 1, current_point[1], current_point[2] - 1]
    if instance_voxels[left_down[0], left_down[1], left_down[2]] == 1:
        number_of_instance_neighbors += 1
    left_back_up = [current_point[0] - 1, current_point[1] + 1, current_point[2] + 1]
    if instance_voxels[left_back_up[0], left_back_up[1], left_back_up[2]] == 1:
        number_of_instance_neighbors += 1
    left_back_down = [current_point[0] - 1, current_point[1] + 1, current_point[2] - 1]
    if instance_voxels[left_back_down[0], left_back_down[1], left_back_down[2]] == 1:
        number_of_instance_neighbors += 1
    left_back = [current_point[0] - 1, current_point[1] + 1, current_point[2]]
    if instance_voxels[left_back[0], left_back[1], left_back[2]] == 1:
        number_of_instance_neighbors += 1
    front_up = [current_point[0], current_point[1] - 1, current_point[2] + 1]
    if instance_voxels[front_up[0], front_up[1], front_up[2]] == 1:
        number_of_instance_neighbors += 1
    front_down = [current_point[0], current_point[1] - 1, current_point[2] - 1]
    if instance_voxels[front_down[0], front_down[1], front_down[2]] == 1:
        number_of_instance_neighbors += 1
    back_up = [current_point[0], current_point[1] + 1, current_point[2] + 1]
    if instance_voxels[back_up[0], back_up[1], back_up[2]] == 1:
        number_of_instance_neighbors += 1
    back_down = [current_point[0], current_point[1] + 1, current_point[2] - 1]
    if instance_voxels[back_down[0], back_down[1], back_down[2]] == 1:
        number_of_instance_neighbors += 1
    right_front_up = [current_point[0] + 1, current_point[1] - 1, current_point[2] + 1]
    if instance_voxels[right_front_up[0], right_front_up[1], right_front_up[2]] == 1:
        number_of_instance_neighbors += 1
    right_front = [current_point[0] + 1, current_point[1] - 1, current_point[2]]
    if instance_voxels[right_front[0], right_front[1], right_front[2]] == 1:
        number_of_instance_neighbors += 1
    right_front_down = [current_point[0] + 1, current_point[1] - 1, current_point[2] - 1]
    if instance_voxels[right_front_down[0], right_front_down[1], right_front_down[2]] == 1:
        number_of_instance_neighbors += 1
    right_up = [current_point[0] + 1, current_point[1], current_point[2] + 1]
    if instance_voxels[right_up[0], right_up[1], right_up[2]] == 1:
        number_of_instance_neighbors += 1
    right_down = [current_point[0] + 1, current_point[1], current_point[2] - 1]
    if instance_voxels[right_down[0], right_down[1], right_down[2]] == 1:
        number_of_instance_neighbors += 1
    right_back_up = [current_point[0] + 1, current_point[1] + 1, current_point[2] + 1]
    if instance_voxels[right_back_up[0], right_back_up[1], right_back_up[2]] == 1:
        number_of_instance_neighbors += 1
    right_back = [current_point[0] + 1, current_point[1] + 1, current_point[2]]
    if instance_voxels[right_back[0], right_back[1], right_back[2]] == 1:
        number_of_instance_neighbors += 1
    right_back_down = [current_point[0] + 1, current_point[1] + 1, current_point[2] - 1]
    if instance_voxels[right_back_down[0], right_back_down[1], right_back_down[2]] == 1:
        number_of_instance_neighbors += 1
    return number_of_instance_neighbors < 9

def get_point_cloud(instance_points, instance_voxels):
    point_cloud = []
    for current_point in instance_points:
        if check_if_boundary_point(current_point, instance_voxels):
            point_cloud.append(current_point)
    return point_cloud



def extract_ga_instances(volume):
    # instance_volume hrani vrednosti, kateremu GA pripada voksel, ce 0 -> ne pripada nobenemu
    # covered_volume: vrednost 0 -> voksel se ni bil obdelan, 1 -> je ze bil obdelan
    instance_volume, covered_volume = np.zeros(volume.shape), np.zeros(volume.shape)
    current_instance = 1
    ga_instances = {}
    for x in range(len(instance_volume)):
        for y in range(len(instance_volume[x])):
            for z in range(len(instance_volume[x][y])):
                if volume[x][y][z] == 1 and covered_volume[x][y][z] == 0:
                    print('found new instance at', x, '-', y, '-', z)
                    ga_instances[current_instance] = get_entire_instance_2([x, y, z], volume, covered_volume)
                    current_instance += 1
    print('found', current_instance - 1, 'instances in this volume')
    for index in ga_instances:
        print('instance', index, 'has', len(ga_instances[index]), 'voxels')
        point_cloud = get_point_cloud(ga_instances[index], instance_volume)
        print(point_cloud)
    return ga_instances

instances_folder = '../ga_instances'

def save_files(image_data, data, og_filename):
    print('saving files')
    counter = 1
    pca = PCA(n_components=2)
    for key in data:
        new_filename = og_filename[:og_filename.find('.')] + '_' + str(counter)
        counter += 1
        voxels = data[key]
        final_instance_object = np.zeros(image_data.shape)
        for voxel in voxels:
            final_instance_object[voxel[0], voxel[1], voxel[2]] = 1
        np.savetxt('../ga_instances/' + new_filename + '.csv', voxels, delimiter=',', fmt='%-0d')
        dataset = pd.read_csv(instances_folder + '/' + new_filename + '.csv')
        pca.fit(dataset)
        eig_vec = pca.components_
        first_unit = eig_vec[0]
        # for t in range(1, 15):
        #     point = (t * first_unit[0], t * first_unit[1], t * first_unit[2])
        #     final_instance_object[int(point[0])][int(point[1])][int(point[2])] = 120
        new_image = nib.Nifti1Image(final_instance_object, image_data.affine)
        nib.save(new_image, extracted_data_directory + '/' + new_filename)


for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()
    ga_instances = extract_ga_instances(image_data)
    save_files(nib_image, ga_instances, filename)
