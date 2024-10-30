import os

import nibabel as nib
import numpy as np

data_directory = '../data_ga/approximate'
extracted_data_directory = '../extracted_ga_data'


def find_next_instance_start(volume, covered_volume):
    for i in range(len(covered_volume)):
        for j in range(len(covered_volume[i])):
            for k in range(len(covered_volume[i][j])):
                if covered_volume[i][j][k] == 0:
                    covered_volume[i][j][k] = 1
                    if volume[i][j][k] == 0:
                        # continue searching for next value
                        continue
                    return i, j, k
    return None, None, None


def process_voxel(voxel, volume, covered_volume, instance_volume):
    if voxel[0] < 0 or voxel[0] > 255 or voxel[1] < 0 or voxel[1] > 255 or voxel[2] < 0 or voxel[2] > 255:
        # invalid range
        return None
    if covered_volume[voxel[0]][voxel[1]][voxel[2]] == 1 and instance_volume[voxel[0]][voxel[1]][voxel[2]] > 0:
        return instance_volume[voxel[0]][voxel[1]][voxel[2]]
    return None


def find_instance(volume, current_point, covered_volume, instance_volume):
    left = [current_point[0] - 1, current_point[1], current_point[2]]
    instance = process_voxel(left, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    right = [current_point[0] + 1, current_point[1], current_point[2]]
    instance = process_voxel(right, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    top = [current_point[0], current_point[1], current_point[2] + 1]
    instance = process_voxel(top, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    bot = [current_point[0], current_point[1], current_point[2] - 1]
    instance = process_voxel(bot, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    front = [current_point[0], current_point[1] - 1, current_point[2]]
    instance = process_voxel(front, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    back = [current_point[0], current_point[1] + 1, current_point[2]]
    instance = process_voxel(back, volume, covered_volume, instance_volume)
    if instance is not None:
        return instance
    # for point in indices_to_search:
    #     if point is not None:
    #         instance_volume = find_whole_object(volume, point, covered_volume, instance_volume, instance_index)
    # return instance_volume


def extract_ga_instances(volume):
    # instance_volume hrani vrednosti, kateremu GA pripada voksel, ce 0 -> ne pripada nobenemu
    # covered_volume: vrednost 0 -> voksel se ni bil obdelan, 1 -> je ze bil obdelan
    instance_volume, covered_volume = np.zeros(volume.shape), np.zeros(volume.shape)
    # instance_index = 1
    # while 0 in covered_volume:
    #     x, y, z = find_next_instance_start(volume, covered_volume)
    #     if x is None and y is None and z is None:
    #         # we found all instances
    #         return instance_volume
    #     instance_volume[x][y][z] = instance_index
    #     find_whole_object(volume, [x, y, z], covered_volume, instance_volume, instance_index)
    #     instance_index += 1
    # instances = {}
    # for x in range(len(instance_volume)):
    #     for y in range(len(instance_volume[x])):
    #         for z in range(len(instance_volume[x][y])):
    #             if instance_volume[x][y][z] > 0:
    #                 index = instance_volume[x][y][z] - 1
    #                 if index not in instances:
    #                     instances[index] = []
    #                 instances[index].append([x, y, z])
    biggest_instance = 0
    for x in range(len(instance_volume)):
        for y in range(len(instance_volume[x])):
            for z in range(len(instance_volume[x][y])):
                if covered_volume[x][y][z] == 0:
                    # voxel wasn't processed yet
                    covered_volume[x][y][z] = 1
                    if volume[x][y][z] > 0:
                        instance = find_instance(volume, [x, y, z], covered_volume, instance_volume)
                        if instance is None:
                            biggest_instance += 1
                            instance = biggest_instance
                        instance_volume[x][y][z] = instance
    print('found', biggest_instance - 1, 'instances in this volume')


for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()

    extract_ga_instances(image_data)
