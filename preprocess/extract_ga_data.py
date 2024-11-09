import os

import nibabel as nib
import numpy as np

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
    return ga_instances


def save_files(image_data, data, og_filename):
    counter = 1
    for key in data:
        new_filename = og_filename[:og_filename.find('.')] + '_' + str(counter)
        counter += 1
        voxels = data[key]
        final_instance_object = np.zeros(image_data.shape)
        for voxel in voxels:
            final_instance_object[voxel[0], voxel[1], voxel[2]] = 1
        new_image = nib.Nifti1Image(final_instance_object, image_data.affine)
        nib.save(new_image, extracted_data_directory + '/' + new_filename)


for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()
    ga_instances = extract_ga_instances(image_data)
    save_files(nib_image, ga_instances, filename)
