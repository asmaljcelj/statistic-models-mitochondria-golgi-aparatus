import os

import nibabel as nib
import numpy as np

data_directory = '../data_ga/approximate'
extracted_data_directory = '../extracted_ga_data'


def process_voxel(voxel, volume, covered_volume, instance_volume, instance_index):
    if voxel[0] < 0 or voxel[0] > 255 or voxel[1] < 0 or voxel[1] > 255 or voxel[2] < 0 or voxel[2] > 255:
        # invalid range
        return None
    if volume[voxel[0]][voxel[1]][voxel[2]] != 0:
        # voxel represent GA instance
        if covered_volume[voxel[0]][voxel[1]][voxel[2]] == 0:
            # GA wasn't covered yet
            covered_volume[voxel[0]][voxel[1]][voxel[2]] = 1
            instance_volume[voxel[0]][voxel[1]][voxel[2]] = instance_index
        # this voxel was processed already, get its instance
        return instance_volume[voxel[0]][voxel[1]][voxel[2]]
    return None


def find_instance(current_point, covered_volume, instance_volume, volume, current_instance):
    left = [current_point[0] - 1, current_point[1], current_point[2]]
    instance = process_voxel(left, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance
    right = [current_point[0] + 1, current_point[1], current_point[2]]
    instance = process_voxel(right, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance
    top = [current_point[0], current_point[1], current_point[2] + 1]
    instance = process_voxel(top, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance
    bot = [current_point[0], current_point[1], current_point[2] - 1]
    instance = process_voxel(bot, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance
    front = [current_point[0], current_point[1] - 1, current_point[2]]
    instance = process_voxel(front, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance
    back = [current_point[0], current_point[1] + 1, current_point[2]]
    instance = process_voxel(back, volume, covered_volume, instance_volume, current_instance)
    if instance is not None:
        return instance


def extract_ga_instances(volume):
    # instance_volume hrani vrednosti, kateremu GA pripada voksel, ce 0 -> ne pripada nobenemu
    # covered_volume: vrednost 0 -> voksel se ni bil obdelan, 1 -> je ze bil obdelan
    instance_volume, covered_volume = np.zeros(volume.shape), np.zeros(volume.shape)
    current_instance = 1
    for x in range(len(instance_volume)):
        for y in range(len(instance_volume[x])):
            for z in range(len(instance_volume[x][y])):
                if covered_volume[x][y][z] == 0:
                    # voxel wasn't processed yet
                    covered_volume[x][y][z] = 1
                    if volume[x][y][z] == 1:
                        # unprocessed voxel is a GA
                        instance = find_instance([x, y, z], covered_volume, instance_volume, volume, current_instance)
                        if instance is None:
                            print('found new instance at', x, '-', y, '-', z)
                            instance = current_instance
                            current_instance += 1
                        instance_volume[x][y][z] = instance
    print('found', current_instance, 'instances in this volume')


for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()

    extract_ga_instances(image_data)
