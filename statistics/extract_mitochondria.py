import os
import shutil
import time

import numpy as np

import utils


def read_voxels(volumetric_data):
    # read every voxel in 3d grid, if it has non-zero value, store it in the correct space
    # because we use instance representation every non-zero value in the grid means that voxels belongs to a specific instance
    mitochondria_instances, instances_on_edge = {}, []
    for x, data_yz in enumerate(volumetric_data):
        for y, data_z in enumerate(data_yz):
            for z, value in enumerate(data_z):
                if value != 0:
                    if value not in mitochondria_instances:
                        mitochondria_instances[value] = []
                    if value in instances_on_edge:
                        continue
                    mitochondria_instances[value].append([x, y, z])
                    # add objects on edge in a seperate list
                    if x == 0 or x == 255 or y == 0 or y == 255 or z == 0 or z == 255:
                        instances_on_edge.append(value)
    return mitochondria_instances, instances_on_edge


def remove_empty_space_from_object(instance):
    # remove empty space from object
    # find max and min coordinate with value
    min_x, max_x, min_y, max_y, min_z, max_z = -1, -1, -1, -1, -1, -1
    for x, data_yz in enumerate(instance):
        for y, data_z in enumerate(data_yz):
            for z, value in enumerate(data_z):
                if value == 1:
                    if z < min_z or min_z == -1:
                        min_z = z
                    if z > max_z:
                        max_z = z
                    if y < min_y or min_y == -1:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    if x < min_x or min_x == -1:
                        min_x = x
                    if x > max_x:
                        max_x = x
    # create new smaller object
    final_instance_object = np.zeros((max_x - min_x + 10, max_y - min_y + 10, max_z - min_z + 10))
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            for z in range(min_z, max_z + 1):
                final_instance_object[i - min_x + 1][j - min_y + 1][z - min_z + 1] = instance[i][j][z]
    return final_instance_object


# learning_testing_split: ratio, how to split instances into learning and testing set
def extract_mitochondria(data_directory, extracted_data_directory):
    # read every file from specified directory
    for filename in os.listdir(data_directory):
        start_time = time.time()
        relative_file_path = data_directory + '/' + filename
        print('processing file:', filename)
        volumetric_data, image = utils.read_nii_data(relative_file_path)
        mitochondria_instances, instances_on_edge = read_voxels(volumetric_data)

        # # read voxels
        # for x, data_yz in enumerate(image_data):
        #     for y, data_z in enumerate(data_yz):
        #         for z, value in enumerate(data_z):
        #             if value != 0:
        #                 if value not in mitochondria_instances:
        #                     mitochondria_instances[value] = []
        #                 if value in instances_on_edge:
        #                     continue
        #                 mitochondria_instances[value].append([x, y, z])
        #                 # eliminate objects that are on the edge as they don't represent full shape
        #                 if x == 0 or x == 255 or y == 0 or y == 255 or z == 0 or z == 255:
        #                     instances_on_edge.append(value)
        # print('done with object extraction for file', filename)

        image_shape = image.shape
        print('beginning object processing for file', filename)
        # process each object and save it to separate file
        for instance in mitochondria_instances:
            # before processing, if the
            new_filename = filename[:filename.find('.')] + '_' + str(int(instance))
            # if new_filename in invalid_instances:
            #     continue
            # if object is on the edge, ignore it as it doesn't reflect full, realistic shape
            if instance in instances_on_edge:
                continue
            # initialize grid the size of original where we populate the object
            isolated_instance_object = np.zeros((image_shape[0], image_shape[1], image_shape[2]))
            # put value 1 on those voxels
            for instance_voxel in mitochondria_instances[instance]:
                isolated_instance_object[instance_voxel[0]][instance_voxel[1]][instance_voxel[2]] = 1

            # remove empty space from object
            # find max and min coordinate with value
            # min_x, max_x, min_y, max_y, min_z, max_z = -1, -1, -1, -1, -1, -1
            # for x, data_yz in enumerate(isolated_instance_object):
            #     for y, data_z in enumerate(data_yz):
            #         for z, value in enumerate(data_z):
            #             if value == 1:
            #                 if z < min_z or min_z == -1:
            #                     min_z = z
            #                 if z > max_z:
            #                     max_z = z
            #                 if y < min_y or min_y == -1:
            #                     min_y = y
            #                 if y > max_y:
            #                     max_y = y
            #                 if x < min_x or min_x == -1:
            #                     min_x = x
            #                 if x > max_x:
            #                     max_x = x
            # # create new smaller object
            # final_instance_object = np.zeros((max_x - min_x + 10, max_y - min_y + 10, max_z - min_z + 10))
            # for i in range(min_x, max_x + 1):
            #     for j in range(min_y, max_y + 1):
            #         for z in range(min_z, max_z + 1):
            #             final_instance_object[i - min_x + 1][j - min_y + 1][z - min_z + 1] = isolated_instance_object[i][j][z]
            final_instance_object = remove_empty_space_from_object(isolated_instance_object)
            # new_image = nib.Nifti1Image(final_instance_object, image.affine)
            # nib.save(new_image, extracted_data_directory + '/' + new_filename)
            # save instance as .nii file
            utils.save_new_object(final_instance_object, new_filename, extracted_data_directory, image)
        print('done with file', filename, 'time taken:', time.time() - start_time, 'seconds')
    print('done with data extraction')


def split_learn_test(data_directory, split_ratio=80):
    # split the data into learning and testing set
    nii_files = [f for f in os.listdir(data_directory) if f.endswith('.nii')]
    num_split = int(len(nii_files) * split_ratio / 100)
    index = 0
    for filename in os.listdir(data_directory):
        if not filename.endswith('.nii'):
            continue
        if index < num_split:
            shutil.copy(os.path.join(data_directory, filename), os.path.join(data_directory + '/learning', filename))
        else:
            shutil.copy(os.path.join(data_directory, filename), os.path.join(data_directory + '/test', filename))
        index += 1


data_directory = '../data'
extracted_data_directory = '../extracted_data'
# these instances represent shapes that aren't fully captured in the volume so we manually removed them from
# todo: poglej, kaj točno je s temi primerki???
invalid_instances = [
    'fib1-0-0-0_10.nii',
    # non-normal shape
    'fib1-0-0-0_33.nii',
    'fib1-1-0-3_2.nii',
    'fib1-1-0-3_3.nii',
    'fib1-3-2-1_1.nii',
    'fib1-3-2-1_2.nii',
    'fib1-3-2-1_8.nii',
    'fib1-3-3-0_4.nii',
    'fib1-3-3-0_14.nii',
    'fib1-3-3-0_28.nii',
    'fib1-3-3-0_56.nii',
    'fib1-3-3-0_59.nii',
    'fib1-4-3-0_3.nii',
    'fib1-4-3-0_7.nii',
    'fib1-4-3-0_14.nii',
    'fib1-4-3-0_38.nii'
]
extract_mitochondria(data_directory, extracted_data_directory)
split_learn_test(extracted_data_directory)
