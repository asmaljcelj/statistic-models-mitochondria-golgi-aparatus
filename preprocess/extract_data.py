import nibabel as nib
import os
import numpy as np
import time

data_directory = '../data'
extracted_data_directory = '../extracted_data'

for filename in os.listdir(data_directory):
    start_time = time.time()
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()
    mitochondria_instances = {}
    instances_on_edge = []

    # read voxels
    for x, data_yz in enumerate(image_data):
        for y, data_z in enumerate(data_yz):
            for z, value in enumerate(data_z):
                if value != 0:
                    if value not in mitochondria_instances:
                        mitochondria_instances[value] = []
                    if value in instances_on_edge:
                        continue
                    mitochondria_instances[value].append([x, y, z])
                    # eliminate objects that are on the edge as they don't represent full shape
                    if x == 0 or x == 255 or y == 0 or y == 255 or z == 0 or z == 255:
                        instances_on_edge.append(value)
    print('done with object extraction for file', filename)

    image_shape = nib_image.shape
    print('beginning object processing for file', filename)
    # process each object and save it to separate file
    for instance in mitochondria_instances:
        # if object is on the edge, don't create new file
        if instance in instances_on_edge:
            continue
        isolated_instance_object = np.zeros((image_shape[0], image_shape[1], image_shape[2]))
        for instance_voxel in mitochondria_instances[instance]:
            isolated_instance_object[instance_voxel[0]][instance_voxel[1]][instance_voxel[2]] = 1

        # remove empty space from object
        # find max and min coordinate with value
        min_x, max_x, min_y, max_y, min_z, max_z = -1, -1, -1, -1, -1, -1
        for x, data_yz in enumerate(isolated_instance_object):
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
                    final_instance_object[i - min_x + 1][j - min_y + 1][z - min_z + 1] = isolated_instance_object[i][j][z]
        # save instance as .nii file
        new_filename = filename[:filename.find('.')] + '_' + str(int(instance))
        new_image = nib.Nifti1Image(final_instance_object, nib_image.affine)
        nib.save(new_image, extracted_data_directory + '/' + new_filename)
    print('done with file', filename, 'time taken:', time.time() - start_time, 'seconds')

print('done with data extraction')
