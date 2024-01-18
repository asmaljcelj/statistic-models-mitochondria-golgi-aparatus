import nibabel as nib
import os
import numpy as np

directory = 'data'
target_directory = 'extracted_data'

for filename in os.listdir(directory):
    print('working on', filename)
    full_file_name = directory + '/' + filename
    print('processing file:', filename)
    image = nib.load(full_file_name)
    image_array = image.get_fdata()
    objects = {}

    # read voxels
    for z, seznam_xy in enumerate(image_array):
        for y, seznam_x in enumerate(seznam_xy):
            for x, vrednost in enumerate(seznam_x):
                if vrednost != 0:
                    if vrednost not in objects:
                        objects[vrednost] = [False]
                    if objects[vrednost][0]:
                        continue
                    objects[vrednost].append([x, y, z])
                    # eliminate all objects that are on the edge as they don't represent full shape
                    if x == 0 or x == 255 or y == 0 or y == 255 or z == 0 or z == 255:
                        objects[vrednost][0] = True
    print('done with object extraction for file', filename)

    image_shape = image.shape
    print('beginning object processing for file', filename)
    # process each object and save it to separate file
    for object in objects:
        new_object = np.zeros((image_shape[0], image_shape[1], image_shape[2]))
        # if object is on the edge, don't create new file as
        if objects[object][0]:
            continue
        for voxel in objects[object]:
            if type(voxel) == list:
                new_object[voxel[0]][voxel[1]][voxel[2]] = 1
        # remove empty space from object
        # min_x, max_x, min_y, max_y, min_z, max_z = -1, -1, -1, -1, -1, -1
        # for z, dim_xy in enumerate(new_object):
        #     for y, dim_x in enumerate(dim_xy):
        #         for x, element in enumerate(dim_x):
        #             if element == 1:
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
        # print('found min; x:', min_x, 'y:', min_y, 'z:', min_z)
        # print('found max; x:', max_x, 'y:', max_y, 'z:', max_z)
        # final_object = np.zeros((max_z - min_z + 2, max_y - min_y + 2, max_x - min_x + 2))
        # for i in range(min_x, max_x + 1):
        #     for j in range(min_y, max_y + 1):
        #         for z in range(min_z, max_z + 1):
        #             final_object[z - min_z + 1][j - min_y + 1][i - min_x + 1] = new_object[z][j][i]

        new_filename = filename[:filename.find('.')] + '_' + str(int(object))
        new_image = nib.Nifti1Image(new_object, image.affine)
        # new_image = nib.Nifti1Image(final_object, image.affine)
        nib.save(new_image, target_directory + '/' + new_filename)
    break
print('done')
