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
    print('done with object extraction')

    image_shape = image.shape

    # process each object and save it to separate file
    for object in objects:
        new_object = np.zeros((image_shape[0], image_shape[1], image_shape[2]))
        # if object is on the edge, don't create new file as
        if objects[object][0]:
            continue
        for voxel in objects[object]:
            if type(voxel) == list:
                new_object[voxel[0]][voxel[1]][voxel[2]] = 1
        new_filename = filename[:filename.find('.')] + '_' + str(int(object))
        new_image = nib.Nifti1Image(new_object, image.affine)
        nib.save(new_image, target_directory + '/' + new_filename)
