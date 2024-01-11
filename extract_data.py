import nibabel as nib
import os

directory = 'data'
objects = {}

for filename in os.listdir(directory):
    full_file_name = directory + '/' + filename
    print('processing file:', filename)
    image = nib.load(full_file_name)
    image_array = image.get_fdata()

    # read voxels
    for z, seznam_xy in enumerate(image_array):
        for y, seznam_x in enumerate(seznam_xy):
            for x, vrednost in enumerate(seznam_x):
                if vrednost != 0:
                    if vrednost not in objects:
                        objects[vrednost] = []
                    objects[vrednost].append([x, y, z])
print('done')

# process each obejct

