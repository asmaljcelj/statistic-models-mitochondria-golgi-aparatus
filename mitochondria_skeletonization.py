import os

import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_voxels(voxels):
    # skeletonize and return the list of voxels that represent skeleton
    result = skeletonize(voxels)
    points_x, points_y, points_z = [], [], []
    final_points = []
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            for z in range(result.shape[2]):
                if result[x][y][z] > 0:
                    points_x.append(x)
                    points_y.append(y)
                    points_z.append(z)
                    final_points.append([x, y, z])
    return final_points


def perform_skeletonization(folder, results_folder):
    # extract skeleton from each mitochondria object
    for filename in os.listdir(folder):
        if not filename.endswith('.nii'):
            continue
        filepath = folder + '/' + filename
        print('processing file', filename)
        image = nib.load(filepath)
        array = np.array(image.get_fdata())
        final_points = skeletonize_voxels(array)
        if len(final_points) == 0:
            continue
        np.savetxt('skeletons/' + results_folder + '/' + os.path.splitext(filename)[0] + '.csv', final_points, delimiter=',', fmt='%-0d')


perform_skeletonization('extracted_data' + '/learning', 'learn')
perform_skeletonization('extracted_data' + '/test', 'test')
