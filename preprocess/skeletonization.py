import os

import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

folder = '../extracted_data'
results_folder = '../skeletons'


def sort_points(points):
    all_points = []
    for i in range(len(points[0])):
        all_points.append([points[0][i], points[1][i], points[2][i]])
    all_points = np.array(all_points, dtype='uint8')
    sorted_points = []
    tree = KDTree(all_points)
    current_point = all_points[0]
    while len(sorted_points) < len(all_points):
        dist, nearest_index = tree.query(current_point)
        current_point = all_points[nearest_index]
        sorted_points.append(current_point)
        all_points = np.delete(all_points, nearest_index, axis=0)
        tree = KDTree(all_points)
    return sorted_points


for filename in os.listdir(folder):
    filepath = folder + '/' + filename
    print('processing file', filename)

    image = nib.load(filepath)
    array = np.array(image.get_fdata())
    result = skeletonize(array)

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

    # sorted_values = sort_points(final_points)
    np.savetxt('../skeletons/' + os.path.splitext(filename)[0] + '.csv', final_points, delimiter=',', fmt='%-0d')

    # save interpolated result as nii image to view the skeleton
    # interpolated_image = nib.Nifti1Image(smoothed_result, image.affine)
    # nib.save(interpolated_image, results_folder + '/skeleton_interpolated_' + filename)
