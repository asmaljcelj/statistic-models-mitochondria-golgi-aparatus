import nibabel as nib
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from skimage.morphology import skeletonize_3d

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
    result = skeletonize_3d(array)

    points_x, points_y, points_z = [], [], []
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            for z in range(result.shape[2]):
                if result[x][y][z] > 0:
                    points_x.append(x)
                    points_y.append(y)
                    points_z.append(z)

    x_space = np.linspace(0, result.shape[0], result.shape[0], endpoint=False)
    y_space = np.linspace(0, result.shape[1], result.shape[1], endpoint=False)
    z_space = np.linspace(0, result.shape[2], result.shape[2], endpoint=False)

    interp = RegularGridInterpolator((x_space, y_space, z_space), result, method='cubic')

    smoothed_result = np.zeros((result.shape[0], result.shape[1], result.shape[2]), dtype='uint8')
    nonzero_values = np.nonzero(interp.values)
    for i in range(len(nonzero_values[0])):
        x = nonzero_values[0][i]
        y = nonzero_values[1][i]
        z = nonzero_values[2][i]
        if interp([x, y, z]) > 100:
            smoothed_result[x][y][z] = 255

    sorted_values = sort_points(nonzero_values)
    np.savetxt('../skeletons/' + os.path.splitext(filename)[0] + '.csv', sorted_values, delimiter=',', fmt='%-0d')

    # save interpolated result as nii image to view the skeleton
    # interpolated_image = nib.Nifti1Image(smoothed_result, image.affine)
    # nib.save(interpolated_image, results_folder + '/skeleton_interpolated_' + filename)
