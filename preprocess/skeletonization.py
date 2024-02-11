import nibabel as nib
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import skeletonize_3d

folder = '../extracted_data'
results_folder = '../skeletons'

for filename in os.listdir(folder):
    filepath = folder + '/' + filename

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

    new_image = nib.Nifti1Image(result, image.affine)
    nib.save(new_image, results_folder + '/' + filename + '_skeleton')

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

    interpolated_image = nib.Nifti1Image(smoothed_result, image.affine)
    nib.save(interpolated_image, results_folder + '/' + filename + '_skeleton_interpolated')
