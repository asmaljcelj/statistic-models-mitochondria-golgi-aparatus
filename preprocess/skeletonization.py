import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize_3d
import scipy as sp
import scipy.interpolate
from scipy.interpolate import make_interp_spline
from scipy.interpolate import RegularGridInterpolator

filepath = '../extracted_data/fib1-0-0-0_5.nii'

image = nib.load(filepath)
array = np.array(image.get_fdata())
result = skeletonize_3d(array)

# result[result == 255] = 1

points_x, points_y, points_z = [], [], []
for x in range(result.shape[0]):
    for y in range(result.shape[1]):
        for z in range(result.shape[2]):
            if result[x][y][z] > 0:
                points_x.append(x)
                points_y.append(y)
                points_z.append(z)
#
# for x in range(min_x, max_x + 1):
#     for y in range(min_y, max_y + 1):
#         for z in range(min_z, max_z + 1):
#             if result[x][y][z] == 0:
#                 points_x.append(x)
#                 points_y.append(y)
#                 points_z.append(z)

# spline = sp.interpolate.Rbf(points_x, points_y, points_z, [1 for _ in points_x], smooth=0, episilon=5)
#
x_space = np.linspace(0, result.shape[0], result.shape[0], endpoint=False)
y_space = np.linspace(0, result.shape[1], result.shape[1], endpoint=False)
z_space = np.linspace(0, result.shape[2], result.shape[2], endpoint=False)
# B1, B2, B3 = np.meshgrid(x_space, y_space, z_space)
#
# Z = spline(B1, B2, B3)
#
# interpolated = np.zeros((Z.shape[0], Z.shape[1], Z.shape[2]))
# for x in range(Z.shape[0]):
#     for y in range(Z.shape[1]):
#         for z in range(Z.shape[2]):
#             if Z[x][y][z] >= 1:
#                 interpolated[x][y][z] = 1

new_image = nib.Nifti1Image(result, image.affine)
nib.save(new_image, 'skeleton')

# new_image_interpolated = nib.Nifti1Image(interpolated, image.affine)
# nib.save(new_image_interpolated, 'interpolated')

# another interpolation method:
# result = make_interp_spline(points_x, np.array([points_y, points_z]), axis=1)
# print(result)

# another one
interp = RegularGridInterpolator((x_space, y_space, z_space), result, method='cubic')
print(interp)

smoothed_result = np.zeros((result.shape[0], result.shape[1], result.shape[2]))
for x in range(result.shape[0]):
    for y in range(result.shape[1]):
        for z in range(result.shape[2]):
            value = interp([x, y, z])
            if value > 100:
                smoothed_result[x][y][z] = 1
interpolated_image = nib.Nifti1Image(smoothed_result, image.affine)
nib.save(interpolated_image, 'interpolated')

