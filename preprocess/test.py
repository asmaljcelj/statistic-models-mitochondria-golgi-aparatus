import csv
import os

# import yaml
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib

# path = '../ga_instances'
#
# data = {
#     '1': {
#         '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '2': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '3': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#     },
#     '2': {
#         '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '2': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '3': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#     },
#     '3': {
#         '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#     },
#     '4': {
#         '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#         '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
#     }
# }

# for i in range(4):
#     with open('../ga_instances/' + str(i) + '.csv', 'w') as outfile:
#         writer = csv.writer(outfile)
#         for d in data:
#             writer.writerow(data[d].values())
#             writer.writerow({})

# for d in data:
#     with open('../ga_instances/' + str(d) + '.yaml', 'w') as outfile:
#         yaml.dump(data[d], outfile, default_flow_style=False)
#         # writer = csv.writer(outfile)
#         # for d in data:
#         #     writer.writerow(data[d].values())
#         #     writer.writerow({})
#
# with open('../ga_instances/1.yaml', 'r') as stream:
#     test = yaml.safe_load(stream)
#     print()
# for d in data:
#     # with open('../ga_instances/' + new_filename, 'w') as outfile:
#     #     yaml.dump(all_ga_data[d], outfile, default_flow_style=False)
#     for i in data[d]:
#         # data1 = np.array(data[d][i])
#         filename = 'd_' + d + '_' + i
#         np.savez(filename, *data[d][i])
#
# test = np.load('d_1_0.npz', allow_pickle=True)
# print()


# Create a list of 2D numpy arrays
# array_2d_1 = np.array([[1, 2], [3, 4], [6, 7, 8]], dtype=object)
# array_2d_2 = np.array([[5, 6], [7, 8]])
# array_2d_3 = np.array([[9, 10], [11, 12]])
#
# # List of 2D arrays
# list_of_2d_arrays = [array_2d_1, array_2d_2, array_2d_3]
#
# # Create a 3D numpy array from the list of 2D arrays
# array_3d = np.array(list_of_2d_arrays, dtype=object)
#
# # Print the 3D array
# print(array_3d)

import matplotlib

# from preprocess.extract_data import mitochondria_instances

skeletons_folder = '../skeletons'
mitochondria_folder = '../extracted_data'

for filename in os.listdir(skeletons_folder):
    file_path = skeletons_folder + '/' + filename
    mitochondria_instances_file_path = mitochondria_folder + '/' + filename.replace('.csv', '.nii')
    matplotlib.use('TkAgg')

    image = nib.load(mitochondria_instances_file_path)
    array = np.array(image.get_fdata())
    array = np.argwhere(array == 1)
    mito_x = [p[0] for p in array]
    mito_y = [p[1] for p in array]
    mito_z = [p[2] for p in array]

    with open(file_path) as csv_file:
        points = []
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            points.append([int(row[0]), int(row[1]), int(row[2])])
        fig = plt.figure()
        # plt.title(filename)
        ax = fig.add_subplot(111, projection='3d')
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]
        ax.scatter(x, y, z, color='red')
        ax.scatter(mito_x, mito_y, mito_z, alpha=0.01)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        plt.grid(b=None)
        plt.show()
