import csv
# import yaml
import numpy as np

path = '../ga_instances'

data = {
    '1': {
        '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '2': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '3': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    },
    '2': {
        '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '2': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '3': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    },
    '3': {
        '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    },
    '4': {
        '0': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        '1': [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    }
}

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
array_2d_1 = np.array([[1, 2], [3, 4], [6, 7, 8]], dtype=object)
array_2d_2 = np.array([[5, 6], [7, 8]])
array_2d_3 = np.array([[9, 10], [11, 12]])

# List of 2D arrays
list_of_2d_arrays = [array_2d_1, array_2d_2, array_2d_3]

# Create a 3D numpy array from the list of 2D arrays
array_3d = np.array(list_of_2d_arrays, dtype=object)

# Print the 3D array
print(array_3d)
