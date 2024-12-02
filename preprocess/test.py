import csv
import yaml
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
#     data1 = [data[d]]
#     np.savez('d', *data1)

test = np.load('d.npz', allow_pickle=True)
print()
