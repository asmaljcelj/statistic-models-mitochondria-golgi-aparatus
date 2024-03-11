import csv
import os

import numpy as np
from matplotlib import pyplot as plt


def multiply_num_with_list(list, num):
    return [element * num for element in list]


def sum_same_elements(list1, list2):
    return [el1 + el2 for el1, el2 in zip(list1, list2)]


def cubic_bezier(p0, p1, p2, p3):
    result = []
    t_space = np.linspace(0, 1, 20)
    for t in range(len(t_space) - 1):
        result.append(
            sum_same_elements(
                sum_same_elements(
                    multiply_num_with_list(p0, (1 - t_space[t])**3),
                    multiply_num_with_list(p1, 3 * (1 - t_space[t])**2 * t_space[t])
                ),
                sum_same_elements(
                    multiply_num_with_list(p2, 3 * (1 - t_space[t]) * t_space[t]**2),
                    multiply_num_with_list(p3, t_space[t]**3)
                )
            )
        )
    return result


skeletons_folder = '../skeletons/'

number_of_plots = 10
current_plot_count = 0

for filename in os.listdir(skeletons_folder):
    if filename.endswith('.nii'):
        continue
    print('processing', filename)
    file_path = skeletons_folder + filename

    f = open('../skeletons_bezier/' + filename, 'w', newline='')
    writer = csv.writer(f)

    x, y, z = [], [], []
    points = []

    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            x.append(int(row[0]))
            y.append(int(row[1]))
            z.append(int(row[2]))
            points.append([int(row[0]), int(row[1]), int(row[2])])

    points1 = np.array(points)

    whole_curve = []
    for i in range(0, len(points), 3):
        if i > len(points) - 4:
            # todo: how to handle last elements if they are not dividable by 4?
            # remaining = len(points) - i
            # i -= (4 - remaining)
            # whole_curve += cubic_bezier(points[i], points[i + 1], points[i + 2], points[i + 3])
            break
        whole_curve += cubic_bezier(points[i], points[i + 1], points[i + 2], points[i + 3])

    for point in whole_curve:
        writer.writerow([point[0], point[1], point[2]])

    f.close()

    if current_plot_count < number_of_plots:
        whole_curve = np.array(whole_curve)
        current_plot_count += 1
        # PLOTTING
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(
            whole_curve[:, 0],  # x-coordinates.
            whole_curve[:, 1],  # y-coordinates.
            whole_curve[:, 2],  # y-coordinates.
        )
        ax.plot(
            points1[:, 0],  # x-coordinates.
            points1[:, 1],  # y-coordinates.
            points1[:, 2],  # y-coordinates.
            'ro:'  # Styling (red, circles, dotted).
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(50, 20)
        plt.title(filename)
        plt.show()


