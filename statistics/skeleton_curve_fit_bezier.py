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
    t_space = np.linspace(0, 1, 50)
    for t in range(len(t_space)):
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

    control_point1_index = int(0.33 * len(points))
    control_point2_index = int(0.66 * len(points))
    control_point1 = points[control_point1_index]
    control_point2 = points[control_point2_index]
    whole_curve = cubic_bezier(points[0], control_point1, control_point2, points[len(points) - 1])

    new_points = []

    for point in whole_curve:
        p = [round(point[0]), round(point[1]), round(point[2])]
        if p not in new_points:
            new_points.append(p)
            writer.writerow(p)

    new_points = np.array(new_points)

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
            'ro:',  # Styling (red, circles, dotted).
        )
        ax.plot(
            new_points[:, 0],  # x-coordinates.
            new_points[:, 1],  # y-coordinates.
            new_points[:, 2],  # y-coordinates.
            'yo:',  # Styling (yellow, circles, dotted).
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.view_init(50, 20)
        plt.title(filename)
        plt.show()


