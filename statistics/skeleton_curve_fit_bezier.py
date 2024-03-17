import csv
import os
import time

import numpy as np
from matplotlib import pyplot as plt

skeletons_folder = '../skeletons/'
number_of_plots = 10
current_plot_count = 0
plotting_enabled = False


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
                    multiply_num_with_list(p0, (1 - t_space[t]) ** 3),
                    multiply_num_with_list(p1, 3 * (1 - t_space[t]) ** 2 * t_space[t])
                ),
                sum_same_elements(
                    multiply_num_with_list(p2, 3 * (1 - t_space[t]) * t_space[t] ** 2),
                    multiply_num_with_list(p3, t_space[t] ** 3)
                )
            )
        )
    return result


def extract_points():
    x_coord, y_coord, z_coord, points = [], [], [], []
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        x_coord.append(int(row[0]))
        y_coord.append(int(row[1]))
        z_coord.append(int(row[2]))
        points.append([int(row[0]), int(row[1]), int(row[2])])
    return x_coord, y_coord, z_coord, points


def plot_save_result(num_of_points, bezier_curve, original_points):
    # plot the result and save it
    if num_of_points == number_of_plots:
        time.sleep(15)
        num_of_points = 0
    bezier_curve = np.array(bezier_curve)
    num_of_points += 1
    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        bezier_curve[:, 0],  # x-coordinates.
        bezier_curve[:, 1],  # y-coordinates.
        bezier_curve[:, 2],  # y-coordinates.
    )
    ax.plot(
        original_points[:, 0],  # x-coordinates.
        original_points[:, 1],  # y-coordinates.
        original_points[:, 2],  # y-coordinates.
        'ro:',  # Styling (red, circles, dotted).
    )
    # new calculated points
    # ax.plot(
    #     new_points[:, 0],  # x-coordinates.
    #     new_points[:, 1],  # y-coordinates.
    #     new_points[:, 2],  # y-coordinates.
    #     'yo:',  # Styling (yellow, circles, dotted).
    # )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(50, 20)
    plt.title(filename)
    plt.savefig('../plots/' + filename + '.png')
    plt.close()


def write_curve_to_file(curve):
    new_points = []
    f = open('../skeletons_bezier/' + filename, 'w', newline='')
    writer = csv.writer(f)
    for point in curve:
        p = [round(point[0]), round(point[1]), round(point[2])]
        if p not in new_points:
            new_points.append(p)
            writer.writerow(p)
    f.close()
    return np.array(new_points)


for filename in os.listdir(skeletons_folder):
    if filename.endswith('.nii'):
        continue
    print('processing', filename)
    file_path = skeletons_folder + filename

    with open(file_path) as csv_file:
        x, y, z, points = extract_points()

        np_points = np.array(points)

        if len(points) < 7:
            control_point1_index = int(0.33 * len(points))
            control_point2_index = int(0.66 * len(points))
            control_point1 = points[control_point1_index]
            control_point2 = points[control_point2_index]
            whole_curve = cubic_bezier(points[0], control_point1, control_point2, points[len(points) - 1])
        else:
            # take first half of the curve
            control_point1_index = int(0.16 * len(points))
            control_point2_index = int(0.33 * len(points))
            end_point2_index = int(0.5 * len(points))
            control_point1 = points[control_point1_index]
            control_point2 = points[control_point2_index]
            end_point2 = points[end_point2_index]
            whole_curve = cubic_bezier(points[0], control_point1, control_point2, end_point2)
            # take second half of the curve
            start_point = end_point2
            control_point1_index = int(0.66 * len(points))
            control_point2_index = int(0.83 * len(points))
            end_point2_index = len(points) - 1
            control_point1 = points[control_point1_index]
            control_point2 = points[control_point2_index]
            end_point2 = points[end_point2_index]
            whole_curve += cubic_bezier(start_point, control_point1, control_point2, end_point2)

        new_points = write_curve_to_file(whole_curve)

        if plotting_enabled:
            plot_save_result(current_plot_count, whole_curve, np_points)
