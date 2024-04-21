import csv
import os
import time

import numpy as np
from matplotlib import pyplot as plt

skeletons_folder = '../skeletons/'
number_of_plots = 10
current_plot_count = 0
plotting_enabled = True


def multiply_num_with_list(list, num):
    return [element * num for element in list]


def sum_same_elements(list1, list2):
    return [el1 + el2 for el1, el2 in zip(list1, list2)]


def magnitude(point):
    return np.sqrt(np.sum(point ** 2))


def calculate_bezier_derivative(P0, P1, P2, P3, t):
    return 3 * (1 - t) ** 2 * (P1 - P0) + 6 * (1 - t) * t * (P2 - P1) + 3 * t ** 2 * (P3 - P2)


def cubic_bezier(p0, p1, p2, p3):
    result = []
    t_space = np.linspace(0, 1, 7)
    for t in t_space:
        result.append(
            # sum_same_elements(
            #     sum_same_elements(
            #         multiply_num_with_list(p0, (1 - t_space[t]) ** 3),
            #         multiply_num_with_list(p1, 3 * (1 - t_space[t]) ** 2 * t_space[t])
            #     ),
            #     sum_same_elements(
            #         multiply_num_with_list(p2, 3 * (1 - t_space[t]) * t_space[t] ** 2),
            #         multiply_num_with_list(p3, t_space[t] ** 3)
            #     )
            # )
            (1 - t) ** 3 * p0 +
            3 * (1 - t) ** 2 * t * p1 +
            3 * (1 - t) * t ** 2 * p2 +
            t ** 3 * p3
        )
    # arc length parametrization
    # get length of cubic bezier curve (Simpson rule)
    translation = p0
    p0_translated = np.array([0, 0, 0])
    p1_translated = p1 - translation
    p2_translated = p2 - translation
    p3_translated = p3 - translation

    steps = 1000
    length = 0
    dt = 1 / steps
    for i in range(steps):
        t0 = i * dt
        t1 = (i + 1) * dt
        length += (
                    magnitude(calculate_bezier_derivative(p0_translated, p1_translated, p2_translated, p3_translated, t0)) +
                    4 * magnitude(calculate_bezier_derivative(p0_translated, p1_translated, p2_translated, p3_translated, (t0 + t1) / 2)) +
                    magnitude(calculate_bezier_derivative(p0_translated, p1_translated, p2_translated, p3_translated, t1))
        ) * dt / 6
    print('length is', length)
    # calculate derivative at t = 0 and t = 1
    N = 7
    t0_deriv_scaled = magnitude(calculate_bezier_derivative(p0, p1, p2, p3, 0)) / length
    t1_deriv_scaled = magnitude(calculate_bezier_derivative(p0, p1, p2, p3, 1)) / length
    c = 1 / t0_deriv_scaled
    a = c + 1 / t1_deriv_scaled - 2
    b = 1 - c - a
    delta_L = 1 / N
    l_0 = 0
    prev_l = l_0
    arc_length_approx = []
    for i in range(1, N + 1):
        l_i = prev_l + delta_L
        f_i = ((a * l_i + b) * l_i + c) * l_i
        reference_point = (1 - f_i) ** 3 * p0 + 3 * (1 - f_i) ** 2 * f_i * p1 + 3 * (1 - f_i) * f_i ** 2 * p2 + f_i ** 3 * p3
        arc_length_approx.append(reference_point)
        prev_l = l_i
    return result, arc_length_approx


def extract_points():
    x_coord, y_coord, z_coord, points = [], [], [], []
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        x_coord.append(int(row[0]))
        y_coord.append(int(row[1]))
        z_coord.append(int(row[2]))
        points.append([int(row[0]), int(row[1]), int(row[2])])
    return x_coord, y_coord, z_coord, points


def generate_points_between_points(p1, p2, num_of_points):
    p1 = np.array(p1)
    p2 = np.array(p2)

    step = 1.0 / (num_of_points + 1)
    points = [p1 + (p2 - p1) * i for i in np.arange(step, 1, step)]
    return points


def plot_save_result(num_of_points, bezier_curve, original_points, arc_length_approx):
    # plot the result and save it
    if num_of_points == number_of_plots:
        time.sleep(15)
        num_of_points = 0
    bezier_curve = np.array(bezier_curve)
    arc_length_approx = np.array(arc_length_approx)
    num_of_points += 1
    # PLOTTING
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        bezier_curve[:, 0],  # x-coordinates.
        bezier_curve[:, 1],  # y-coordinates.
        bezier_curve[:, 2],  # y-coordinates.
        'o:',
        label='Bezier curve'
    )
    ax.plot(
        arc_length_approx[:, 0],  # x-coordinates.
        arc_length_approx[:, 1],  # y-coordinates.
        arc_length_approx[:, 2],  # y-coordinates.
        'ro:',  # Styling (red, circles, dotted).
        label='Arc length parametrization'
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
    ax.legend()
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


def calculate_statistics(straight_curve, fitted_curve):
    pass


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
            control_point1 = np_points[control_point1_index]
            control_point2 = np_points[control_point2_index]
            whole_curve, approx = cubic_bezier(np_points[0], control_point1, control_point2, np_points[len(points) - 1])
        else:
            # take first half of the curve
            control_point1_index = int(0.16 * len(points))
            control_point2_index = int(0.33 * len(points))
            end_point2_index = int(0.5 * len(points))
            control_point1 = np_points[control_point1_index]
            control_point2 = np_points[control_point2_index]
            end_point2 = np_points[end_point2_index]
            whole_curve, approx = cubic_bezier(np_points[0], control_point1, control_point2, end_point2)
            # take second half of the curve
            start_point = end_point2
            control_point1_index = int(0.66 * len(points))
            control_point2_index = int(0.83 * len(points))
            end_point2_index = len(points) - 1
            control_point1 = np_points[control_point1_index]
            control_point2 = np_points[control_point2_index]
            end_point2 = np_points[end_point2_index]
            whole_curve2, approx2 = cubic_bezier(start_point, control_point1, control_point2, end_point2)
            whole_curve += whole_curve2
            approx += approx2

        # new_points = write_curve_to_file(whole_curve)

        if plotting_enabled:
            plot_save_result(current_plot_count, whole_curve, np_points, approx)

        # straight_line = generate_points_between_points(new_points[0], new_points[len(new_points) - 1], len(new_points))
