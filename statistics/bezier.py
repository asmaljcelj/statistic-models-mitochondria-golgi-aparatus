import math
import os

import numpy as np
from math_utils import magnitude, extract_points, calculate_B
from utils import plot_save_result

skeletons_folder = '../skeletons/'
number_of_plots = 10
current_plot_count = 0
plotting_enabled = True


def calculate_bezier_derivative(n, points, t):
    # from 0 to n - 1
    value = 0
    for i in range(n):
        q = n * (points[i + 1] - points[i])
        value += calculate_B(n - 1, i, t) * q
    return value


def calculate_bezier_point(n, points, t):
    value = 0
    # from 0 to n
    for i in range(n + 1):
        value += calculate_B(n, i, t) * points[i]
    return value


def bezier_nth_order_and_parametrization(n, points, num_of_points_on_curve):
    if len(points) != (n + 1):
        print('wrong number of points: expecting ', n + 1, 'points for ', n, 'th order Bezier curve')
    t_space = np.linspace(0, 1, num_of_points_on_curve)
    result = []
    for t in t_space:
        value = calculate_bezier_point(n, points, t)
        result.append(value)
    length = 0
    # calculate length of the curve
    for i in range(len(points) - 1):
        length += math.dist(points[0], points[1])
    # Cubic interpolator
    # calculate derivative at t = 0 and t = 1
    t0_deriv_scaled = magnitude(calculate_bezier_derivative(n, points, 0)) / length
    t1_deriv_scaled = magnitude(calculate_bezier_derivative(n, points, 1)) / length
    c = 1 / t0_deriv_scaled
    a = c + 1 / t1_deriv_scaled - 2
    b = 1 - c - a
    delta_L = 1 / num_of_points_on_curve
    l_0 = 0
    prev_l = l_0
    arc_length_approx = [calculate_bezier_point(n, points, 0)]
    for i in range(num_of_points_on_curve):
        l_i = prev_l + delta_L
        f_i = ((a * l_i + b) * l_i + c) * l_i
        reference_point = calculate_bezier_point(n, points, f_i)
        arc_length_approx.append(reference_point)
        prev_l = l_i
    return result, arc_length_approx


def perform_arc_length_parametrization_bezier_curve(n, points, num_points_on_the_curve):
    np_points = np.array(points)
    control_points = [np_points[0]]
    # če je v skeletonu premalo točk (manj kot n) -> zmanjšaj stopnjo Bezierjeve krivulje
    if len(points) < n + 1:
        n = len(points) - 1
    else:
        n = 5
    if n > 0:
        ratio = 1 / n
        for i in range(1, n):
            # from 1 to n
            control_point_index = int(i * ratio * len(points))
            control_points.append(np_points[control_point_index])
        control_points.append(np_points[len(points) - 1])
        bezier_curve, arc_length_parametrization = bezier_nth_order_and_parametrization(n, control_points, num_points_on_the_curve)
        return bezier_curve, arc_length_parametrization
    else:
        print('not enough points to construct Bezier curve')
        return None, None


if __name__ == '__main__':
    for filename in os.listdir(skeletons_folder):
        if filename.endswith('.nii'):
            continue
        print('processing', filename)
        file_path = skeletons_folder + filename

        n = 5
        with open(file_path) as csv_file:
            points = extract_points(csv_file)

            np_points = np.array(points)

            whole_curve, approx = perform_arc_length_parametrization_bezier_curve(n, np_points, 10)
            # char_points = [np_points[0]]
            # # če je v skeletonu premalo točk (manj kot n) -> zmanjšaj stopnjo Bezierjeve krivulje
            # if len(points) < n + 1:
            #     n = len(points) - 1
            # else:
            #     n = 5
            # if n > 0:
            #     ratio = 1 / n
            #     for i in range(1, n):
            #         # from 1 to n
            #         control_point_index = int(i * ratio * len(points))
            #         char_points.append(np_points[control_point_index])
            #     char_points.append(np_points[len(points) - 1])
            #     whole_curve, approx = bezier_nth_order_and_parametrization(n, char_points, 10)
            #     if plotting_enabled:
            plot_save_result(current_plot_count, whole_curve, np_points, approx, number_of_plots, filename)
            # else:
            #     print('not enough points to construct Bezier curve')
