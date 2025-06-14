import math

import numpy as np

import math_utils


def calculate_bezier_derivative(n, points, t):
    # from 0 to n - 1
    value = 0
    for i in range(n):
        q = n * (points[i + 1] - points[i])
        value += math_utils.calculate_B(n - 1, i, t) * q
    return value


def calculate_bezier_second_derivative(n, points, t):
    value = 0
    for i in range(n - 1):
        q_i = points[i + 1] - points[i]
        q_i_plus_1 = points[i + 2] - points[i + 1]
        value += math_utils.calculate_B(n - 2, i, t) * (n - 1) * (q_i_plus_1 - q_i)
    return value


def calculate_bezier_third_derivative(n, points, t):
    value = n * (n - 1) * (n - 2)
    for i in range(n - 2):
        value += (math_utils.calculate_B(n - 3, i, t) * (points[i + 3] - 3 * points[i + 2] + 3 * points[i + 1] - points[i]))
    return value


def calculate_bezier_point(n, points, t):
    value = 0
    # from 0 to n
    for i in range(n + 1):
        value += math_utils.calculate_B(n, i, t) * points[i]
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
        length += math.dist(points[i], points[i + 1])
    # Cubic interpolator
    # calculate derivative at t = 0 and t = 1
    t0_deriv_scaled = math_utils.magnitude(calculate_bezier_derivative(n, points, 0)) / length
    t1_deriv_scaled = math_utils.magnitude(calculate_bezier_derivative(n, points, 1)) / length
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
        if f_i > 1:
            print()
        reference_point = calculate_bezier_point(n, points, f_i)
        arc_length_approx.append(reference_point)
        prev_l = l_i
    return result, arc_length_approx, length


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
        bezier_curve, arc_length_parametrization, length = bezier_nth_order_and_parametrization(n, control_points, num_points_on_the_curve)
        return bezier_curve, arc_length_parametrization, length
    else:
        print('not enough points to construct Bezier curve')
        return None, None, None
