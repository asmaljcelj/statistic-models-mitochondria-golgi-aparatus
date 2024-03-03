from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import csv
import numpy as np
import os
from Bezier import Bezier


def function(xy, a, b, c, d, e, f, g, h, i):
    x, y = xy
    # return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y
    return a * x ** 3 + b * y ** 2 + c * x ** 2 * y + d * x * y ** 2 + e * x ** 2 + f * y ** 2 + g * x + h * y + i


def multiply_num_with_list(list, num):
    return [element * num for element in list]

def sum_same_elements(list1, list2):
    return [el1 + el2 for el1, el2 in zip(list1, list2)]

def cubic_Bezier(p0, p1, p2, p3):
    result = []
    t_space = np.linspace(0, 1, 20)
    for t in range(len(t_space) - 1):
        result.append(sum_same_elements(sum_same_elements(multiply_num_with_list(p0, (1 - t_space[t])**3), multiply_num_with_list(p1, 3 * (1 - t_space[t])**2 * t_space[t])), sum_same_elements(multiply_num_with_list(p2, 3 * (1 - t_space[t]) * t_space[t]**2), multiply_num_with_list(p3, t_space[t]**3))))
    return result


skeletons_folder = '../skeletons/'

f = open('skeletons_approximations.csv', 'w')
writer = csv.writer(f)

for filename in os.listdir(skeletons_folder):
    print('processing', filename)
    file_path = skeletons_folder + filename

    x, y, z = [], [], []
    points = []

    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            x.append(int(row[0]))
            y.append(int(row[1]))
            z.append(int(row[2]))
            points.append([int(row[0]), int(row[1]), int(row[2])])

    # t_points = np.arange(0, 1, 0.001)
    points1 = np.array(points)
    # curve = Bezier.Curve(t_points, points1)

    whole_curve = []
    for i in range(0, len(points), 3):
        if i > len(points) - 3:
            diff = len(points) % 3

        whole_curve += cubic_Bezier(points[i], points[i + 1], points[i + 2], points[i + 3])

    print('done')
    whole_curve = np.array(whole_curve)


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
    plt.show()
    break

    # arguments, covariance = curve_fit(function, (x, y), z)
    # print('covariance values for', filename)
    # print(covariance)
    #
    # writer.writerow(arguments)

    # PLOTTING ----------------------
    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, color='blue', alpha=0.2)
    # sample_points = []
    # for i in range(len(points) - 1):
    #     points_between = np.linspace(points[i], points[i + 1], 20)[1:-1]
    #     sample_points.extend(points_between)
    # resulting_points = np.array(sample_points)
    # X = np.array(resulting_points[:, 0])
    # Y = np.array(resulting_points[:, 1])
    # X_data, Y_data = np.meshgrid(X, Y)
    # TEMP COMMENT FOR TESTING
    # x_range = np.linspace(min(x), max(x), 1000)
    # y_range = np.linspace(min(y), max(y), 1000)
    # X_1, Y_1 = np.meshgrid(x_range, y_range)
    # END TEMP COMMENT; this is for plotting
    # Z = function((X, Y), *popt)
    # Z_1 = function((X_1, Y_1), *popt)
    # ax.plot(X, Y, Z, color='red', alpha=0.5)
    # ax.view_init(20, 30)
    # plt.show()

f.close()
