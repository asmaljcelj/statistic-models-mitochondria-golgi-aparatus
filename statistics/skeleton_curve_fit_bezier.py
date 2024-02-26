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

    t_points = np.arange(0, 1, 0.001)
    points1 = np.array(points)
    curve = Bezier.Curve(t_points, points1)
    print('done')



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        curve[:, 0],  # x-coordinates.
        curve[:, 1],  # y-coordinates.
        curve[:, 2],  # y-coordinates.
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

    # Rotate the axes and update
    # for angle in range(0, 360 * 4 + 1):
    #     # Normalize the angle to the range [-180, 180] for display
    #     angle_norm = (angle + 180) % 360 - 180
    #
    #     # Cycle through a full rotation of elevation, then azimuth, roll, and all
    #     elev = azim = roll = 0
    #     if angle <= 360:
    #         elev = angle_norm
    #     elif angle <= 360 * 2:
    #         azim = angle_norm
    #     elif angle <= 360 * 3:
    #         roll = angle_norm
    #     else:
    #         elev = azim = roll = angle_norm
    #
    #     # Update the axis view and title
    #     ax.view_init(elev=elev, azim=azim, vertical_axis='z')
    #     plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))
    #
    #     plt.draw()
    #     plt.pause(.001)



    # plt.grid()
    ax.view_init(0, 0)
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
