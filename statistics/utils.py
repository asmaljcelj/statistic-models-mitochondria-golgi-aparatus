import csv
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D


def plot_save_result(num_of_points, bezier_curve, original_points, arc_length_approx, number_of_plots, filename):
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
    #     original_points[:, 0],  # x-coordinates.
    #     original_points[:, 1],  # y-coordinates.
    #     original_points[:, 2],  # y-coordinates.
    #     'yo:',  # Styling (yellow, circles, dotted).
    #     label='Original skeleton'
    # )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.view_init(50, 20)
    plt.title(filename)
    plt.savefig('../plots/' + filename + '.png')
    plt.close()


def plot_sampling_with_shape(shape, sampled_points, skeleton, parametrized_points):
    matplotlib.use('TkAgg')
    colors = [
        (0.0, 1.0, 0.0),
         (0.1111111111111111, 0.8861111111111111, 0.0),
         (0.2222222222222222, 0.7722222222222221, 0.0),
         (0.3333333333333333, 0.6583333333333333, 0.0),
         (0.4444444444444444, 0.5444444444444444, 0.0),
         (0.5555555555555556, 0.4305555555555556, 0.0),
         (0.6666666666666666, 0.31666666666666665, 0.0),
         (0.7777777777777777, 0.20277777777777775, 0.0),
         (0.8888888888888888, 0.0888888888888889, 0.0),
         (1.0, 0.0, 0.0)
    ]
    count = 0
    # todo: verificiraj, ali je okej samplano!!!
    # sampled_points = np.array(sampled_points)
    skeleton = np.array(skeleton)
    parametrized_points = np.array(parametrized_points)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.voxels(shape, facecolors=[0, 0, 1, 0.2])
    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], 'yo:')
    ax.plot(parametrized_points[:, 0], parametrized_points[:, 1], parametrized_points[:, 2], 'gx:')
    for i in sampled_points:
        list = np.array(sampled_points[i])
        # ax.plot(list[:, 0], list[:, 1], list[:, 2], 'o:', color=colors[count])
        ax.plot(list[:, 0], list[:, 1], list[:, 2], 'co:')
        count += 1
    ax.view_init(azim=-125, elev=-40)
    plt.show()


def read_file_collect_points(filename, base_folder):
    if filename.endswith('.nii'):
        return None
    print('processing', filename)
    file_path = base_folder + filename
    points = []
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            points.append([int(row[0]), int(row[1]), int(row[2])])
        points = np.array(points)
    return points


def read_nii_file(base_folder, filename):
    nib_image = nib.load(base_folder + filename)
    image_data = nib_image.get_fdata()
    return np.array(image_data)

