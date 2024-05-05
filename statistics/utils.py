import csv
import time
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


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


def plot_sampling_with_shape(shape, origin, distance):
    # todo: verificiraj, ali je okej samplano!!!
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(shape, facecolors=[0, 0, 1, 0.4])
    ax.view_init(azim=50, elev=10)
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
