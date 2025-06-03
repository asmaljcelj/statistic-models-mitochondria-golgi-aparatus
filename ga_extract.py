import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from sklearn.decomposition import PCA

import utils


def extract_ga_instances(volume):
    labeled, num_stacks = ndi.label(volume)
    ga_instances = {}
    print('found', num_stacks, 'instances in this volume')
    for i in range(num_stacks):
        ga_instances[(i + 1)] = np.array(np.where(labeled == (i + 1))).T
    return ga_instances


def read_files(dataset):
    dataset = np.array(dataset)
    dataset_aligned, eigenvectors, mean = align_cisterna(dataset)
    # utils.plot_ga_object(dataset)
    min_z_index = np.argmin(dataset_aligned[:, 2])
    lowest_point = dataset_aligned[min_z_index]
    max_z_index = np.argmax(dataset_aligned[:, 2])
    highest_point = dataset_aligned[max_z_index]
    current_z_value = lowest_point[2]
    final_list = {}
    while current_z_value <= highest_point[2]:
        # zberi voksle v posamezno cisterno
        filtered_points = dataset_aligned[np.floor(dataset_aligned[:, 2]) == np.floor(current_z_value)]
        if len(filtered_points) < 4:
            print()
        final_list[current_z_value] = np.array(filtered_points)
        current_z_value += 1
    bank = []
    final_list1 = []
    for i, key in enumerate(final_list):
        values = final_list[key]
        if len(values) + len(bank) < 10 and i < len(final_list) - 1:
            bank += list(values)
            continue
        if len(bank) > 0:
            values = np.concatenate((values, np.array(bank)))
            bank = []
        final_list1.append(values)
    for i in range(len(final_list1)):
        final_list1[i] = np.array(final_list1[i], dtype=object)
    return final_list1, eigenvectors


def align_cisterna(ga_object):
    ga_object = np.array(ga_object)
    mean = ga_object.mean(axis=0)
    centered_points = ga_object - mean
    centered_points = np.array(centered_points, dtype='float64')
    pca = PCA(n_components=3)
    pca.fit_transform(centered_points)
    matrix = pca.components_
    result = centered_points @ matrix
    return result, matrix, mean


data_directory = 'data_ga/approximate'

for filename in os.listdir(data_directory):
    relative_file_path = data_directory + '/' + filename
    print('processing file:', filename)
    nib_image = nib.load(relative_file_path)
    image_data = nib_image.get_fdata()
    ga_instances = extract_ga_instances(image_data)
    all_ga_data, all_eigenvectors = {}, {}
    for instance in ga_instances:
        print('processing instance ', instance)
        cisternae, eigenvectors = read_files(ga_instances[instance])
        if len(cisternae) <= 3:
            print('ignoring instance', instance, '; length = ', len(cisternae))
            continue
        all_ga_data[instance] = cisternae
        all_eigenvectors[instance] = eigenvectors
        print('found', len(cisternae), 'cisternae')
    print('done with processing, saving files')
    new_filename = filename.replace('.nii.gz', '')
    for d in all_ga_data:
        filename = 'ga_instances/' + new_filename + '_' + str(d)
        np.savez(filename, *all_ga_data[d])
        filename += '_ev'
        np.savez(filename, *all_eigenvectors[d])
