import os
import shutil

file_directory = '../extracted_data'
learning_directory = '../extracted_data/learning'
test_directory = '../extracted_data/test'

split_percentage = 80

# remove curvy mitochondria shapes (shapes that have more than 1 curve)
# manual inspection performed
curvy_instances = [
    'fib1-0-0-0_10.nii',
    # non-normal shape
    'fib1-0-0-0_33.nii',
    'fib1-1-0-3_2.nii',
    'fib1-1-0-3_3.nii',
    'fib1-3-2-1_1.nii',
    'fib1-3-2-1_2.nii',
    'fib1-3-2-1_8.nii',
    'fib1-3-3-0_4.nii',
    'fib1-3-3-0_14.nii',
    'fib1-3-3-0_28.nii',
    'fib1-3-3-0_56.nii',
    'fib1-3-3-0_59.nii',
    'fib1-4-3-0_3.nii',
    'fib1-4-3-0_7.nii',
    'fib1-4-3-0_14.nii',
    'fib1-4-3-0_38.nii'
]

for filename in os.listdir(file_directory):
    if filename in curvy_instances:
        os.remove(os.path.join(file_directory, filename))

nii_files = [f for f in os.listdir(file_directory) if f.endswith('.nii')]
num_split = int(len(nii_files) * split_percentage / 100)
index = 0

for filename in os.listdir(file_directory):
    if not filename.endswith('.nii'):
        continue
    if index < num_split:
        shutil.copy(os.path.join(file_directory, filename), os.path.join(learning_directory, filename))
    else:
        shutil.copy(os.path.join(file_directory, filename), os.path.join(test_directory, filename))
    index += 1
