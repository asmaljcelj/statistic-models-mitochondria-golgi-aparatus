import os

file_directory = '../extracted_data'

# remove curvy mitochondria shapes (shapes that have more than 1 curve)
# manual inspection performed
curvy_instances = [
    'fib1-0-0-0_10.nii',
    'fib1-1-0-3_2.nii',
    'fib1-1-0-3_3.nii',
    'fib1-3-2-1_1.nii',
    'fib1-3-2-1_2.nii',
    'fib1-3-2-1_8.nii',
    'fib1-3-3-0_4.nii',
    'fib1-3-3-0_14.nii',
    'fib1-3-3-0_28.nii',
    'fib1-4-3-0_3.nii',
    'fib1-4-3-0_7.nii',
    'fib1-4-3-0_14.nii'
]

for filename in os.listdir(file_directory):
    if filename in curvy_instances:
        os.remove(os.path.join(file_directory, filename))
