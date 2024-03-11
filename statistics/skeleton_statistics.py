import csv
import os

bezier_folder = '../skeletons_bezier'
raw_folder = '../skeletons'

for filename in os.listdir(raw_folder):
    raw_reader = csv.reader(raw_folder + '/' + filename)
    number_of_points = sum(1 for _ in raw_reader)
    first_point, last_point = [], []
    for i, row in enumerate(raw_reader):
        if i == 0:
            first_point = row
        if i == number_of_points - 1:
            last_point = row
    print()



    # bezier_reader = csv.reader(bezier_folder + '/' + filename)


