def calculate_average(skeleton_distances, start_distances, end_distances):
    average_skeleton_distances = {}
    # calculate average skeleton distances
    for skeleton_distance in skeleton_distances:
        distances = skeleton_distances[skeleton_distance]
        for i, distances_on_point in distances.items():
            if i not in average_skeleton_distances:
                average_skeleton_distances[i] = {}
            for j, distance in enumerate(distances_on_point):
                if j not in average_skeleton_distances[i]:
                    average_skeleton_distances[i][j] = [0, 0]
                average_skeleton_distances[i][j][0] += distance
                average_skeleton_distances[i][j][1] += 1
    # calculate averages
    for i in average_skeleton_distances:
        for j in average_skeleton_distances[i]:
            average_skeleton_distances[i][j][0] /= average_skeleton_distances[i][j][1]
    print(average_skeleton_distances)
    # calculate average start distances
    average_start_distances = {}
    for start_distance in start_distances:
        distances = start_distances[start_distance]
        for i, distances_on_point in distances.items():
            if i not in average_start_distances:
                average_start_distances[i] = [0, 0]
            average_start_distances[i][0] += distances_on_point
            average_start_distances[i][1] += 1
    for i in average_start_distances:
        average_start_distances[i][0] /= average_start_distances[i][1]
    print(average_start_distances)
    # calculate average end distances
    average_end_distances = {}
    for end_distance in end_distances:
        distances = end_distances[end_distance]
        for i, distances_on_point in distances.items():
            if i not in average_end_distances:
                average_end_distances[i] = [0, 0]
            average_end_distances[i][0] += distances_on_point
            average_end_distances[i][1] += 1
    for i in average_end_distances:
        average_end_distances[i][0] /= average_end_distances[i][1]
    print(average_end_distances)
    return average_skeleton_distances, average_start_distances, average_end_distances
