import time

import cv2
import numpy as np
import math
def detect_and_label_0bstacles(map):
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh
def find_points_on_lines(centroid, boundary_coordinates):
    centroid_x, centroid_y = centroid
    horizontal_points = []
    vertical_points = []
    tollerance = 0
    if abs(centroid_y - boundary_coordinates[1]) <= tollerance:
        # horizontal_points.append(point)
        return True
    if abs(centroid_x - boundary_coordinates[0]) <= tollerance:
        return True

def slope_(x1, y1, x2, y2):
    if x1 != x2:
        s = (y2 - y1) / (x2 - x1)
    else:
        s = 0
    return s

def line_equation(coord1, coord2, step):
    points = []
    x1, y1 = coord1
    x2, y2 = coord2
    slope = slope_(x1, y1, x2, y2)
    intercept = y1 - slope * x1
    max_x = max(x1, x2)
    min_x = min(x1, x2)
    x = min_x
    max_y = max(y1, y2)
    min_y = min(y1, y2)
    y = min_y

    if x1 == x2:
        while y < max_y:
            points.append((x, abs(y)))
            y += step
    elif y1 == y2:
        while x < max_x:
            points.append((x, abs(y)))
            x += step
    else:
        while x < max_x:
            y = slope * x + intercept
            if abs(y) <= max_y:
                points.append((x, abs(y)))
            x += step

    if not points:
        print('debug')
    return points

def on_line(coord, cent1, cent2, tolerance=10):
    x, y = coord
    x1, y1 = cent1
    x2, y2 = cent2

    # Check if the points are coincident
    if (x1, y1) == (x2, y2) == (x, y):
        return True

    if x1 == x2 or (x1+1)==x2 or (x2+1)==x1:
        return abs(x - x1) <= tolerance

    # Calculate the slope and y-intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1

    # Calculate the expected y-coordinate on the line
    expected_y = slope * x + y_intercept


    return abs(expected_y - y) <= tolerance

def get_line(x1, y1, x2, y2):
    """Bresenham's line algorithm to get all points between two coordinates"""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        points.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    points.append((x2, y2))
    return points

def check_points(image, points):
    """Check if all points in the image are in free space"""
    itter = 0
    counter = 0
    for x, y in points:
        # cv2.circle(map, (int(x), int(y)), 3, (127, 127, 0), -1)
        value = image[y, x]
        if value>0:
            value = 255
        else:
            value = 0
        if itter==0:
            itter+=1
            continue
        # print(value)
        if value ==255 and itter>0:
            counter+=1
            if counter>2:
                return False
            else:
                continue
    return True


# def relevant_points_extractor(map, points_collection_1, points_collection_2, boundary_coordinates_1,
#                               boundary_coordinates_2):
#     relevant_points_1 = []
#     relevant_points_2 = []
#
#     for point in points_collection_1:
#         for point_2 in points_collection_2:
#             candidate_point_1 = [coord for coord in boundary_coordinates_1 if on_line(coord, point_2, point)]
#             candidate_point_2 = [coord for coord in boundary_coordinates_2 if on_line(coord, point, point_2)]
#             if candidate_point_1 and candidate_point_2:
#                 distance_matrix = np.zeros((len(candidate_point_1), len(candidate_point_2)))
#                 for i, p1 in enumerate(candidate_point_1):
#                     for j, p2 in enumerate(candidate_point_2):
#                         p1=np.array(p1)
#                         p2=np.array(p2)
#                         distance_matrix[i][j] = np.linalg.norm(p2 - p1)
#
#                 # min_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
#                 min_index = np.unravel_index(np.argsort(distance_matrix, axis=None)[:5], distance_matrix.shape)
#                 for index_pair in zip(*min_index):
#                     relevant_points_1.append(candidate_point_1[index_pair[0]])
#                     relevant_points_2.append(candidate_point_2[index_pair[1]])
#                 # relevant_points_1.append(candidate_point_1[min_index[0]])
#                 # relevant_points_2.append(candidate_point_2[min_index[1]])
#
#     for point_1, point_2 in zip(relevant_points_1, relevant_points_2):
#         cv2.circle(map, (int(point_1[0]), int(point_1[1])), 8, (127, 127, 0), -1)
#         cv2.circle(map, (int(point_2[0]), int(point_2[1])), 8, (127, 127, 0), -1)
#
#     # cv2.imshow('test', map)
#     # cv2.waitKey(0)
#
#     return relevant_points_1, relevant_points_2
# def relevant_points_extractor(map, points_collection_1, points_collection_2, boundary_coordinates_1,
#                               boundary_coordinates_2):
#     relevant_points_1 = []
#     relevant_points_2 = []
#
#     for point in points_collection_1:
#         for point_2 in points_collection_2:
#             candidate_point_1 = [coord for coord in boundary_coordinates_1 if on_line(coord, point_2, point)]
#             candidate_point_2 = [coord for coord in boundary_coordinates_2 if on_line(coord, point, point_2)]
#             if candidate_point_1 and candidate_point_2:
#                 distances = []
#                 for p1 in candidate_point_1:
#                     for p2 in candidate_point_2:
#                         distances.append((p1, p2, np.linalg.norm(np.array(p1) - np.array(p2))))
#
#                 distances.sort(key=lambda x: x[2])  # Sort based on distance
#                 for p1, p2, _ in distances:
#                     relevant_points_1.append(p1)
#                     relevant_points_2.append(p2)
#                 break  # Exit inner loop after finding relevant points
#             if relevant_points_1 and relevant_points_2:
#                 break  # Exit outer loop after finding relevant points
#
#     return relevant_points_1, relevant_points_2


from sklearn.neighbors import NearestNeighbors
import numpy as np

from sklearn.neighbors import BallTree

from sklearn.neighbors import BallTree










from collections import defaultdict
def find_nearest_points_on_lines(map,obj1_points, obj2_points):
    def find_equation(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:
            slope = float('inf')
            intercept = x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        return slope, intercept

    def check_point_on_line(slope, intercept, point):
        x, y = point
        return y == slope * x + intercept

    lines = defaultdict(list)

    for p1 in obj1_points:
        for p2 in obj2_points:
            slope, intercept = find_equation(p1, p2)
            current_line_points = [p1, p2]

            for p in obj1_points + obj2_points:
                if p not in current_line_points and check_point_on_line(slope, intercept, p):
                    current_line_points.append(p)

            lines[(slope, intercept)].extend(current_line_points)

    clustered_points = []

    for line_points in lines.values():
        groups = []
        for point in line_points:
            added = False
            for group in groups:
                if any(check_point_on_line(*find_equation(group_point, point), point) for group_point in group):
                    group.append(point)
                    added = True
                    break
            if not added:
                groups.append([point])
        clustered_points.extend(groups)

    return clustered_points

def next_point_on_line(point1, point2):
    def equation_of_line(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        # Calculate slope
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 0  # Vertical line, assign slope as 0

        # Calculate y-intercept
        c = y1 - m * x1

        return m, c

    m, c = equation_of_line(point1, point2)
    x, y = point1
    next_x = x + 1  # Increment x-coordinate to find the next point
    next_y = m * next_x + c  # Calculate the corresponding y-coordinate

    return next_x, next_y
def line_of_sight_checker(thresh,point1,point2):
    def value_mapper(value):
        if value>0:
            return 255
        else:
            return 0
    points = line_equation(point1,point2,1)
    # print(points)
    non_zero = False
    count = 0
    for point in points:
        x, y = point
        metric_value = thresh[int(y)][int(x)]
        # print(metric_value)
        metric_value = value_mapper(metric_value)
        # print(metric_value)
        if metric_value == 255:
            non_zero=True
            break
    if not non_zero:
        return True
    else:
        return False

def cordinates_maker(contours):
    pixel_values = []
    for full in contours:
        for point in full:
            x = point[0][0]
            y = point[0][1]
            pixel_values.append([x, y])
    return pixel_values

def points_towards_centroid(centroid, other_points):
    centroid_x, centroid_y = centroid

    # Calculate distances between centroid and other points
    distances = {}
    for index, point in enumerate(other_points):
        point_x, point_y = point
        distance = math.sqrt((point_x - centroid_x) ** 2 + (point_y - centroid_y) ** 2)
        distances[index] = distance

    # Find points closer to the centroid
    closer_points_indices = [index for index, distance in distances.items() if distance < max(distances.values())]
    closer_points = [other_points[index] for index in closer_points_indices]

    return closer_points



def useful_objects(threshold_image,output):
    def pair_exists(pair_set, target_pair):
        return (target_pair[0], target_pair[1]) in pair_set or (target_pair[1], target_pair[0]) in pair_set

    def find_nearest_points(b1, b2):
        # Create nearest neighbor models using Ball Tree
        nbrs_b1_to_b2 = BallTree(b2)
        nbrs_b2_to_b1 = BallTree(b1)
        # Find the nearest point in b2 for each point in b1
        distances_b1_to_b2, indices_b1_to_b2 = nbrs_b1_to_b2.query(b1, k=1)
        # Find the nearest point in b1 for each point in b2
        distances_b2_to_b1, indices_b2_to_b1 = nbrs_b2_to_b1.query(b2, k=1)
        # Extract single integer indices
        indices_b1_to_b2 = indices_b1_to_b2.squeeze()
        indices_b2_to_b1 = indices_b2_to_b1.squeeze()
        # Collect unique nearest points from b2 for each point in b1
        unique_indices_b1_to_b2 = set(indices_b1_to_b2)
        nearest_points_b1_to_b2 = [b2[index] for index in unique_indices_b1_to_b2]
        # Collect unique nearest points from b1 for each point in b2
        unique_indices_b2_to_b1 = set(indices_b2_to_b1)
        nearest_points_b2_to_b1 = [b1[index] for index in unique_indices_b2_to_b1]
        return nearest_points_b1_to_b2, nearest_points_b2_to_b1
    useful_object_pair = []
    checked_pair = set()
    (numLabels, labels, stats, centroids) = output

    # Pre-calculate contours and coordinates outside the loop
    all_contours = {}
    all_coordinates = {}
    unique_labels = np.unique(labels)

    for obj_label in unique_labels[unique_labels != 0]:
        obstacle = (labels == obj_label).astype("uint8") * 255
        ret, thresh_obj = cv2.threshold(obstacle, 127, 255, 0)
        contours = cv2.findContours(thresh_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        start_1 = time.time()
        all_contours[obj_label] = contours
        all_coordinates[obj_label] = cordinates_maker(contours)
        contour_execution = time.time()-start_1

    for obj1_label in unique_labels[unique_labels != 0]:
        obstacle_1 = (labels == obj1_label).astype("uint8") * 255
        centroid_obj1 = centroids[obj1_label]

        for obj2_label in unique_labels[(unique_labels != 0) & (unique_labels != obj1_label)]:
            obstacle_2 = (labels == obj2_label).astype("uint8") * 255
            start_2 =time.time()
            pair_checked = pair_exists(checked_pair, (obj1_label, obj2_label))
            pair_execution = time.time()-start_2

            if pair_checked and len(useful_object_pair) >=1:
                continue

            checked_pair.add((obj1_label, obj2_label))
            centroid_obj2 = centroids[obj2_label]

            binary_map_temp = obstacle_1 + obstacle_2
            cent_1 = (int(centroid_obj1[0]), int(centroid_obj1[1]))
            cent_2 = (int(centroid_obj2[0]), int(centroid_obj2[1]))

            coordinates_1 = all_coordinates[obj1_label]
            coordinates_2 = all_coordinates[obj2_label]

            start_3 = time.time()
            points_on_line_object_1 = [coord for coord in coordinates_1 if find_points_on_lines(coord, cent_1)]
            points_on_line_object_2 = [coord for coord in coordinates_2 if find_points_on_lines(coord, cent_2)]
            point_on_line_execution = time.time()-start_3

            start_4 = time.time()
            # relavent_point_1, relavent_point_2 = relevant_points_extractor(binary_map_temp, points_on_line_object_1,
            #                                                                points_on_line_object_2,
            #                                                                coordinates_1, coordinates_2)
            relavent_point_1,relavent_point_2 = find_nearest_points(coordinates_1,coordinates_2)
            relevant_execution = time.time()-start_4
            print(f'points on line: {point_on_line_execution}, relevant point filtering: {relevant_execution}')
            break_ = False
            start_5 = time.time()
            binary_map_temp_draw = cv2.cvtColor(binary_map_temp, cv2.COLOR_GRAY2BGR)
            # map_draw = map.copy()
            for i, point_1 in enumerate(relavent_point_1):
                cv2.circle(map, (int(point_1[0]), int(point_1[1])), 10, (0, 0, 255), -1)
                cv2.circle(binary_map_temp_draw, (int(point_1[0]), int(point_1[1])), 10, (255, 255, 0), -1)
            for j, point_2 in enumerate(relavent_point_2):
                cv2.circle(map, (int(point_2[0]), int(point_2[1])), 10, (0, 0, 255), -1)
                cv2.circle(binary_map_temp_draw, (int(point_2[0]), int(point_2[1])), 10, (255, 255, 0), -1)
            cv2.imshow('binary temp',~binary_map_temp_draw)
            cv2.imshow('map',map)
            cv2.waitKey(0)
            # for i, point_1 in enumerate(relavent_point_1):
            #     for j, point_2 in enumerate(relavent_point_2):
            #         x1, y1 = point_1
            #         x2, y2 = point_2
            #         print(f'attempting to get points on line {point_1}{point_2}')
            #         point_on_line = get_line(x1, y1, x2, y2)
            #         print('done')
            #         print('getting status')
            #         status = check_points(threshold_image, point_on_line)
            #         print('done')
            #         if status:
            #             break_ = True
            #             useful_object_pair.append((obj1_label, obj2_label))
            #             break
            #     if break_:
            #         break
            relevant_freespace_execution = time.time()-start_5
            print(f'status checking: {relevant_freespace_execution}')
    return useful_object_pair

# name = 'major_fail_case'
name = 'usefulpair_case'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
# map_path = str(r'C:\Users\Asus\Desktop\results/' + name + '.jpg')
map = cv2.imread(map_path)
output,theshold_image = detect_and_label_0bstacles(map)
(numLabels, labels, stats, centroids) = output
useful_object_pair = []
start = time.time()
pair = useful_objects(threshold_image=theshold_image,output=output)
execution_time = time.time()-start
print(execution_time)
print(pair)
# cv2.imshow('binary temp',map)
# cv2.waitKey(0)

