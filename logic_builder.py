import cv2
import numpy as np
import time
from sklearn.neighbors import BallTree
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import constant
def cordinates_maker(contours):
    pixel_values = []
    for full in contours:
        for point in full:
            x = point[0][0]
            y = point[0][1]
            pixel_values.append([x, y])
    return pixel_values

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

        # Collect unique nearest points from b2 for each point in b1
        nearest_points_b1_to_b2 = [b2[index] for index in np.unique(indices_b1_to_b2)]
        # Collect unique nearest points from b1 for each point in b2
        nearest_points_b2_to_b1 = [b1[index] for index in np.unique(indices_b2_to_b1)]
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

            if pair_checked and len(useful_object_pair) >= 1:
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
            relavent_point_1,relavent_point_2 = find_nearest_points(coordinates_1,coordinates_2)
            relevant_execution = time.time()-start_4
            print(f'points on line: {point_on_line_execution}, relevant point filtering: {relevant_execution}')
            break_ = False
            start_5 = time.time()
            for i, point_1 in enumerate(relavent_point_1):
                for j, point_2 in enumerate(relavent_point_2):
                    x1, y1 = point_1
                    x2, y2 = point_2
                    print(f'attempting to get points on line {point_1}{point_2}')
                    point_on_line = get_line(x1, y1, x2, y2)
                    print('done')
                    print('getting status')
                    status = check_points(threshold_image, point_on_line)
                    print('done')
                    if status:
                        break_ = True
                        useful_object_pair.append((obj1_label, obj2_label))
                        break
                if break_:
                    break
            relevant_freespace_execution = time.time()-start_5
            print(f'status checking: {relevant_freespace_execution}')
    return useful_object_pair

def occupancy_grid_handler(map):
    # Placeholder functionality
    # Simulate processing the map as an occupancy grid
    print("Processing as Occupancy Grid")
    _, binary_map = cv2.threshold(map, 220, 255, cv2.THRESH_BINARY)
    binary_map = cv2.cvtColor(binary_map, cv2.COLOR_BGR2GRAY)
    cv2.imshow('shoe',~binary_map)
    cv2.waitKey(0)
    return ~binary_map
def detect_and_label_0bstacles(map):
    output = cv2.connectedComponentsWithStats(map, cv2.CV_32S)
    return output, map

path = r"C:\Users\Lenovo\OneDrive\Pictures\test_maps\map_4.jpg"

image = cv2.imread(path)
processed_map = occupancy_grid_handler(image)
output,thresh = detect_and_label_0bstacles(processed_map)

pairs = useful_objects(thresh, output)