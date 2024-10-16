import cv2
import numpy as np
import time
from sklearn.neighbors import BallTree
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import constant

CONSTANTS = constant.constants()

def cordinates_maker(contours):
    pixel_values = []
    for full in contours:
        for point in full:
            x = point[0][0]
            y = point[0][1]
            pixel_values.append([x, y])
    return pixel_values
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

def on_line(coord, cent1, cent2, tolerance=4):
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


def relevant_points_extractor(map, points_collection_1, points_collection_2, boundary_coordinates_1,
                              boundary_coordinates_2):
    relevant_points_1 = []
    relevant_points_2 = []

    for point in points_collection_1:
        for point_2 in points_collection_2:
            candidate_point_1 = [coord for coord in boundary_coordinates_1 if on_line(coord, point_2, point)]
            candidate_point_2 = [coord for coord in boundary_coordinates_2 if on_line(coord, point, point_2)]
            if candidate_point_1 and candidate_point_2:
                distance_matrix = np.zeros((len(candidate_point_1), len(candidate_point_2)))
                for i, p1 in enumerate(candidate_point_1):
                    for j, p2 in enumerate(candidate_point_2):
                        p1=np.array(p1)
                        p2=np.array(p2)
                        distance_matrix[i][j] = np.linalg.norm(p2 - p1)

                # min_index = np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)
                min_index = np.unravel_index(np.argsort(distance_matrix, axis=None)[:5], distance_matrix.shape)
                for index_pair in zip(*min_index):
                    relevant_points_1.append(candidate_point_1[index_pair[0]])
                    relevant_points_2.append(candidate_point_2[index_pair[1]])
                # relevant_points_1.append(candidate_point_1[min_index[0]])
                # relevant_points_2.append(candidate_point_2[min_index[1]])

    for point_1, point_2 in zip(relevant_points_1, relevant_points_2):
        cv2.circle(map, (int(point_1[0]), int(point_1[1])), 8, (127, 127, 0), -1)
        cv2.circle(map, (int(point_2[0]), int(point_2[1])), 8, (127, 127, 0), -1)

    # cv2.imshow('test', map)
    # cv2.waitKey(0)

    return relevant_points_1, relevant_points_2


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
            count+=1
    if count<2:
        return True
    else:
        return False

def detect_and_label_0bstacles(map):
    output = cv2.connectedComponentsWithStats(map, cv2.CV_32S)
    return output, map


# def useful_objects(threshold_image,output):
#     def pair_exists(pair_set, target_pair):
#         return (target_pair[0], target_pair[1]) in pair_set or (target_pair[1], target_pair[0]) in pair_set
#
#     def find_nearest_points(b1, b2):
#         # Create nearest neighbor models using Ball Tree
#         nbrs_b1_to_b2 = BallTree(b2)
#         nbrs_b2_to_b1 = BallTree(b1)
#         # Find the nearest point in b2 for each point in b1
#         distances_b1_to_b2, indices_b1_to_b2 = nbrs_b1_to_b2.query(b1, k=1)
#         # Find the nearest point in b1 for each point in b2
#         distances_b2_to_b1, indices_b2_to_b1 = nbrs_b2_to_b1.query(b2, k=1)
#         # Extract single integer indices
#         indices_b1_to_b2 = indices_b1_to_b2.squeeze()
#         indices_b2_to_b1 = indices_b2_to_b1.squeeze()
#         # Collect unique nearest points from b2 for each point in b1
#         unique_indices_b1_to_b2 = set(indices_b1_to_b2)
#         nearest_points_b1_to_b2 = [b2[index] for index in unique_indices_b1_to_b2]
#         # Collect unique nearest points from b1 for each point in b2
#         unique_indices_b2_to_b1 = set(indices_b2_to_b1)
#         nearest_points_b2_to_b1 = [b1[index] for index in unique_indices_b2_to_b1]
#         return nearest_points_b1_to_b2, nearest_points_b2_to_b1
#     useful_object_pair = []
#     checked_pair = set()
#     (numLabels, labels, stats, centroids) = output
#
#     # Pre-calculate contours and coordinates outside the loop
#     all_contours = {}
#     all_coordinates = {}
#     unique_labels = np.unique(labels)
#
#     for obj_label in unique_labels[unique_labels != 0]:
#         obstacle = (labels == obj_label).astype("uint8") * 255
#         ret, thresh_obj = cv2.threshold(obstacle, 127, 255, 0)
#         contours = cv2.findContours(thresh_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
#         start_1 = time.time()
#         all_contours[obj_label] = contours
#         all_coordinates[obj_label] = cordinates_maker(contours)
#         contour_execution = time.time()-start_1
#
#     for obj1_label in unique_labels[unique_labels != 0]:
#         obstacle_1 = (labels == obj1_label).astype("uint8") * 255
#         centroid_obj1 = centroids[obj1_label]
#
#         for obj2_label in unique_labels[(unique_labels != 0) & (unique_labels != obj1_label)]:
#             obstacle_2 = (labels == obj2_label).astype("uint8") * 255
#             start_2 =time.time()
#             pair_checked = pair_exists(checked_pair, (obj1_label, obj2_label))
#             pair_execution = time.time()-start_2
#
#             if pair_checked and len(useful_object_pair) >= 1:
#                 continue
#
#             checked_pair.add((obj1_label, obj2_label))
#             centroid_obj2 = centroids[obj2_label]
#
#             binary_map_temp = obstacle_1 + obstacle_2
#             cent_1 = (int(centroid_obj1[0]), int(centroid_obj1[1]))
#             cent_2 = (int(centroid_obj2[0]), int(centroid_obj2[1]))
#
#             coordinates_1 = all_coordinates[obj1_label]
#             coordinates_2 = all_coordinates[obj2_label]
#
#             start_3 = time.time()
#             points_on_line_object_1 = [coord for coord in coordinates_1 if find_points_on_lines(coord, cent_1)]
#             points_on_line_object_2 = [coord for coord in coordinates_2 if find_points_on_lines(coord, cent_2)]
#             point_on_line_execution = time.time()-start_3
#
#             start_4 = time.time()
#             relavent_point_1,relavent_point_2 = find_nearest_points(coordinates_1,coordinates_2)
#             relevant_execution = time.time()-start_4
#             print(f'points on line: {point_on_line_execution}, relevant point filtering: {relevant_execution}')
#             break_ = False
#             start_5 = time.time()
#             for i, point_1 in enumerate(relavent_point_1):
#                 for j, point_2 in enumerate(relavent_point_2):
#                     x1, y1 = point_1
#                     x2, y2 = point_2
#                     print(f'attempting to get points on line {point_1}{point_2}')
#                     point_on_line = get_line(x1, y1, x2, y2)
#                     print('done')
#                     print('getting status')
#                     status = check_points(threshold_image, point_on_line)
#                     print('done')
#                     if status:
#                         break_ = True
#                         useful_object_pair.append((obj1_label, obj2_label))
#                         break
#                 if break_:
#                     break
#             relevant_freespace_execution = time.time()-start_5
#             print(f'status checking: {relevant_freespace_execution}')
#     return useful_object_pair


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


def useful_boundary_points_identifier(thresh,boundary_1,boundary_2,robot_width,visualization):
    def value_mapper(value):
        if value>0:
            return 255
        else:
            return 0
    def slope_(x1, y1, x2, y2):
        if x1!=x2:
            s = (y2 - y1) / (x2 - x1)
        else:
            s=0
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
            while (y < max_y):
                points.append((x, abs(y)))
                y += step
        elif y1 == y2:
            while (x < max_x):
                points.append((x, abs(y)))
                x += step
        else:
            while (x < max_x):
                y = slope * x + intercept
                if abs(y) <= max_y:
                    points.append((x, abs(y)))
                x += step
        if points == []:
            print('debug')
        return points
    useful_point_pairs = []
    outer_itterator = 0
    inner_itterator = 0
    length_coords1 = len(boundary_1)
    length_coords2 = len(boundary_2)
    if length_coords1 > length_coords2:
        outer_limit = length_coords1
        inner_limit = length_coords2
        bigger_boundary = boundary_1
        smaller_boundary = boundary_2
    elif length_coords2 > length_coords1:
        outer_limit = length_coords2
        inner_limit = length_coords1
        bigger_boundary = boundary_2
        smaller_boundary = boundary_1
    else:
        outer_limit = length_coords1
        inner_limit = length_coords2
        bigger_boundary = boundary_1
        smaller_boundary = boundary_2
    stop_code =False
    while outer_itterator <outer_limit:
        inner_itterator=0
        point1 = bigger_boundary[outer_itterator]
        while inner_itterator < inner_limit:
            point2 = smaller_boundary[inner_itterator]
            # points_on_line = line_equation(point1,point2,2)
            x1,y1 = point1
            x2,y2 = point2
            points_on_line = get_line(x1,y1,x2,y2)
            # points_on_line = line_equation(point1,point2,2)
            non_zero = False
            status = check_points(thresh,points_on_line)
            if status:
                point1 = np.array(point1)
                point2 = np.array(point2)
                passage_width = np.linalg.norm(point2 - point1)
                if robot_width > passage_width:
                    cv2.line(visualization, point1, point2, (0, 255, 255), thickness=5, lineType=8)
                    cv2.line(thresh, point1, point2, (255, 255, 255), thickness=5, lineType=8)
                    stop_code = True
                    break
            inner_itterator+=10
        if stop_code:
            break
        else:
            outer_itterator+=10
    return thresh,visualization

def check_passages_by_near_pairs(robot_width, thresh, near_pair, labels,draw_map):
    draw_result = draw_map.copy()
    draw_result = ~draw_result
    def cordinates_maker(contours):
        pixel_values = []
        for full in contours:
            for point in full:
                x = point[0][0]
                y = point[0][1]
                pixel_values.append([x, y])
        return pixel_values

    iterator_pair = 0
    while iterator_pair < len(near_pair):
        pair = near_pair[iterator_pair]
        obj1, obj2 = pair
        obstacle_1 = (labels == obj1).astype("uint8") * 255
        obstacle_2 = (labels == obj2).astype("uint8") * 255
        binary_map_temp = obstacle_1 + obstacle_2
        # cv2.imshow('mixed map', binary_map_temp)
        # cv2.waitKey(0)
        ret1, thresh_obj1 = cv2.threshold(obstacle_1, 127, 255, 0)
        contours_obj1 = cv2.findContours(thresh_obj1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_1 = cordinates_maker(contours_obj1)

        ret1, thresh_obj2 = cv2.threshold(obstacle_2, 127, 255, 0)
        contours_obj2 = cv2.findContours(thresh_obj2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_2 = cordinates_maker(contours_obj2)
        start = time.time()
        thresh,processed_visualization = useful_boundary_points_identifier(thresh, coordinates_1, coordinates_2, robot_width, draw_result)
        execution_useful = time.time() - start
        # print(f'the useful points execution time is {execution_useful}')

        iterator_useful = 0
        thresh_cpy = thresh.copy()

        iterator_pair += 1
    return thresh, processed_visualization
def occupancy_grid_handler(map):
    # Placeholder functionality
    # Simulate processing the map as an occupancy grid
    print("Processing as Occupancy Grid")
    _, binary_map = cv2.threshold(map, 220, 255, cv2.THRESH_BINARY)
    binary_map = cv2.cvtColor(binary_map, cv2.COLOR_BGR2GRAY)
    return ~binary_map

def _2D_binary_map_handler(map):
    map = ~map
    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    return thresh

def dimension_integrator(map,robot_width):
    start_1 = time.time()
    output, thresh = detect_and_label_0bstacles(map)
    (numLabels, labels, stats, centroids) = output
    near_pair = useful_objects(thresh, output)
    exe_1 = time.time()-start_1
    print(f'pairs identified')
    start_2 = time.time()
    processed_result,draw_result = check_passages_by_near_pairs(robot_width, thresh, near_pair, labels,map)
    processed_binary_map = ~processed_result
    exe_2 = time.time() - start_2

    print(f'execution time for near pair and segmntation is {exe_1} time for rest is {exe_2}')
    print(f'the number of objects in the map: {(numLabels-1)}')
    return processed_binary_map,draw_result

def upload_map(map_type):
    map_path = filedialog.askopenfilename(title="Select a Map",
                                          filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.jfif")])
    if map_path:
        # CONSTANTS = constant.constants()
        print(f'the value which is of robot width is in pixels: {CONSTANTS.robot_width}')
        query_map = cv2.imread(map_path)
        if map_type == "Occupancy Grid":
            map_input = occupancy_grid_handler(query_map)
        else:
            map_input = _2D_binary_map_handler(query_map)

        cv2.imshow('test', map_input)
        cv2.waitKey(0)
        processed_binary_map, result_visualization = dimension_integrator(map_input, CONSTANTS.robot_width)
        processed_binary_map = cv2.resize(processed_binary_map, (960, 720))
        result_visualization = cv2.resize(result_visualization, (960, 720))
        cv2.imshow('input', map_input)
        cv2.imshow('final result', processed_binary_map)
        cv2.imshow('final result visualization', result_visualization)
        cv2.imwrite(str(CONSTANTS.path_to_processed_BM + "_procesed.png"), processed_binary_map)
        cv2.imwrite(str(CONSTANTS.path_to_processed_BM + "_visualization.png"), result_visualization)
        cv2.waitKey(0)
        # return processed_binary_map, result_visualization


def process_video():
    # Simulate capturing video and allowing user to draw bounding box
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            r = cv2.selectROI("Select Robot Size", frame)
            CONSTANTS.robot_width = r[2]  # Robot width in pixels
            print(f"Robot width in pixels: {CONSTANTS.robot_width}")
        cap.release()
    cv2.destroyAllWindows()


# def calculate_robot_width_with_reference(reference_width):
#     # Function to capture or upload a reference image and calculate pixels per meter
#     def capture_reference_image():
#         cap = cv2.VideoCapture(0)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 r = cv2.selectROI("Select Reference Object", frame)
#                 ref_width_pixels = r[2]  # Reference object width in pixels
#                 pixels_per_meter = ref_width_pixels / reference_width
#                 print(f"Pixels per meter: {pixels_per_meter}")
#                 # Now calculate robot pixel width based on pixels_per_meter
#                 cap.release()
#         cv2.destroyAllWindows()
#
#     def upload_reference_image():
#         file_path = filedialog.askopenfilename(title="Select Reference Image")
#         if file_path:
#             frame = cv2.imread(file_path)
#             r = cv2.selectROI("Select Reference Object", frame)
#             ref_width_pixels = r[2]  # Reference object width in pixels
#             pixels_per_meter = ref_width_pixels / reference_width
#             print(f"Pixels per meter: {pixels_per_meter}")
#             # Now calculate robot pixel width based on pixels_per_meter
#         cv2.destroyAllWindows()
#
#     # Capture or upload reference image based on user choice
#     return capture_reference_image, upload_reference_image


# def upload_map(map_type):
#     # Open a file dialog to select a map file
#     map_file_path = filedialog.askopenfilename(title="Select a Map",
#                                                filetypes=(("Image Files", "*.bmp;*.png;*.jpg"), ("All Files", "*.*")))
#     if map_file_path:
#         print(f"Selected map: {map_file_path}")

# def calculate_robot_width_with_reference(reference_width, capture_option_value):
#     # Function to capture a reference image using a webcam
#     def capture_reference_image():
#         cap = cv2.VideoCapture(0)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 r = cv2.selectROI("Select Reference Object", frame)
#                 ref_width_pixels = r[2]  # Reference object width in pixels
#                 pixels_per_meter = ref_width_pixels / reference_width
#                 print(f"Pixels per meter: {pixels_per_meter}")
#                 cap.release()
#         cv2.destroyAllWindows()
#
#     # Function to upload a reference image from a file
#     def upload_reference_image():
#         file_path = filedialog.askopenfilename(title="Select Reference Image")
#         if file_path:
#             frame = cv2.imread(file_path)
#             r = cv2.selectROI("Select Reference Object", frame)
#             ref_width_pixels = r[2]  # Reference object width in pixels
#             pixels_per_meter = ref_width_pixels / reference_width
#             print(f"Pixels per meter: {pixels_per_meter}")
#         cv2.destroyAllWindows()
#
#     # Determine whether to capture or upload the image based on the user's selection
#     if capture_option_value == "Capture":
#         capture_reference_image()
#     elif capture_option_value == "Upload":
#         upload_reference_image()
#
# def main():
#     root = tk.Tk()
#     root.title("Map Selection UI")
#     root.geometry("400x300")
#     root.resizable(True, True)  # Make the window resizable
#
#     # Set the dark mode theme
#     style = ttk.Style(root)
#     root.tk_setPalette(background="#2e2e2e", foreground="#ffffff", activeBackground="#4e4e4e", activeForeground="#ffffff")
#
#     # Configure ttk style for the dark theme
#     style.configure("TNotebook", background="#2e2e2e", borderwidth=0)
#     style.configure("TFrame", background="#2e2e2e")
#     style.configure("TLabel", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#     style.configure("TRadiobutton", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#     style.configure("BlackText.TButton", background="#ffffff", foreground="#000000", font=("Arial", 12))
#
#     # Label for map type selection
#     map_type_label = ttk.Label(root, text="Select Map Type:")
#     map_type_label.pack(pady=10)
#
#     # Radio buttons for selecting map type
#     map_type = tk.StringVar(value="2D Binary Map")
#     ttk.Radiobutton(root, text="2D Binary Map", variable=map_type, value="2D Binary Map", style="TRadiobutton").pack(anchor="center", padx=20)
#     ttk.Radiobutton(root, text="Occupancy Grid", variable=map_type, value="Occupancy Grid", style="TRadiobutton").pack(anchor="center", padx=20)
#
#     # Create the notebook (tabs)
#     notebook = ttk.Notebook(root, style="TNotebook")
#     notebook.pack(expand=True, fill='both')
#
#     # Tab for getting robot width in pixels
#     get_width_tab = ttk.Frame(notebook)
#     notebook.add(get_width_tab, text="Get Robot Width")
#
#     # Button to get robot width in pixels
#     get_width_button = ttk.Button(get_width_tab, text="Get Robot Width in Pixels", command=lambda: process_video(), style="BlackText.TButton")
#     get_width_button.pack(pady=10)
#
#     # Tab for calculating robot width in meters
#     calculate_width_tab = ttk.Frame(notebook)
#     notebook.add(calculate_width_tab, text="Calculate Robot Width")
#
#     # Label for robot width input
#     width_label = ttk.Label(calculate_width_tab, text="Enter Real Robot Width (in meters):")
#     width_label.pack(pady=10)
#
#     # Entry for user to input robot width
#     global robot_width_entry
#     robot_width_entry = ttk.Entry(calculate_width_tab)
#     robot_width_entry.pack(pady=5)
#
#     # Labels and entries for reference object
#     ref_object_label = ttk.Label(calculate_width_tab, text="Reference Object Width (in meters):")
#     ref_object_label.pack(pady=10)
#
#     ref_object_entry = ttk.Entry(calculate_width_tab)
#     ref_object_entry.pack(pady=5)
#
#     # Radio buttons for image capture options
#     capture_option = tk.StringVar(value="Capture")
#     ttk.Radiobutton(calculate_width_tab, text="Capture Reference Image", variable=capture_option, value="Capture").pack(anchor="center", padx=20)
#     ttk.Radiobutton(calculate_width_tab, text="Upload Reference Image", variable=capture_option, value="Upload").pack(anchor="center", padx=20)
#
#     # Button to process the reference image
#     process_ref_button = ttk.Button(calculate_width_tab, text="Process Reference Image", command=lambda: calculate_robot_width_with_reference(capture_option.get()), style="BlackText.TButton")
#     process_ref_button.pack(pady=10)
#
#     # Upload button for map
#     upload_button = ttk.Button(root, text="Upload & Process Map", command=lambda: upload_map(map_type.get()), style="BlackText.TButton")
#     upload_button.pack(pady=20)
#
#     # Start the GUI loop
#     root.mainloop()
#
#
# if __name__ == "__main__":
#     main()
#
# def main():
#     root = tk.Tk()
#     root.title("Map Selection UI")
#     root.geometry("400x300")
#
#     # Set the dark mode theme
#     style = ttk.Style(root)
#     root.tk_setPalette(background="#2e2e2e", foreground="#ffffff", activeBackground="#4e4e4e",
#                        activeForeground="#ffffff")
#
#     # Configure ttk style for the dark theme
#     style.configure("TLabel", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#     style.configure("TRadiobutton", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#
#     # Configure buttons to have black text
#     style.configure("BlackText.TButton", background="#4e4e4e", foreground="#000000", font=("Arial", 12))
#     style.map("BlackText.TButton", background=[('active', '#666666')])
#
#     # Label for map type selection
#     label = ttk.Label(root, text="Select Map Type:")
#     label.pack(pady=10)
#
#     # Radio buttons for selecting map type
#     map_type = tk.StringVar(value="2D Binary Map")
#     ttk.Radiobutton(root, text="2D Binary Map", variable=map_type, value="2D Binary Map").pack(anchor="center", padx=20)
#     ttk.Radiobutton(root, text="Occupancy Grid", variable=map_type, value="Occupancy Grid").pack(anchor="center",
#                                                                                                  padx=20)
#
#     # Label for robot width input
#     width_label = ttk.Label(root, text="Enter Real Robot Width (in meters):", foreground="#000000")
#     width_label.pack(pady=10)
#
#     # Entry for user to input robot width
#     global robot_width_entry
#     robot_width_entry = ttk.Entry(root, style="TEntry")
#     robot_width_entry.pack(pady=5)
#
#     # Button to get robot width in pixels
#     get_width_button = ttk.Button(root, text="Get Robot Width in Pixels", style="BlackText.TButton",
#                                   command=lambda: process_video(float(robot_width_entry.get())))
#     get_width_button.pack(pady=10)
#
#     # Upload button with black text
#     upload_button = ttk.Button(root, text="Upload & Process Map", style="BlackText.TButton",
#                                command=lambda: upload_map(map_type.get()))
#     upload_button.pack(pady=20)
#
#     # Start the GUI loop
#     root.mainloop()

def calculate_robot_width_with_reference(reference_width, capture_option_value):
    # Function to capture a reference image using a webcam
    def capture_reference_image():
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                r = cv2.selectROI("Select Reference Object", frame)
                CONSTANTS.robot_width = r[2]  # Reference object width in pixels
                # pixels_per_meter = ref_width_pixels / reference_width
                print(f"Pixels width of robot: {CONSTANTS.robot_width}")
                cap.release()
        cv2.destroyAllWindows()

    # Function to upload a reference image from a file
    def upload_reference_image():
        file_path = filedialog.askopenfilename(title="Select Reference Image")
        if file_path:
            frame = cv2.imread(file_path)
            r = cv2.selectROI("Select Reference Object", frame)
            CONSTANTS.robot_width = r[2]  # Reference object width in pixels
            # pixels_per_meter = ref_width_pixels / reference_width

            print(f"Pixels width of robot: {CONSTANTS.robot_width}")
        cv2.destroyAllWindows()

    # Determine whether to capture or upload the image based on the user's selection
    if capture_option_value == "Capture":
        capture_reference_image()
    elif capture_option_value == "Upload":
        upload_reference_image()

def main():
    root = tk.Tk()
    root.title("Map Selection UI")
    root.geometry("400x300")
    root.resizable(True, True)  # Make the window resizable

    # Set the dark mode theme
    style = ttk.Style(root)
    root.tk_setPalette(background="#2e2e2e", foreground="#ffffff", activeBackground="#4e4e4e", activeForeground="#ffffff")

    # Configure ttk style for the dark theme
    style.configure("TNotebook", background="#2e2e2e", borderwidth=0)
    style.configure("TFrame", background="#2e2e2e")
    style.configure("TLabel", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
    style.configure("TRadiobutton", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
    style.configure("BlackText.TButton", background="#ffffff", foreground="#000000", font=("Arial", 12))

    # Label for map type selection
    map_type_label = ttk.Label(root, text="Select Map Type:")
    map_type_label.pack(pady=10)

    # Radio buttons for selecting map type
    map_type = tk.StringVar(value="2D Binary Map")
    ttk.Radiobutton(root, text="2D Binary Map", variable=map_type, value="2D Binary Map", style="TRadiobutton").pack(anchor="center", padx=20)
    ttk.Radiobutton(root, text="Occupancy Grid", variable=map_type, value="Occupancy Grid", style="TRadiobutton").pack(anchor="center", padx=20)

    # Create the notebook (tabs)
    notebook = ttk.Notebook(root, style="TNotebook")
    notebook.pack(expand=True, fill='both')

    # Tab for getting robot width in pixels
    get_width_tab = ttk.Frame(notebook)
    notebook.add(get_width_tab, text="Get Robot Width")

    # Button to get robot width in pixels
    get_width_button = ttk.Button(get_width_tab, text="Get Robot Width in Pixels", command=lambda: process_video(), style="BlackText.TButton")
    get_width_button.pack(pady=10)

    # Tab for calculating robot width in meters
    calculate_width_tab = ttk.Frame(notebook)
    notebook.add(calculate_width_tab, text="Calculate Robot Width")

    # Label for robot width input
    width_label = ttk.Label(calculate_width_tab, text="Enter Real Robot Width (in meters):")
    width_label.pack(pady=10)

    # Entry for user to input robot width
    global robot_width_entry
    robot_width_entry = ttk.Entry(calculate_width_tab)
    robot_width_entry.pack(pady=5)

    # Labels and entries for reference object
    ref_object_label = ttk.Label(calculate_width_tab, text="Reference Object Width (in meters):")
    ref_object_label.pack(pady=10)

    ref_object_entry = ttk.Entry(calculate_width_tab)
    ref_object_entry.pack(pady=5)

    # Radio buttons for image capture options
    capture_option = tk.StringVar(value="Capture")
    ttk.Radiobutton(calculate_width_tab, text="Capture Reference Image", variable=capture_option, value="Capture").pack(anchor="center", padx=20)
    ttk.Radiobutton(calculate_width_tab, text="Upload Reference Image", variable=capture_option, value="Upload").pack(anchor="center", padx=20)

    # Button to process the reference image
    def process_reference_image():
        try:
            reference_width = float(ref_object_entry.get())  # Get the reference width entered by the user
            capture_option_value = capture_option.get()  # Get the selected option (Capture or Upload)
            calculate_robot_width_with_reference(reference_width, capture_option_value)  # Pass both the reference width and capture option to the function
        except ValueError:
            print("Please enter a valid number for the reference width")

    process_ref_button = ttk.Button(calculate_width_tab, text="Process Reference Image", command=process_reference_image, style="BlackText.TButton")
    process_ref_button.pack(pady=10)

    # Upload button for map
    upload_button = ttk.Button(root, text="Upload & Process Map", command=lambda: upload_map(map_type.get()), style="BlackText.TButton")
    upload_button.pack(pady=20)

    # Start the GUI loop
    root.mainloop()

if __name__ == "__main__":
    main()


# def main():
#     root = tk.Tk()
#     root.title("Map Selection UI")
#     root.geometry("400x300")
#
#     # Set the dark mode theme
#     style = ttk.Style(root)
#     root.tk_setPalette(background="#2e2e2e", foreground="#ffffff", activeBackground="#4e4e4e",
#                        activeForeground="#ffffff")
#
#     # Configure ttk style for the dark theme
#     style.configure("TLabel", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#     style.configure("TRadiobutton", background="#2e2e2e", foreground="#ffffff", font=("Arial", 12))
#     style.configure("TButton", background="#4e4e4e", foreground="#ffffff", font=("Arial", 12))
#     style.map("TButton", background=[('active', '#666666')])
#
#     # Label for selection
#     label = ttk.Label(root, text="Select Map Type:")
#     label.pack(pady=10)
#
#     # Radio buttons for selecting map type
#     map_type = tk.StringVar(value="2D Binary Map")
#     ttk.Radiobutton(root, text="2D Binary Map", variable=map_type, value="2D Binary Map").pack(anchor="center", padx=20)
#     ttk.Radiobutton(root, text="Occupancy Grid", variable=map_type, value="Occupancy Grid").pack(anchor="center", padx=20)
#
#     style.configure("TButton", background="#4e4e4e", foreground="#000000", font=("Arial", 12))
#     style.map("Upload.TButton", background=[('active', '#666666')])
#     # Upload button
#     upload_button = ttk.Button(root, text="Upload & Process Map", command=lambda: upload_map(map_type.get()))
#     upload_button.pack(pady=20)
#
#     # Start the GUI loop
#     root.mainloop()
# `


