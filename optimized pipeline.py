import cv2
import numpy as np
import time
from sklearn.neighbors import BallTree



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

# def relevant_points_extractor(map,points_collection,centroid,boundary_coordinates):
#     candidate_points = []
#     relevant_points = []
#     temp=[]
#     minimum_distance = 0
#     ittr = 0
#     for point in points_collection:
#         candidate_point = [coord for coord in boundary_coordinates if on_line(coord, centroid, point)]
#         candidate_points.append(candidate_point)
#     for candidate in candidate_points:
#         # print(candidate)
#         ittr = 0
#         minimum_distance = 0
#         for point_candidate in candidate:
#             distance = np.linalg.norm(centroid - point_candidate)
#             if ittr==0:
#                 minimum_distance=distance
#                 temp = point_candidate
#                 ittr+=1
#             else:
#                 if distance<=minimum_distance:
#                     selected_point = point_candidate
#                     ittr+=1
#         if ittr == 1:
#             selected_point = temp
#         cv2.circle(map, (int(selected_point[0]), int(selected_point[1])), 8, (127, 127, 0), -1)
#         # print(selected_point)
#         #
#         cv2.imshow('test', map)
#         cv2.waitKey(0)
#         relevant_points.append(selected_point)
#     return relevant_points

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
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh
#
# def identify_near_obstacle_pairs(thresh,numLabels,stats,centroids):
#     dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
#     distances = []
#     for i in range(1, numLabels):  # Start from 1 to exclude the background label 0
#         for j in range(i + 1, numLabels):  # Compare current object with the rest
#             dist = np.linalg.norm(centroids[i] - centroids[j])  # Euclidean distance between centroids
#             distances.append(dist)
#     # Calculate the average object size as a reference for the dynamic fraction
#     average_size = np.mean(stats[1:, cv2.CC_STAT_AREA])  # Exclude background label 0, use area statistic
#     print(average_size)
#
#     # Define a function to determine the dynamic fraction based on the average size
#     def calculate_dynamic_fraction(avg_size):
#         def convert_to_fraction(number):
#             # Convert the number to a string and split it at the decimal point
#             number_str = str(int(number))
#             # Calculate the division factor based on the number of decimal places
#             decimal_places = len(number_str)
#             division_factor = 10 ** decimal_places
#             # Convert the number to the desired fraction
#             fraction = float(f"{number_str}") / division_factor
#             return fraction
#
#         # You can define any function here based on your criteria
#         # Here's a simple example using a linear relationship between average size and fraction
#         fraction = convert_to_fraction(avg_size)
#         print(fraction)
#         # return 0.0001 * avg_size  # Adjust this relationship based on your specific scenario
#         return fraction  # Adjust this relationship based on your specific scenario
#     # Calculate the dynamic fraction using the average size
#     dynamic_fraction = calculate_dynamic_fraction(average_size)
#     print(f'the dynamic fraction is {dynamic_fraction}')
#     # Calculate the dynamic threshold using the dynamic fraction
#     dynamic_threshold = dynamic_fraction * np.mean(distances)  # Use np.mean or np.max based on your preference
#     print(f'the distances are {distances}')
#     print(dynamic_threshold)
#     # Identify objects that are near each other based on the dynamic threshold
#     near_objects = []
#     far_objects = []
#     for i in range(1, numLabels):  # Start from 1 to exclude the background label 0
#         for j in range(i + 1, numLabels):  # Compare current object with the rest
#             dist = np.linalg.norm(centroids[i] - centroids[j])  # Euclidean distance between centroids
#             if dist < dynamic_threshold:
#                 near_objects.append((i, j))
#             else:
#                 far_objects.append((i, j))
#     print(near_objects)
#
#     return near_objects,far_objects

# def near_objects_identifier(map,output):
#     (numLabels, labels, stats, centroids) = output
#     h, w = map.shape
#
#     # mean_area = np.mean(stats[1:, cv2.CC_STAT_AREA])  # Exclude background label
#     #
#     # # Normalize mean area between 0 and 1
#     # normalized_area = (mean_area - np.min(stats[1:, cv2.CC_STAT_AREA])) / (
#     #             np.max(stats[1:, cv2.CC_STAT_AREA]) - np.min(stats[1:, cv2.CC_STAT_AREA]))
#
#     # print(normalized_area)
#     # diagonal points
#     top_diag = (0, 0)
#     bottom_diag = (abs(int((w - 1))), abs(int((h - 1))))
#     diagonal_distance = np.linalg.norm(np.array(top_diag) - np.array(bottom_diag))
#     threshold_distance = diagonal_distance * 0.4
#     # threshold_distance = diagonal_distance * normalized_area
#     # Loop through each unique object label
#     near_objects = []
#     for obj1_label in np.unique(labels):
#         if obj1_label == 0:  # Skip the background label (if labeled as 0)
#             continue
#
#         # Create a binary mask for obj1_label
#         obstacle_1 = (labels == obj1_label).astype("uint8") * 255
#         centroid_obj1 = centroids[obj1_label]
#
#         # Loop through other labels to compare distances
#         for obj2_label in np.unique(labels):
#             if obj2_label == 0 or obj2_label == obj1_label:
#                 continue
#
#             # Create a binary mask for obj2_label
#             obstacle_2 = (labels == obj2_label).astype("uint8") * 255
#             centroid_obj2 = centroids[obj2_label]
#
#             # Combine binary masks
#             binary_map_temp = obstacle_1 + obstacle_2
#             cv2.circle(binary_map_temp, (int(centroid_obj1[0]), int(centroid_obj1[1])), 5, (127, 255, 0), -1)
#             cv2.circle(binary_map_temp, (int(centroid_obj2[0]), int(centroid_obj2[1])), 5, (127, 255, 0), -1)
#
#             # dist_transform = cv2.distanceTransform(binary_map_temp, cv2.DIST_L2, 3)
#             dist_between_objects = np.linalg.norm(centroid_obj1 - centroid_obj2)
#
#             # cv2.imshow('frame', binary_map_temp)
#             # cv2.waitKey(0)
#             # Compare distance with threshold
#             if dist_between_objects < threshold_distance:
#                 print(f"Objects {obj1_label} and {obj2_label} are near")
#                 near_objects.append((obj1_label, obj2_label))
#     return near_objects


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

            if pair_checked and len(useful_object_pair) > 1:
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
                    cv2.line(visualization, point1, point2, (0, 219, 0), thickness=3, lineType=8)
                    cv2.line(thresh, point1, point2, (255, 255, 255), thickness=3, lineType=8)
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
        thresh,processed_visualization = useful_boundary_points_identifier(thresh, coordinates_1, coordinates_2,robot_width,draw_result)
        execution_useful = time.time() - start
        # print(f'the useful points execution time is {execution_useful}')

        iterator_useful = 0
        thresh_cpy = thresh.copy()

        iterator_pair += 1
    # cv2.imshow('test', thresh_cpy)
    # cv2.waitKey(0)
    return thresh,processed_visualization

def dimension_integrator(map):
    start_1 = time.time()
    output, thresh = detect_and_label_0bstacles(map)
    (numLabels, labels, stats, centroids) = output
    near_pair= useful_objects(thresh, output)
    exe_1 = time.time()-start_1
    print(f'pairs identified')
    start_2 = time.time()
    processed_result,draw_result = check_passages_by_near_pairs(11, thresh, near_pair, labels,map)
    # print(f'result generation with useful points')
    processed_binary_map = ~processed_result
    exe_2 = time.time() - start_2

    print(f'execution time for near pair and segmntation is {exe_1} time for rest is {exe_2}')
    print(f'the number of objects in the map: {(numLabels-1)}')
    return processed_binary_map,draw_result
def main():
    # name = 'map_5'
    name = 'Berlin_0_1024'
    # map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
    map_path = str(r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\selected_maps/' + name + '.png')
    map = cv2.imread(map_path)
    map = cv2.resize(map, (250,250))

    start = time.time()
    processed_binary_map,result_visualization = dimension_integrator(map)
    end = time.time() - start
    processed_binary_map = cv2.resize(processed_binary_map, (960,720))
    result_visualization = cv2.resize(result_visualization, (960,720))
    print(end)
    cv2.imshow('input', map)
    cv2.imshow('final result',processed_binary_map)
    cv2.imwrite(str(r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\results_elimination_pipleline/'+name+'_visualization.png'),processed_binary_map)
    cv2.imshow('final result visualization',result_visualization)
    cv2.imwrite(
        str(r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\results_elimination_pipleline/' + name + '_processed.png'),result_visualization)
    cv2.waitKey(0)


if __name__=="__main__":
    main()