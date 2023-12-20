import cv2
import numpy as np
def detect_and_label_0bstacles(map):
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh

def slope_(x1, y1, x2, y2):
    if x1!=x2:
        s = (y2 - y1) / (x2 - x1)
    else:
        s=0
    return s
def slope_(x1, y1, x2, y2):
    # Calculate slope between two points
    if x2 - x1 == 0:
        return None  # Avoid division by zero
    return (y2 - y1) / (x2 - x1)

def on_line(coord, cent1, cent2, tolerance=1):
    x, y = coord
    x1, y1 = cent1
    x2, y2 = cent2

    slope = slope_(x1, y1, x2, y2)
    if slope is not None:
        expected_y = slope * (x - x1) + y1
        if abs(expected_y - y) <= tolerance:
            return True
    return False

def on_line(coord, cent1, cent2, tolerance=4):
    x, y = coord
    x1, y1 = cent1
    x2, y2 = cent2

    # Check if the points are coincident
    if (x1, y1) == (x2, y2) == (x, y):
        return True

    # Check if the line is vertical (to avoid division by zero)
    if x1 == x2 or (x1+1)==x2 or (x2+1)==x1:
        return abs(x - x1) <= tolerance

    # Calculate the slope and y-intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1

    # Calculate the expected y-coordinate on the line
    expected_y = slope * x + y_intercept

    # Check if the y-coordinate of the given point is close to the expected y-coordinate
    return abs(expected_y - y) <= tolerance



def cordinates_maker(contours):
    pixel_values = []
    for full in contours:
        for point in full:
            x = point[0][0]
            y = point[0][1]
            pixel_values.append([x, y])
    return pixel_values




name = 'major_fail_case'
# name = 'environment_2'
# map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.png')
map_path = str(r'C:\Users\Asus\Desktop\results/' + name + '.jpg')
map = cv2.imread(map_path)
output,theshold_image = detect_and_label_0bstacles(map)
(numLabels, labels, stats, centroids) = output
for obj1_label in np.unique(labels):
    if obj1_label == 0:  # Skip the background label (if labeled as 0)
        continue

    # Create a binary mask for obj1_label
    obstacle_1 = (labels == obj1_label).astype("uint8") * 255
    centroid_obj1 = centroids[obj1_label]


    # Loop through other labels to compare distances
    for obj2_label in np.unique(labels):
        if obj2_label == 0 or obj2_label == obj1_label:
            continue

        # Create a binary mask for obj2_label
        obstacle_2 = (labels == obj2_label).astype("uint8") * 255
        centroid_obj2 = centroids[obj2_label]

        ret1, thresh_obj1 = cv2.threshold(obstacle_1, 127, 255, 0)
        contours_obj1 = cv2.findContours(thresh_obj1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_1 = cordinates_maker(contours_obj1)

        ret1, thresh_obj2 = cv2.threshold(obstacle_2, 127, 255, 0)
        contours_obj2 = cv2.findContours(thresh_obj2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_2 = cordinates_maker(contours_obj2)

        # Combine binary masks
        binary_map_temp = obstacle_1 + obstacle_2
        cv2.circle(binary_map_temp, (int(centroid_obj1[0]), int(centroid_obj1[1])), 5, (127, 255, 0), -1)
        cv2.circle(binary_map_temp, (int(centroid_obj2[0]), int(centroid_obj2[1])), 5, (127, 255, 0), -1)
        cent_1 = (int(centroid_obj1[0]), int(centroid_obj1[1]))
        cent_2 = (int(centroid_obj2[0]), int(centroid_obj2[1]))
        cv2.line(binary_map_temp, cent_1, cent_2, (127, 127, 127), thickness=2, lineType=8)
        cv2.line(theshold_image, cent_1, cent_2, (127, 127, 127), thickness=2, lineType=8)
        points_on_line_object_1 = [coord for coord in coordinates_1 if on_line(coord, cent_1, cent_2)]
        points_on_line_object_2 = [coord for coord in coordinates_2 if on_line(coord, cent_1, cent_2)]

        # for point_1 in points_on_line_object_1:
        #     cv2.circle(binary_map_temp, (int(point_1[0]), int(point_1[1])), 5, (127, 0, 127), -1)
        #
        #     # Draw circles on points lying on the line for object 2
        # for point_2 in points_on_line_object_2:
        #     cv2.circle(binary_map_temp, (int(point_2[0]), int(point_2[1])), 5, (127, 0, 127), -1)
        minimum_distance = 0
        ittr=0
        for point_1 in points_on_line_object_1:
            for point_2 in points_on_line_object_2:
                point_1 = np.array(point_1)
                point_2 = np.array(point_2)
                distance = np.linalg.norm(point_2 - point_1)
                if ittr==0:
                    minimum_distance=distance
                    ittr+=1
                if distance<=minimum_distance:
                    minimum_distance=distance
                    minimum_point_1=point_1
                    minimum_point_2=point_2
        print(minimum_point_1,minimum_point_2,minimum_distance)
        cv2.circle(binary_map_temp, (int(minimum_point_1[0]), int(minimum_point_1[1])), 8, (127, 127, 0), -1)
        cv2.circle(binary_map_temp, (int(minimum_point_2[0]), int(minimum_point_2[1])), 8, (127, 127, 0), -1)
        cv2.imshow('binary map temp',binary_map_temp)
        cv2.imshow('binary map',theshold_image)
        cv2.waitKey(0)