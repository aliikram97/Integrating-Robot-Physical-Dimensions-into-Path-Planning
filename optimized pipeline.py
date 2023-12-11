import cv2
import numpy as np
import time


def detect_and_label_0bstacles(map):
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh

def identify_near_obstacle_pairs(thresh,numLabels,stats,centroids):
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    distances = []
    for i in range(1, numLabels):  # Start from 1 to exclude the background label 0
        for j in range(i + 1, numLabels):  # Compare current object with the rest
            dist = np.linalg.norm(centroids[i] - centroids[j])  # Euclidean distance between centroids
            distances.append(dist)
    # Calculate the average object size as a reference for the dynamic fraction
    average_size = np.mean(stats[1:, cv2.CC_STAT_AREA])  # Exclude background label 0, use area statistic
    print(average_size)

    # Define a function to determine the dynamic fraction based on the average size
    def calculate_dynamic_fraction(avg_size):
        def convert_to_fraction(number):
            # Convert the number to a string and split it at the decimal point
            number_str = str(int(number))
            # Calculate the division factor based on the number of decimal places
            decimal_places = len(number_str)
            division_factor = 10 ** decimal_places
            # Convert the number to the desired fraction
            fraction = float(f"{number_str}") / division_factor
            return fraction

        # You can define any function here based on your criteria
        # Here's a simple example using a linear relationship between average size and fraction
        fraction = convert_to_fraction(avg_size)
        print(fraction)
        # return 0.0001 * avg_size  # Adjust this relationship based on your specific scenario
        return fraction  # Adjust this relationship based on your specific scenario
    # Calculate the dynamic fraction using the average size
    dynamic_fraction = calculate_dynamic_fraction(average_size)
    print(f'the dynamic fraction is {dynamic_fraction}')
    # Calculate the dynamic threshold using the dynamic fraction
    dynamic_threshold = dynamic_fraction * np.mean(distances)  # Use np.mean or np.max based on your preference
    print(f'the distances are {distances}')
    print(dynamic_threshold)
    # Identify objects that are near each other based on the dynamic threshold
    near_objects = []
    far_objects = []
    for i in range(1, numLabels):  # Start from 1 to exclude the background label 0
        for j in range(i + 1, numLabels):  # Compare current object with the rest
            dist = np.linalg.norm(centroids[i] - centroids[j])  # Euclidean distance between centroids
            if dist < dynamic_threshold:
                near_objects.append((i, j))
            else:
                far_objects.append((i, j))
    print(near_objects)

    return near_objects,far_objects

def useful_boundary_points_identifier(thresh,boundary_1,boundary_2):
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
    while outer_itterator <outer_limit:
        inner_itterator=0
        point1 = bigger_boundary[outer_itterator]
        while inner_itterator < inner_limit:
            point2 = smaller_boundary[inner_itterator]
            points_on_line = line_equation(point1,point2,1)
            non_zero = False
            for point in points_on_line:
                x,y = point
                x = round(x)
                y=round(y)
                metric_value = thresh[int(y)][int(x)]
                metric_value = value_mapper(metric_value)
                # print(metric_value)
                if metric_value == 255:
                    # print(metric_value)
                    non_zero = True
                    break
            if not non_zero:
                # print(point1,point2)
                useful_point_pairs.append((point1,point2))
            inner_itterator+=3
        outer_itterator+=3
    return useful_point_pairs

def check_passages_by_near_pairs(robot_width, thresh, near_pair, labels):
    def cordinates_maker(contours):
        pixel_values = []
        for full in contours:
            for point in full:
                x = point[0][0]
                y = point[0][1]
                pixel_values.append([x, y])
        return pixel_values

    for pair in near_pair:
        obj1,obj2 = pair
        obstacle_1 = (labels == obj1).astype("uint8") * 255
        obstacle_2 = (labels == obj2).astype("uint8") * 255
        binary_map_temp = obstacle_1+obstacle_2
        # cv2.imshow('mixed map',binary_map_temp)
        # cv2.waitKey(0)
        ret1, thresh_obj1 = cv2.threshold(obstacle_1, 127, 255, 0)
        contours_obj1 = cv2.findContours(thresh_obj1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_1 = cordinates_maker(contours_obj1)

        ret1, thresh_obj2 = cv2.threshold(obstacle_2, 127, 255, 0)
        contours_obj2 = cv2.findContours(thresh_obj2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        coordinates_2 = cordinates_maker(contours_obj2)
        useful_points = useful_boundary_points_identifier(thresh,coordinates_1,coordinates_2)
        itterator = 0
        while itterator < len(useful_points):
            points = useful_points[itterator]
            point1,point2 =points
            point1 = np.array(point1)
            point2 = np.array(point2)
            passage_width = np.linalg.norm(point2 - point1)
            if robot_width>passage_width:
                # cv2.line(thresh, point1, point2, (127, 127, 127), thickness=8, lineType=8)
                cv2.line(thresh, point1, point2, (255, 255, 255), thickness=8, lineType=8)
            itterator+=3
    return thresh

def dimension_integrator(map):
    output, thresh = detect_and_label_0bstacles(map)
    (numLabels, labels, stats, centroids) = output
    near_pair, far_pair = identify_near_obstacle_pairs(thresh, numLabels, stats, centroids)
    processed_result = check_passages_by_near_pairs(5000, thresh, near_pair, labels)
    processed_binary_map = ~processed_result
    return processed_binary_map
def main():
    name = 'environment_2'
    map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.png')
    map = cv2.imread(map_path)
    map = cv2.resize(map, (480,360))

    start = time.time()
    processed_binary_map = dimension_integrator(map)
    end = time.time() - start
    processed_binary_map = cv2.resize(processed_binary_map, (960,720))
    print(end)
    cv2.imshow('final result', map)
    cv2.imshow('final result',processed_binary_map)
    cv2.waitKey(0)


if __name__=="__main__":
    main()