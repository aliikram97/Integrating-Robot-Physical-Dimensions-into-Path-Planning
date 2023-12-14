import cv2
import numpy as np

def detect_and_label_0bstacles(map):
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh

name = 'map_5'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
map = cv2.imread(map_path)
# Assuming you have a labeled image 'labels' with objects identified by unique labels
output, thresh = detect_and_label_0bstacles(map)
(numLabels, labels, stats, centroids) = output
# Perform distance transform
# dist_transform = cv2.distanceTransform(labels, cv2.DIST_C, 3)

# Calculate the threshold value (40% of image width or height)
# threshold_distance = max(map.shape) * 0.43
h,w,ch = map.shape
#diagnal points
# diagonal points
top_diag = (0, 0)
bottom_diag = (abs(int((w - 1))), abs(int((h - 1))))
diagonal_distance = np.linalg.norm(np.array(top_diag) - np.array(bottom_diag))

# vertical points
top_vert = (int((w / 2)), 0)
bottom_vert = (int(w / 2), int(h))
vertical_distance = np.linalg.norm(np.array(top_vert) - np.array(bottom_vert))

# horizontal points
top_hor = (0, int(h / 2))
bottom_hor = (w, int(h / 2))
horizontal_distance = np.linalg.norm(np.array(top_hor) - np.array(bottom_hor))
# cv2.circle(map, top_hor, 5, (255, 0, 0), -1)
# cv2.circle(map, bottom_hor, 5, (255, 0, 0), -1)
# cv2.imshow('dexter',map)
# cv2.waitKey(0)
# print(f'the diag is {diagonal_distance * 0.4}, the vertical dist is {vertical_distance *0.4}, horizontal distance is {horizontal_distance*0.4}')

threshold_distance = diagonal_distance *0.4
# Loop through each unique object label
near_objects = []
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


        # Combine binary masks
        binary_map_temp = obstacle_1 + obstacle_2
        cv2.circle(binary_map_temp, (int(centroid_obj1[0]), int(centroid_obj1[1])), 5, (127, 255, 0), -1)
        cv2.circle(binary_map_temp, (int(centroid_obj2[0]), int(centroid_obj2[1])), 5, (127, 255, 0), -1)


        # dist_transform = cv2.distanceTransform(binary_map_temp, cv2.DIST_L2, 3)
        dist_between_objects = np.linalg.norm(centroid_obj1 - centroid_obj2)
        # print(f'the distance is {dist_between_objects}')
        # print(
        #     f'the diag is {diagonal_distance * 0.4}, the vertical dist is {vertical_distance * 0.4}, horizontal distance is {horizontal_distance * 0.4}')
        # cv2.imshow('frame', binary_map_temp)
        # cv2.waitKey(0)
        # distance_in_horizontal = abs((centroid_obj1[0]-centroid_obj2[0]))
        # distance_in_vertical = abs((centroid_obj1[1]-centroid_obj2[1]))
        # dist_between_objects = max(distance_in_horizontal,distance_in_vertical)


        # Calculate distance between obj1_label and obj2_label
        # dist_between_objects = np.max(dist_transform[binary_map_temp != 0])
        # print(dist_between_objects)
        # print(threshold_distance)

        # Compare distance with threshold
        if dist_between_objects < threshold_distance:
            print(f"Objects {obj1_label} and {obj2_label} are near")
            near_objects.append((obj1_label,obj2_label))
print(near_objects)

