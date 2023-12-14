import cv2
import numpy as np



def detect_and_label_0bstacles(map):
    map = ~map

    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)

    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    return output,thresh
def near_objects_identifier(map,output):
    (numLabels, labels, stats, centroids) = output
    h, w, ch = map.shape

    # mean_area = np.mean(stats[1:, cv2.CC_STAT_AREA])  # Exclude background label
    #
    # # Normalize mean area between 0 and 1
    # normalized_area = (mean_area - np.min(stats[1:, cv2.CC_STAT_AREA])) / (
    #             np.max(stats[1:, cv2.CC_STAT_AREA]) - np.min(stats[1:, cv2.CC_STAT_AREA]))

    # print(normalized_area)
    # diagonal points
    top_diag = (0, 0)
    bottom_diag = (abs(int((w - 1))), abs(int((h - 1))))
    diagonal_distance = np.linalg.norm(np.array(top_diag) - np.array(bottom_diag))
    threshold_distance = diagonal_distance * 0.4
    # threshold_distance = diagonal_distance * normalized_area
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

            # cv2.imshow('frame', binary_map_temp)
            # cv2.waitKey(0)
            # Compare distance with threshold
            if dist_between_objects < threshold_distance:
                print(f"Objects {obj1_label} and {obj2_label} are near")
                near_objects.append((obj1_label, obj2_label))
    return near_objects


name = 'map_5'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
map = cv2.imread(map_path)
# Assuming you have a labeled image 'labels' with objects identified by unique labels
output, thresh = detect_and_label_0bstacles(map)
near = near_objects_identifier(map,output)
print(near)