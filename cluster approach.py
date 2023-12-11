import cv2
import numpy as np

# Read the binary image
name = 'map_5'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
map = cv2.imread(map_path)
binary_image = map
map = ~map

imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)
# Apply distance transform
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)  # DIST_L2 for Euclidean distance
# Define thresholds for near and far objects
near_threshold = 10  # Adjust this threshold value
far_threshold = 20  # Adjust this threshold value

# Threshold the distance transform image
near_objects = np.where(dist_transform < near_threshold, 255, 0).astype(np.uint8)
far_objects = np.where(dist_transform > far_threshold, 255, 0).astype(np.uint8)

# Object labeling
output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# Calculate average area of objects
total_area = 0
for label in range(1, numLabels):  # Skip the background (label 0)
    total_area += stats[label, cv2.CC_STAT_AREA]

average_area = total_area / (numLabels - 1)  # Average area excluding background

# Calculate distance threshold based on image dimensions and average area
image_height, image_width = map.shape[:2]
image_diagonal = np.sqrt(image_height ** 2 + image_width ** 2)  # Diagonal length of the image

# List to store pairs of near objects and distances
near_pairs = []
distances = []

# Iterate through near objects to find pairs and calculate distances
for near_label in range(1, numLabels):  # Skip the background (label 0)
    near_obj_mask = np.uint8(labels == near_label)  # Mask for the current near object

    # Find contours of the current near object
    near_contours, _ = cv2.findContours(near_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate centroid of the current near object
    near_moments = cv2.moments(near_contours[0])
    near_cx = int(near_moments['m10'] / near_moments['m00'])
    near_cy = int(near_moments['m01'] / near_moments['m00'])

    # Iterate through far objects to find pairs
    for far_label in range(1, numLabels):  # Skip the background (label 0)
        if far_label != near_label:  # Avoid comparing the same object
            far_obj_mask = np.uint8(labels == far_label)  # Mask for the current far object

            # Find contours of the current far object
            far_contours, _ = cv2.findContours(far_obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate centroid of the current far object
            far_moments = cv2.moments(far_contours[0])
            far_cx = int(far_moments['m10'] / far_moments['m00'])
            far_cy = int(far_moments['m01'] / far_moments['m00'])

            # Calculate distance between centroids
            distance = np.sqrt((far_cx - near_cx) ** 2 + (far_cy - near_cy) ** 2)
            distances.append(distance)

# Calculate mean distance
mean_distance = np.mean(distances)

# Adjust distance threshold based on mean distance
distance_threshold = (average_area / (image_height * image_width)) * image_diagonal * 0.01  # Original calculation
print(distance_threshold)
distance_threshold = 1-distance_threshold
distance_threshold *= mean_distance

# List to store pairs of near objects after adjusting threshold
near_pairs_adjusted_threshold = []

# Iterate through distances to filter near pairs using adjusted threshold
print(distance_threshold)
print(distances)
for idx, distance in enumerate(distances):
    if distance < distance_threshold:
        near_pairs_adjusted_threshold.append((idx // (numLabels - 1) + 1, idx % (numLabels - 1) + 1))

# Output the pairs of near objects after adjusting threshold
print("Objects near each other with adjusted threshold:", near_pairs_adjusted_threshold)
