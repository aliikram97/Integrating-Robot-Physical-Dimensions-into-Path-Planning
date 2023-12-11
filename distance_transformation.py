import cv2
import numpy as np

# Your code to obtain the binary image and connected components
# map_path = r'C:\Users\Asus\Desktop\real simulatrion\ue/test_top_binary.png'
name ='convex_maps_2'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/'+name+'.jpg')
map = cv2.imread(map_path)
map = ~map

imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)

output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
obstacle_number = numLabels-1
print(f'number of obstacles is {obstacle_number}')
# Calculate distance transform
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# Calculate pairwise distances between object centroids
distances = []
for i in range(1, numLabels):  # Start from 1 to exclude the background label 0
    for j in range(i + 1, numLabels):  # Compare current object with the rest
        dist = np.linalg.norm(centroids[i] - centroids[j])  # Euclidean distance between centroids
        distances.append(dist)

# Determine a dynamic threshold as a fraction of the average or maximum distance


dynamic_threshold = 0.7 * np.mean(distances)

print(f'the distances are {distances}')
print(dynamic_threshold)# You can use np.mean or np.max based on your preference

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

# Draw bounding boxes for near objects in blue and far objects in red
for near_pair in near_objects:
    i, j = near_pair
    x, y, w, h, _ = stats[i]
    cv2.rectangle(map, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue bounding box
    cv2.putText(map, f'Obj {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    x, y, w, h, _ = stats[j]
    cv2.rectangle(map, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue bounding box
    cv2.putText(map, f'Obj {j}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

for far_pair in far_objects:
    i, j = far_pair
    x, y, w, h, _ = stats[i]
    cv2.rectangle(map, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red bounding box
    cv2.putText(map, f'Obj {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    x, y, w, h, _ = stats[j]
    cv2.rectangle(map, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red bounding box
    cv2.putText(map, f'Obj {j}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the image with bounding boxes and labels
print("Objects near each other:", near_objects)
cv2.imshow('Objects Visualization', map)
cv2.waitKey(0)
cv2.destroyAllWindows()
