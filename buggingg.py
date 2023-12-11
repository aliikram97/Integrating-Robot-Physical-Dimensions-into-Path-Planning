import cv2
import numpy as np



# Your code to obtain the binary image and connected components
# map_path = r'C:\Users\Asus\Desktop\real simulatrion\ue/test_top_binary.png'
name ='environment_2'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/'+name+'.png')
map = cv2.imread(map_path)
map = ~map

imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)

output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
obstacle_number = numLabels-1
for centroid in centroids:
    radius = 20
    color = (255, 0, 0)
    thickness = 2
    x,y = centroid
    plot = (int(x),int(y))
    image = cv2.circle(map, plot, 2, color, thickness)
cv2.imshow('map',map)
cv2.waitKey(0)
print(thresh.shape)
i=1
while i<numLabels-1:
    cent1 = centroids[i]
    cent2 = centroids[i+1]
    x1,y1 = cent1
    x2,y2= cent2
    horizontal_difference = abs(x2-x1)
    vertical_difference = abs(y2-y1)
    i+=1
    print(f'the horizontal diff is [{horizontal_difference} || the vertical difference is [{vertical_difference}] ]')
print(f'number of obstacles is {obstacle_number}')
# Calculate distance transform
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# Calculate pairwise distances between object centroids
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
    return fraction   # Adjust this relationship based on your specific scenario
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

# Draw bounding boxes for near objects in blue and far objects in red
for near_pair in near_objects:
    i, j = near_pair
    x, y, w, h, _ = stats[i]
    cv2.rectangle(map, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue bounding box
    cv2.putText(map, f'Obj {i}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    x, y, w, h, _ = stats[j]
    cv2.rectangle(map, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue bounding box
    cv2.putText(map, f'Obj {j}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
print("Objects far from each other:", far_objects)
cv2.imshow('Objects Visualization', map)
cv2.waitKey(0)
cv2.destroyAllWindows()
