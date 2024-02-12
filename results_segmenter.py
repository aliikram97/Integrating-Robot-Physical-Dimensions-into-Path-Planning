import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_objects_with_labels(binary_image):
    # Find connected components
    def detect_and_label_0bstacles(map):
        map = ~map

        imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 50, 255, 0)

        output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
        return output, thresh

    # Plot objects with labels on centroids
    output,image = detect_and_label_0bstacles(binary_image)
    num_labels, labels, stats, centroids = output
    for label in range(1, num_labels):
        object_mask = (labels == label).astype(np.uint8) * 255
        object_centroid = (int(centroids[label][0]), int(centroids[label][1]))

        # Get a random color for this object


        # Draw object
        # output_image = cv2.drawContours(output_image, [
        #     cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, color, -1)

        # Put label on centroid
        cv2.putText(binary_image, f"{label}", object_centroid, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
        cv2.circle(binary_image, object_centroid, 3, (255, 255, 255), -1)

    return binary_image


# Example binary image (replace this with your own binary image)
binary_image = cv2.imread(r'C:\Users\Asus\Desktop\presentation waste/usefulpair_case.jpg')

# Plot objects with labels on centroids
output = plot_objects_with_labels(binary_image)

# Display the output
plt.imshow(output)
plt.axis('off')
plt.show()

# plt.savefig(r'C:\Users\Asus\Desktop\presentation waste/segmented.png',)
