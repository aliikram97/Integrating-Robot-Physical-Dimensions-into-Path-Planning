import cv2
from matplotlib import pyplot as plt

# Load the binary image (replace 'image_path' with the path to your binary image)
image_path = r'C:\Users\Asus\Desktop\presentation waste/usefulpair_case.jpg'
binary_image = cv2.imread(image_path, 0)  # Read the image in grayscale
binary_image=~binary_image

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create an empty canvas to draw contours on
contour_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Draw contours on the canvas
cv2.drawContours(contour_image, contours, -1, (255, 255, 0), 8)  # -1 to draw all contours

# Display the original binary image and the image with contours
# plt.figure(figsize=(10, 5))
binary_image=~binary_image
# plt.subplot(1, 2, 1)
# plt.imshow(binary_image, cmap='gray')
# plt.title('Binary Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(~contour_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.axis('off')

plt.tight_layout()
plt.show()
