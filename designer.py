import cv2

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
    for x, y in points:
        if image[y, x] != 0:
            return False
    return True

# Replace 'your_image.png' with the path to your binary image file
image_path = 'your_image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    x1, y1 = 10, 20  # Replace with your first pixel coordinates
    x2, y2 = 100, 150  # Replace with your second pixel coordinates

    line_points = get_line(x1, y1, x2, y2)
    result = check_points(image, line_points)

    print(f"Are all points in the line free space? {result}")
else:
    print("Image not found!")
