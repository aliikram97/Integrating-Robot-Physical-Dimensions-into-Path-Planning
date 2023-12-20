import math


def points_towards_centroid(centroid, other_points):
    centroid_x, centroid_y = centroid

    # Calculate distances between centroid and other points
    distances = {}
    for index, point in enumerate(other_points):
        point_x, point_y = point
        distance = math.sqrt((point_x - centroid_x) ** 2 + (point_y - centroid_y) ** 2)
        distances[index] = distance

    # Find points closer to the centroid
    closer_points_indices = [index for index, distance in distances.items() if distance < max(distances.values())]
    closer_points = [other_points[index] for index in closer_points_indices]

    return closer_points


# Example usage:
centroid = (0, 0)  # Replace with your centroid coordinates
other_points = [(1, 1), (2, 2), (-1, -1), (5, 5), (-3, 0)]  # Replace with your other points

closer_points = points_towards_centroid(centroid, other_points)
print("Points closer to the centroid:", closer_points)
