import cv2

name = 'map_5'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/' + name + '.jpg')
map = cv2.imread(map_path)
map = ~map

imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)


def slope_(x1, y1, x2, y2):
    if x1 != x2:
        s = (y2 - y1) / (x2 - x1)
    else:
        s = 0
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
point1 = (368, 99)
point2 = (238, 128)
points = line_equation(point1,point2,1)
non_zero = False
for point in points:
    x, y = point
    x = round(x)
    y = round(y)
    metric_value = thresh[int(x)][int(y)]
    if metric_value==255:
        print(metric_value)
        non_zero=True
        break
if not non_zero:
    cv2.line(thresh, point1, point2, (127, 127, 127), thickness=2, lineType=8)
cv2.imshow('resu;t',thresh)
cv2.waitKey(0)