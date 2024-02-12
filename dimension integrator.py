import cv2
import numpy as np
import random
import time


def dimension_integrator(map,threshold,name):
    def slope_(x1, y1, x2, y2):
        if x1!=x2:
            s = (y2 - y1) / (x2 - x1)

        else:
            s=0
        return s


    def line_equation(coord1, coord2, step):
        points = []
        x1, y1 = coord1
        x2, y2 = coord2
        # slope, intercept = np.polyfit(coord1, coord2, 1)
        # slope, intercept, r_value, p_value, std_err = linregress([x1, x2], [y1, y2])
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

    def distance(a, b, threshold, image, map_):
        dist = []
        connect_line = True
        i, j = 0, 0
        list_empty_a = []
        list_empty_b = []
        image_values = np.array(thresh)
        print(image_values)

        if len(a) > len(b):
            m = a
            n = b
        elif len(b) > len(a):
            m = b
            n = a
        else:
            m = a
            n = b

        test1 = len(m)
        test2 = len(n)
        while i < len(m):
            j = 0
            while j < len(n):
                point1 = m[i]
                point1 = np.array(point1)
                point2 = n[j]
                point2 = np.array(point2)
                R = random.randint(0, 255)
                G = random.randint(0, 255)
                B = random.randint(0, 255)
                colour = (G, B, R)

                points = line_equation(point1, point2, 0.5)
                counter = 2
                freespace_tester = []
                while counter < len(points):
                    point = points[counter]
                    # metric_value = thresh[int(point[0])][int(point[1])]
                    metric_value = thresh[int(point[1])][int(point[0])]

                    # metric_tester = (point,metric_value)
                    metric_tester = (metric_value)
                    freespace_tester.append(metric_tester)
                    counter += 3
                # cv2.line(map, point1, point2, (255, 0, 255), thickness=1, lineType=8)
                if sum(freespace_tester) == 0 and freespace_tester != []:
                    print('LINE PIXELS  = ', sum(freespace_tester))
                    # cv2.circle(map, point1, 5, colour, 2)
                    # cv2.circle(map, point2, 5, colour, 2)
                    # cv2.imshow('debug screen object boundary', map)
                    # cv2.imshow('debug screen object thresh', thresh)
                    # cv2.waitKey(0)
                    # cv2.circle(map, point1, 3, (127, 255, 127), 2)
                    # cv2.circle(map, point2, 3, (127, 255, 127), 2)
                    d = np.linalg.norm(point2 - point1)
                    if d < threshold:
                        cv2.line(map, point1, point2, (255, 0, 255), thickness=2, lineType=8)
                        cv2.line(map_debug, point1, point2, (255, 255, 255), thickness=3, lineType=8)


                # cv2.imshow('debug screen line of sight',map)
                # cv2.waitKey(0)
                # tester = sum(freespace_tester)
                # if tester==0 or tester==255 or tester==510 or tester==1020:
                #     print('the tester is ', tester)
                #     cv2.line(map, point1, point2, (255, 0, 255), thickness=1, lineType=8)
                # if connect_line:
                #     R=random.randint(0,255)
                #     G=random.randint(0,255)
                #     B=random.randint(0,255)
                #     colour = (G,B,R)
                #
                #     cv2.line(map, point1, point2, colour, thickness=1, lineType=8)
                # elif not connect_line:
                #     print('leaving the line')

                j += 5

            i += 5

    def cordinates_maker(contours):
        pixel_values = []
        for full in contours:
            for point in full:
                x = point[0][0]
                y = point[0][1]
                pixel_values.append([x, y])
        return pixel_values

    # start = t.time()
    x = 1
    cont_itter = 1
    # threshold=400
    total_contours = []
    object_points_a = []
    object_points_b = []
    tester = []

    map = cv2.resize(map, (250, 250))
    # h, w, channel = map.shape

    map = ~map
    map_debug = map.copy()
    imgray = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    # imgray = map
    ret, thresh = cv2.threshold(imgray, 50, 255, 0)
    output = cv2.connectedComponentsWithStats(thresh, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    # threshold= 400
    contours, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    itter = 0
    empty_image = np.full((250, 250), 255, dtype=np.uint8)

    while (x < numLabels):
        object = (labels == x).astype("uint8") * 255
        object_ = (labels == x).astype("uint8") * 58
        empty_image = empty_image + object_
        ret1, thresh_obj = cv2.threshold(object, 127, 255, 0)
        contours_obj, hierarchy1 = cv2.findContours(thresh_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cordinates_maker(contours_obj)
        # while itter in range(len(cnts)):
        #     p1,p2=cnts[itter]
        #     itter+=1

        cv2.drawContours(object, contours_obj, -1, (255, 255, 255), 100)
        total_contours.append(cnts)
        cv2.imwrite('./test_object1.jpg', object_)
        x += 1
    # a=total_contours[0][0]
    # b=total_contours[1][0]
    # for (i,j) in zip(a,b):
    #     point=i
    #     haha=j
    total_number_contours = len(total_contours)
    empty_image = cv2.resize(empty_image, (500, 500))
    # cv2.imshow('testing image', empty_image)
    # cv2.waitKey(0)
    # while(cont_itter<total_number_contours-1 or cont_itter== total_number_contours-1):
    while (cont_itter < total_number_contours):
        inner_itter = 0

        # while(inner_itter<total_number_contours-1 or inner_itter==total_number_contours-1):
        while (inner_itter < total_number_contours and cont_itter < total_number_contours):
            if cont_itter != inner_itter:
                a = total_contours[cont_itter]
                b = total_contours[inner_itter]
                # cv2.circle(map, (int(z), int()), 1, (0, 255, 255), 2)

                t = threshold
                distance(a, b, t, thresh, map)

                inner_itter += 1
                print('inner = ', inner_itter)
            else:
                print('equal')
                inner_itter += 1
                # cont_itter+=1
                print('new_inner = ', inner_itter)
        cont_itter += 1
        print('outer', cont_itter)

    # cv2.drawContours(map, contours, -1, (255, 0, 255), 3)
    map = ~map
    map_debug = ~map_debug
    map = cv2.resize(map, (500, 500))
    map_debug = cv2.resize(map_debug, (500, 500))
    # cv2.imwrite(str(r'E:\Theis\line_of_sight_optimization/'+name+'_.png'),map)
    cv2.imwrite(str(r'C:\Users\Asus\Desktop\real simulatrion\ue/'+name+'_scd.png'),map_debug)
    cv2.imwrite(str(r'C:\Users\Asus\Desktop\real simulatrion\ue/'+name+'test_.png'),map)
    # end = time.time()
    # cv2.putText(map, str('execution time = '), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(map, str(round(execution_time, 2)), (150, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(map, str('size = '), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(map, str((w, h)), (60, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 0, 255), 1, cv2.LINE_AA)
    return map_debug

name ='simple_square'
map_path = str(r'C:\Users\Asus\Desktop\presentation waste\dd/'+name+'.png')
map = cv2.imread(map_path)
start = time.time()
processed_map = dimension_integrator(map,20,name)
execution_time = time.time() - start
print(f'the time taken is: {execution_time}')
cv2.imshow('view',processed_map)
cv2.waitKey(0)