import cv2
import numpy as np

counter = 1
obstacles = []
frame_path  = r'C:\Users\Asus\Desktop\real simulatrion\ue/top_test.png'
frame =cv2.imread(frame_path)

while counter < 3:
    r = cv2.selectROI(str("select the obstacle " + str(counter)), frame)
    obstacles.append(r)
    counter += 1
for point in obstacles:
    l, t, w, h = point
    start_point = (l, t)
    end_point = (int(l + w), int(t + h))
    color = (0, 0, 0)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
noise = cv2.medianBlur(gray, 3)
# thresh1 = cv2.threshold(noise, 80, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(noise,80,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('binary', thresh1)
cv2.imwrite(r'C:\Users\Asus\Desktop\real simulatrion\ue/test_top_binary.png',thresh2)
cv2.imshow('otsu+binary', thresh2)

cv2.waitKey(0)