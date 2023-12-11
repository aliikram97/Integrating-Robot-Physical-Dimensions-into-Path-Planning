# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(2)
def BlackboxMaker(frame,r):
    image =frame
    start_point = (int(r[1]),int((r[1]+r[3])))
    end_point = (int(r[0]),int(r[0] + r[2]))
    color = (0,0,0)
    thickness = -1
    cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
frame_count = 0
counter = 1
obstacles = []
real_world_height = 8.3
real_world_weight = 4.5
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        if frame_count==0:
            r = cv2.selectROI("Object size checker", frame)
            # Crop image
            cropped_image = frame[int(r[1]):int(r[1] + r[3]),
                            int(r[0]):int(r[0] + r[2])]
            print('the size of the robot (width x height) is ',r[2],r[3],'in pixel units')
            resolution_1 = real_world_height / r[3]
            resolution_2 = real_world_weight / r[2]
            resolution = resolution_1+resolution_2
            resolution = resolution / 2
            print('resolution based on W is',resolution_2,'resolution based on H is ',resolution_1)
            print('resolution is',resolution)

        while counter<3:
            r = cv2.selectROI(str("select the obstacle "+str(counter)), frame)
            obstacles.append(r)
            counter+=1
        for point in obstacles:
            l,t,w,h = point
            start_point = (l,t)
            end_point = (int(l+w),int(t+h))
            color = (0, 0, 0)
            thickness = -1
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            # print(l,t,w,h)
        cv2.imshow('blocks', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noise = cv2.medianBlur(gray, 3)
        # thresh1 = cv2.threshold(noise, 80, 255, cv2.THRESH_BINARY)[1]
        thresh2 = cv2.threshold(noise,80,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # cv2.imshow('binary', thresh1)
        cv2.imshow('otsu+binary', thresh2)
        if cv2.waitKey(2) & 0xFF == ord('s'):
            print('save')
            cv2.imwrite('/home/aliikram-99tech/Thesis_work/test/map_binary.bmp',thresh2)
        frame_count+=1

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
