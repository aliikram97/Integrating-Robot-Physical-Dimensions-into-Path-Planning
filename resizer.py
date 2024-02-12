import cv2
import numpy as np
import matplotlib.pyplot as plt
import os 

path = r'C:\Users\Asus\Desktop\paper_specific_output\input\results/'
dir_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#print('files are ',dir_list)
for item in dir_list:
    item_name = item.split(".")
    print('item name is ',item_name[0])


    name = item_name[0]
    extension = item_name[1]
    path_to_image = str(path+ name + '.'+extension)

    image = cv2.imread(path_to_image)

    # Loading the image

    # bigger = cv2.resize(image, (250, 250))
    bigger = cv2.resize(image, (500, 500))

    cv2.imwrite(str(r'C:\Users\Asus\Desktop\paper_specific_output\input\results/'+name+'.png'),bigger)








