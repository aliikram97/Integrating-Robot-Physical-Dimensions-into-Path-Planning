# Main and helper function

from PIL import Image
import numpy as np
from RRT import RRT
from PRM import PRM
import cv2
import time
from constant import constants
import matplotlib.pyplot as plt


def load_map(file_path, resolution_scale):
    ''' Load map from an image and return a 2D binary numpy array
        where 0 represents obstacles and 1 represents free space
    '''
    # Load the image with grayscale
    img = Image.open(file_path).convert('L')
    # Rescale the image
    size_x, size_y = img.size
    new_x, new_y  = int(size_x*resolution_scale), int(size_y*resolution_scale)
    print('the new siz of the image is ',new_x,new_y)
    img = img.resize((new_x, new_y), Image.ANTIALIAS)

    map_array = np.asarray(img, dtype='uint8')

    # Get bianry image
    threshold = 127
    map_array = 1 * (map_array > threshold)

    # Result 2D numpy array
    return map_array


if __name__ == "__main__":
    # Load the map
    # start = (96,89)
    # goal  = (65,115)
    # # name = 'test_top_binary_processed'
    # name = 'Berlin_0_1024'
    # name = 'Sydney_2_256_500_'
    start = constants.start
    goal = constants.goal

    #path_to_orig = str(r'E:\Theis\concave_500/' + name + '.png')
    # path_to_orig = str(r'D:\Thesis\final results\selected-20231108T144248Z-001\selected\New folder/' + name + '.png')
    # path_to_orig = str(r'C:\Users\Asus\Desktop\real simulatrion\ue/' + name + '.png')
    # path_to_orig = str(r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\selected_maps/' + constants.name + '.png')
    path_to_orig = str(r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\results_elimination_pipleline\processed_input/' + constants.name + '.png')
    print('the path is ',path_to_orig)
    map_array = load_map(path_to_orig, 0.3)
    #path_to_closed_path = str(r'E:\Theis\resutls_paper_custom_maps_new/' + name + '.bmp')
    #map_array = load_map(path_to_closed_path, 0.3)

    # Planning class
    PRM_planner = PRM(map_array)
    RRT_planner = RRT(map_array, start, goal)

    # Search with PRM
    # a = PRM_planner.sample(n_pts=1000, sampling_method="uniform")
    # print('PRM_UNIFORM')
    # PRM_planner.search(start, goal)
    # a = PRM_planner.sample(n_pts=1000, sampling_method="random")
    # print('PRM_RANDOM')
    # PRM_planner.search(start, goal)
    # a = PRM_planner.sample(n_pts=2000, sampling_method="gaussian")
    # print('PRM_GAUSSIAN')
    # PRM_planner.search(start, goal)

    a = PRM_planner.sample(n_pts=20000, sampling_method="bridge")
    start_time = time.time()
    print('PRM_BRIDGE')
    PRM_planner.search(start, goal)
    end = time.time()
    prm_exe = end - start_time

    # Search with RRT and RRT*
    start_rrt=time.time()
    RRT_planner.RRT(n_pts=1000)
    end_rrt = time.time()
    rrt_exe = end_rrt-start_rrt
    print(f'the PRM time: {prm_exe}, rrt time: {rrt_exe}')
    # RRT_planner.RRT_star(n_pts=2000)
