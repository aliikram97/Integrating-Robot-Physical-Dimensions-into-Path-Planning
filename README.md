This is an implementation repository of dimension integrator. This repository solves the problem statement "**integrate robot dimensions into path planning**" The algorithm uses computer vision technique to alter the map in a manner which would close all the passages which are smaller than the robot width.

To run the program in the repository you will see a Python file named **constants.py**, in this file you will provide the path to the binary map and where the output should be stored.
    
    robot_width = 20 #enter the robot width in pixels
    name_BM = "sample_map_1"
    path_to_original_BM = str(r"path to input directory/"+name_BM+".png")
    path_to_processed_BM = str(r"path to output directory/"+name_BM)

Above is a code excerpt demonstrating how to use the algorithm. The file **optimization.py** includes the entire algorithm which will process the 2D-Binary map and store the processed map to the given path. to use this algorithm with the 
path planning algorithm you can import the function **dimension_integrator** from **optimized_pipeline.py** and give it the input binary map and robot width in pixels as arguments. Example of this is given below:

    from optimized_pipeline import dimension_integrator
    processed_binary_map,result_visualization = dimension_integrator(map,robot_width)

Experimentation videos can be found in the folder of **experimentation results**.
