# Standard Algorithm Implementation
# Sampling-based Algorithms PRM
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import math 
from numpy import random 
from scipy import spatial
import time
import cv2

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        
        #Covering all the points between P1 and P2
        Coverx = np.linspace(p1[0], p2[0]).astype(int)
        Covery = np.linspace(p1[1], p2[1]).astype(int)
        Cover = zip(Coverx, Covery)
        
        #If Collision or not
        obstacle = False 
        for m in Cover:
            if self.map_array[m[0], m[1]] == 0:
                obstacle = True
                
        return obstacle 
        
    
    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        #Calculating the euclidian distance 
        #Calculating the sqrt of x and y(difference between the length in x and y)
        euclidean = math.hypot((point1[0] - point2[0]),(point1[1] - point2[1]))
        
        return euclidean 


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #Uniformly seperating the plot and n_points between rows (rounding the values to nearest integers)
        rows = np.round(np.linspace(0, self.size_row -1, num = int(math.sqrt(n_pts))),decimals = 0)
       
        #Uniformly seperating the plot and n_points between columns (rounding the values to nearest integers)
        cols = np.round(np.linspace(0, self.size_col -1, num = int(math.sqrt(n_pts))),decimals = 0)
        
        
        #Checking if Obstacle or not for each points in the rows and columns 
        for row in rows:
            for col in cols:
                row = int(row)
                col = int(col)
                if self.map_array[row,col] != 0:
                    self.samples.append((row, col))
                    
                    

    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #Making an empty array to collect the points
        collect = []
        
        #Collecting all the random points between n_pts
        for pt in range(n_pts):
            #Collecting all the random points in the x and y axis from the plot 
            collect.append((random.randint(self.size_row) ,random.randint(self.size_col)))
            
        
        #Checking the collected points if collsiion or not 
        for pt in range(len(collect)):
            ptx = collect[pt][0]
            pty = collect[pt][1]
            if self.map_array[ptx][pty]==1:
                self.samples.append((ptx,pty))
        
        

            

    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))

        #Setting up an empty array for observed points
        detect=[]

        #Creating random integers between n_pts
        for pt in range(n_pts):
            #Checing if they are collsion or not and adding thenm to the empty list 
            if self.map_array[random.randint(self.size_row) ,random.randint(self.size_col)]==0:                
                detect.append((random.randint(self.size_row) ,random.randint(self.size_col)))
        
        
        #Checking all the points in the appended list 
        for pt in range(len(detect)):
            collsion=True
            
            
            while collsion:   
                #Evaluating the gaussian distance                              
                x1, y1 =int(np.random.normal(detect[pt][0],10)) , int(np.random.normal(detect[pt][1],10))
                
                #Checking if its obstacle or not 
                if x1<self.size_row and y1<self.size_col:
                    if self.map_array[(x1,y1)]== 1:
                        collsion=False
                        self.samples.append((x1,y1))
                        
                    
                    

    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))
        
        # Setting up an empty array for observed points
        detect=[]
        
        # Randomly initaising points from row and column and checking if collsion or not 
        for pt in range(n_pts):           
            if self.map_array[random.randint(self.size_row),random.randint(self.size_col)] == 0:                    
                detect.append((random.randint(self.size_row),random.randint(self.size_col)))
        
        #Search for another point using gaussian distance 
        for pt in range(len(detect)):
            ptx , pty = detect[pt][0], detect[pt][1]
            rx , ry = int(np.random.normal(ptx,13)) , int(np.random.normal(pty,13))
        
            #Checking if the another point if collsion or not(Free or not)
            if rx<self.size_row and ry<self.size_col:
                if self.map_array[(rx,ry)]== 0:
                
                    #Checking if the mid point if Collsiion or not (Free or not)
                    cx , cy = int((ptx+rx)/2) , int((pty+ry)/2)
                    if self.map_array[(cx,cy)] == 1:                                            
                        self.samples.append((cx,cy))
                    
                    
                    
              


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=5, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=30,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=30,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        name = 'test_top_binary-pro-PRMB'
        plt.savefig(str(r'C:\Users\Asus\Desktop\real simulatrion\ue/'+name+'.png'))
        # plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
            radius = 15
        elif sampling_method == "random":
            self.random_sample(n_pts)
            radius = 15
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
            radius = 25
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)
            radius = 40 
        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []                                                  
        positions=np.array(self.samples)
        kdtree = spatial.KDTree(positions)
        #Creating a KD tree pairs list 
        kd_pairs = list(kdtree.query_pairs(radius))                             
        
        #Checking for all points in Kdtree
        for pt in range(len(kd_pairs)):
            #Checking Collsion or not 
            collision = self.check_collision(self.samples[kd_pairs[pt][0]] ,self.samples[kd_pairs[pt][1]])   
            
            #if not collsion, Calculating the euclidian distance and appening their indexes 
            if collision == False:
                euclidian = self.dis(self.samples[kd_pairs[pt][0]],self.samples[kd_pairs[pt][1]])                
                if euclidian!=0:
                    indx , indy =self.samples.index(self.samples[kd_pairs[pt][0]]) , self.samples.index(self.samples[kd_pairs[pt][1]])
                    pairs.append((indx,indy,euclidian))
    
            else:
                continue 
                
        
        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from([])
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))
        
        
        
    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        
        #Initializing an start and goal empty list 
        start_pairs = []
        goal_pairs = []  
        #Setting up a goal radius
        goal_radius = 60
        
        #Evaluating the start pairs 
        for pt in range(len(self.samples)):
            
            # Calculating the starting point 
            start = self.samples[len(self.samples)-2]  
            #Calculating all the other points  
            pts = self.samples[pt]
            
            #Checking if collsion or not 
            if start!=pts:   
                if self.check_collision(start,pts)  == False:
                    #Calculating the distance 
                    length = self.dis(start,pts)
                    if length!=0 and length<goal_radius:                      
                        start_pairs.append(('start',self.samples.index(pts),length)) 
                else:
                    continue 
                                                       
        for pt in range(len(self.samples)):
            
            #Calculating the starting point 
            goal = self.samples[len(self.samples)-1]  
            #Calculating all the other points                
            pts = self.samples[pt]
            
            #Checking if collsion or not 
            if goal!=pts: 
                
                if self.check_collision(goal,pts) == False:
                    #calculating the distance 
                    length = self.dis(goal,pts)
                    if length!=0 and length<goal_radius:
                        goal_pairs.append(('goal',self.samples.index(pts),length))

                else:
                    continue 
        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        start_time = time.time()
        self.draw_map()
        execution = time.time() - start_time
        print(f'the time for the draw is {execution}')
        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        