# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from random import randrange 
from numpy import random 
import math 
from scipy import spatial


# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag
        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    
    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        
        #Calculating euclidian distance 
        euclidean = math.hypot((node1.row - node2.row),(node1.col - node2.col))
        return euclidean 


    
    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        #Collecting all the points between node1 and node2:
        collectpoint = zip(np.linspace(node1.row, node2.row).astype(int), np.linspace(node1.col, node2.col).astype(int))
        
        # If collision or not  
        for m in collectpoint:
            #1 -> free 0 -> obstacle 
            if self.map_array[m[0]][m[1]] == 0:
                return True
        return False
    



    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
       
        #Choosing a variable for probability (Randomly)
        goal =randrange(200)
        
        #Checking if the goal is withitn the goal bias or not
        # If bias value outside goal value, return goal
        if goal <= goal_bias: return self.goal
        #If value within goal, return the coordinates of the new generated point. 
        else: return Node(random.randint(self.size_row),random.randint(self.size_col))
    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        #Calculate teh distance of all the nodes and return with the minimum distance 
        length = []
        
        for pt in range(len(self.vertices)):
            #Adding all the length with respect to the node:
                leng = self.dis(point, self.vertices[pt])
                length.append(leng)
        
        #returning vertices with the minimum length
        return self.vertices[length.index(min(length))]
    
    
    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        ### YOUR CODE HERE ###
        #Calculate all the neighboours
        inrange = []
        considered = []
        
        for pt in range(len(self.vertices)):  
            
            #Checking if the length in the range of neighbour_size or not 
            length=self.dis(new_node,self.vertices[pt])
            
            if length<=neighbor_size:                     
                inrange.append(self.vertices[pt])

        #Checking obstacles
        for pt in range(len(inrange)):                        
            obstacle=self.check_collision(inrange[pt],new_node)
            
            #if not obstacles, considereing the points 
            if obstacle == False:
                considered.append(inrange[pt])
            else:
                continue
            
        return considered                            



    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        length=[]
        pts = []
        
        for pt in range(len(neighbors)):    
            #Checking if obstacle or not                     
            if self.check_collision(neighbors[pt],new_node) == True:
                continue
            else:
                pts.append(neighbors[pt])
        
        #Evaluating the cost 
        for pt in range(len(pts)):                             
            update=pts[pt].cost+self.dis(pts[pt],new_node)
            length.append(update)
        
        #Calcualting the smallest index and calculating the cost 
        smallest =length.index(min(length))
        new_node.parent=neighbors[smallest]                     
        new_node.cost=neighbors[smallest].cost+int(self.dis(neighbors[smallest],new_node)) 

        #Updating the lowest cost 
        for pt in range(len(pts)):                             
            old = pts[pt].cost                           
            updated = int(self.dis(new_node,pts[pt])) + new_node.cost

            if old > updated:
                neighbors[pt].cost = updated
                neighbors[pt].parent = new_node
                
            else:
                continue


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=8, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=10, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=10, marker='o', color='r')

        # show image
        name = 'test_top_binary-pro-RRT'
        plt.savefig(str(r'C:\Users\Asus\Desktop\real simulatrion\ue/' + name + '.png'))
        # plt.show()

    def direction(self,point1,point2):                          
        
        step=10 
        
        #Calculating the length of the nearest neighbour in the direction of the angle and comparing it with the steps 
        length = self.dis(point1,point2)
        if length > step:
            length = step
        
        #Calculating the angle and updating the new node 
        col = point2.col - point1.col
        row = point2.row - point1.row
        ang = math.atan2(col, row)
        update = Node((int((point1.row+length*math.cos(ang)))),(int((point1.col+length*math.sin(ang)))))

        return update 


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        #Keeping the probabilty to one percent while picking points 
        goal_bias = 1                               
        step=10
        
        for pt in range(n_pts):
            # Creating new Points  
            ptn = self.get_new_point(goal_bias)
            #Checking obstacles 
            if self.map_array[ptn.row][ptn.col]==1: 
                # Nearest node 
                node = self.get_nearest_node(ptn)    
                #Updating node alone the new point
                update =  self.direction(node,ptn)      

                #Checking if obstacle or not 
                collision = self.check_collision(node,update) 
                if collision == False:
                    # Adding node and updating the cost relative to the nearest node 
                    self.vertices.append(update)  
                    update.parent=node  
                    update.cost = int(node.cost +self.dis(node,update))      
                                  

                    #Comparing the step with the steps 
                    if self.dis(update,self.goal) <= step:          
                        
                        #Updating the values 
                        temp = update               
                        node, update  = temp, self.goal 
                        
                        #Updating the parents and cost 
                        update.cost = int(node.cost +self.dis(node,update))
                        update.parent = node  
                        
                        #Adding the vertices 
                        self.vertices.append(update)
                        self.found=True 

                        break   
                else: 
                    continue 
            else: 
                continue 


        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        
        # Setting up the probability and steps 
        step=10
        goal_bias = 4
        
        
        condition = False
        
        for pt in range(n_pts):
            # Getting new coordinates 
            ptn = self.get_new_point(goal_bias)   
            
            # Checking if obstacle or not 
            if self.map_array[ptn.row][ptn.col] == 1:
                neigh  = self.get_nearest_node(ptn) 
                
                #Updating final goal 
                if condition == False:                            
                    update = self.direction(neigh,ptn) 
                    
                else:
                    update = self.goal
                     
                
                #Checking obstacles     
                if self.check_collision(update,neigh) == False:
                    
                    #Finding neighbours and Rewire
                    self.rewire(update,self.get_neighbors(update,20) )  
                    
                    length  = self.dis(update,self.goal)   
                    
                    if length <= step:
                        condition = True
                        
                    self.vertices.append(update)

                    if length == 0:
                        condition = False
                        self.found=True
                    
                    else: 
                        continue  
            else: 
                continue 

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()