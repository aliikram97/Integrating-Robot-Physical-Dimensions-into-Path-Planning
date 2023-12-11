"""
Created on Sat Feb  5 12:18:52 2022

@author: chinmaya
"""

# Basic searching algorithms
from queue import Queue
from queue import PriorityQueue
import math
import copy

# Class for each node in the grid
class Node:
    def __init__(self, row, col, is_obs, h):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.is_obs = is_obs  # obstacle?s
        self.g = 0         # cost to come (previous g + moving cost)
        self.h = h            # heuristic
        self.cost = 1      # total cost (depend on the algorithm)
        self.parent = [0,0]    # previous node
        self.visited = False
        
        
def path_generation(matrix, start, goal):
    found = False
    path = []
    
    if matrix[goal[0]][goal[1]].parent is None:
        found = False
        path.clear()
    else:
        found = True
        t = goal.copy()
        while (t != start):
            path.append(t)
            t = [matrix[t[0]][t[1]].parent[0], matrix[t[0]][t[1]].parent[1]]
        path.append(start)
        path.reverse()
    
    return found, path
        

def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False
    
    #Making a 2D grid which stores all the object corrosponding to that node. 
    matrix = []
    for i in range(len(grid)):
        x = []
        for j in range(len(grid[0])):
            #Saving all the nodes by running into for loop
            object = Node(i,j,grid[i][j], None)
            #Initialising distance of all the nodes as infinity(-1).
            object.g = -1
            x.append(object)
        matrix.append(x)
    
    #Making an Empty queue
    queue = Queue()
    
    #Initialising the Starting of the Object
    s = start
    matrix[s[0]][s[1]].g = 0
    
    #Putting the starting node into the queue
    queue.put(s)
    u = s
    
    #Running the loop untill u reaches the goal
    while u != goal:
        
        #Updating U value
        u = queue.get()
        #Making an empty neighbouring list
        n_list = []
        
        #Adding all the neighbours to the list        
        if u[1]+1 < len(grid[0]):
            R = [u[0],u[1]+1]
            n_list.append(R)
        if u[0]+1 < len(grid):
            D = [u[0]+1,u[1]]
            n_list.append(D)
        if u[1]-1 >= 0:
            L = [u[0],u[1]-1]
            n_list.append(L)
        if u[0]-1 >= 0:
            T = [u[0]-1,u[1]]
            n_list.append(T)
        
        
        #Running over all the neighbours
        for v in n_list:
            #Checking if the neighbour is not an obsticle
            if not matrix[v[0]][v[1]].is_obs:
                #Checking is the neighbour is not obsevered/visited.
                if not matrix[v[0]][v[1]].visited:
                    
                    #Adding one distance to the next neighbour
                    matrix[v[0]][v[1]].g = matrix[u[0]][u[1]].g + 1
                    #Making u as parent of the neighbour node
                    matrix[v[0]][v[1]].parent = u
                    #Adding neighbour to the queue
                    queue.put(v)
        
        #Changing the parent node to visited. 
        matrix[u[0]][u[1]].visited= True

    #calculating the steps(Number of node visited)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
             if (matrix[i][j].visited == True):
                 steps = steps+1
           
    #Generating the path
    found, path = path_generation(matrix, start, goal)

               
    if found:
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        print("No path found")
    return path, steps


#Making a DFS Sub-function
def dfs_path(grid,matrix,u):
    
    #Assinging the U coodinate as visited
    matrix[u[0]][u[1]].visited = True
    # steps = steps+1
    
    #Creating a neighbouring list
    n_list = []
    
    #Creating all the 4 neighbours and appening them to the list
    if u[1]+1 < len(grid[0]):
        R = [u[0],u[1]+1]
        n_list.append(R)
    if u[0]+1 < len(grid):
        D = [u[0]+1,u[1]]
        n_list.append(D)
    if u[1]-1 >= 0:
        L = [u[0],u[1]-1]
        n_list.append(L)
    if u[0]-1 >= 0:
        T = [u[0]-1,u[1]]
        n_list.append(T)
        
    #Looping over all the neighbours     
    for v in n_list:
        #Checking if the neighbour is an obsticle or not
        if not matrix[v[0]][v[1]].is_obs:
            #Checking if the neighbour is visited or not 
            if not matrix[v[0]][v[1]].visited:
                #Assigning u as parent of the neighbour v
                matrix[v[0]][v[1]].parent = u
                #Calling the recussion function 
                dfs_path(grid,matrix,v)
    return

def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False
    
    
    #Making a 2D grid which stores all the object corrosponding to that node. 
    matrix = []
    for i in range(len(grid)):
        x = []
        for j in range(len(grid[0])):
            #Saving all the nodes by running into for loop
            object = Node(i,j,grid[i][j], None)
            #Initialising distance of all the nodes as infinity(-1).
            object.g = -1
            x.append(object)
        matrix.append(x)
    

    #Making Distance of the start node as 0. 
    matrix[start[0]][start[1]].g = 0
    
    u = [0,0]
    # Visiting all the nodes of the grid 
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            #Checkinf if the matrix is visited or not
            if not matrix[i][j].visited:
                u[0] = matrix[i][j].row
                u[1] = matrix[i][j].col
                #Calling dfs_path function 
                steps = dfs_path(grid,matrix,u)
   
   
    #Generating the path
    if matrix[goal[0]][goal[1]].parent is None:
        found = False
        path.clear()
    else:
        found = True
        t = goal.copy()
        while (t != start):
            path.append(t)
            t = [matrix[t[0]][t[1]].parent[0], matrix[t[0]][t[1]].parent[1]]
        path.reverse()
        
    steps = len(path)    
    path[0] = start
    


    
    if found:
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        print("No path found")
    return path, steps



def dijkstra(grid, start, goal):
    '''Return a path found by Dijkstra alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dij_path, dij_steps = dijkstra(grid, start, goal)
    It takes 10 steps to find a path using Dijkstra
    >>> dij_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ## YOUR CODE HERE ###
    path = []
    node_visited= set()
    steps = 0
    found = False
    
    #Making a 2D grid which stores all the object corrosponding to that node. 
    matrix = []
    for i in range(len(grid)):
        x = []
        for j in range(len(grid[0])):
            #Saving all the nodes by running into for loop
            object = Node(i,j,grid[i][j], None)
            #Initialising distance of all the nodes as infinity(-1).
            object.g = -1
            x.append(object)
        matrix.append(x)

    #Making an Empty queue
    queue = []
    
    #Assisning S variable as the start of the function
    s = start
    
    #Making Distance of the start node as 0. 
    # matrix[s[0]][s[1]].parent = None
    u= Node(s[0], s[1], False, None)
    u.g = 0
    u.parent = None

    #Inserting s in queue
    node_visited.add(tuple(s))
    
    #Running a loop untill it reaches goal
    while not found:
        
        # Sorting the queue according to the distance 
        queue.sort(key=lambda fun: fun[1])
        
        #Making an Empty list
        n_list = []
        
        #Evaluating the neighbours
        neighbours = [[0, +1], [+1, 0], [0, -1], [-1, 0]]
        
        for m, n in neighbours:
            if u.row + m in range(len(grid)) and u.col + n in range(len(grid[0])) and grid[u.row + m][u.col + n ] == 0:
                n_list.append((u.row + m,u.col + n ))
            n_list.reverse()
            
        #Destination reached (Base case)
        if (u.row, u.col) == tuple(goal):
            found = True
            break
        
        else:
        #Running over all the neighbours
            for v in n_list:
                #Checking is the neighbour is not obsevered/visited.
                if v not in node_visited:
                    #Adding new node 
                    newnode= Node(v[0], v[1], False, None)
                    
                    #Calculating the updating distance 
                    alt = newnode.g + u.g
                    
                    #Updating the distance and parent of the node
                    newnode.g = u.g + 1
                    newnode.parent = u
                    
                    #Apeending to the queue
                    queue.append((newnode, newnode.g))
            
            #Poping u from the queue
            u = (queue.pop(0))[0]
            #Calcuating the visited nodes
            node_visited.add((u.row, u.col))


    #Path generation 
    while u.parent is not None:
        #Appending the path 
        path.append([u.row, u.col])
        #Retracting the parents
        u = u.parent
        
    #Adding Start position to the to the path     
    path.append(start)
    #Reversing the path 
    path.reverse()


    #Calculation of the steps + 1 for the start node
    steps = len(node_visited)+1


    if found:
        print(f"It takes {steps} steps to find a path using Dijkstra")
    else:
        print("No path found")
    return path, steps



def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''

    ## YOUR CODE HERE ###
    path = []
    node_visited = set()
    steps = 0
    found = False
    
    #Making a 2D grid which stores all the object corrosponding to that node. 
    matrix = []
    for i in range(len(grid)):
        x = []
        for j in range(len(grid[0])):
            #Saving all the nodes by running into for loop
            object = Node(i,j,grid[i][j], None)
            #Initialising distance of all the nodes as infinity(-1).
            object.g = -1
            x.append(object)
        matrix.append(x)

    #Making an Empty queue
    queue = []
    
    #Assisning S variable as the start of the function
    s = start
    
    #Making Distance of the start node as 0. 
    # matrix[s[0]][s[1]].parent = None
    u= Node(s[0], s[1], False, None)
    u.parent = None
    u.g = 0

    #Inserting s in queue
    node_visited.add(tuple(s))
    
    #Running a loop untill it reaches goal
    while not found:
        
        # Sorting the queue according to the distance 
        queue.sort(key=lambda fun: fun[1])
        
        #Making an Empty list
        n_list = []
        
        #Evaluating the neighbours
        neighbours = [[0, +1], [+1, 0], [0, -1], [-1, 0]]
        
        for m, n in neighbours:
            if u.row + m in range(len(grid)) and u.col + n in range(len(grid[0])) and grid[u.row + m][u.col + n ] == 0:
                n_list.append((u.row + m,u.col + n ))
            n_list.reverse()
            
        #Destination reached (Base case)
        if (u.row, u.col) == tuple(goal):
            found = True
            break
        
        else:
        #Running over all the neighbours
            for v in n_list:
                #Checking is the neighbour is not obsevered/visited.
                if v not in node_visited:
                    #Adding new node 
                    h = abs((v[0] - goal[0]) + abs(v[1] - goal[1]))
                    h = math.sqrt((v[0] - goal[0]) * (v[0] - goal[0]) + (v[1] - goal[1]) * (v[1] - goal[1]))
                    newnode= Node(v[0], v[1], False, h)
                    
                    #Calculating the updating distance 
                    alt = newnode.g + u.g
                    
                    #Updating the distance and parent of the node
                    newnode.g = u.g + 1
                    newnode.parent = u
                    #updating distance 
                    dis = newnode.g + newnode.h
                    
                    #Apeending to the queue
                    queue.append((newnode, dis))
            
            #Poping u from the queue
            u = (queue.pop(0))[0]
            #Calcuating the visited nodes
            node_visited.add((u.row, u.col))
    
    #Path generation 
    while u.parent is not None:
        #Appending the path 
        path.append([u.row, u.col])
        #Retracting the parents
        u = u.parent
        
    #Adding Start position to the to the path     
    path.append(start)
    #Reversing the path 
    path.reverse()


    #Calculation of the steps + 1 for the start node
    steps = len(node_visited)-1


    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()




