import numpy as np 
import copy
import os
class Node:
    def __init__(self, id:int, x:float, y:float, demand:float, ready_time:float, due_time:float, service_time:float):
        super()
        
        """
            Initialize instance variables
        """
        self.id = id  # unique identifier of the node
        
        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False
            
        self.x= x  # x-coordinate of the node
        self.y = y  # y-coordinate of the node
        self.demand = demand  # quantity of goods that must be transported to/from the node
        self.ready_time = ready_time  # earliest time at which the node can be serviced
        self.due_time = due_time  # latest time by which the node must be serviced
        self.service_time = service_time  # time required to service the node

class VrptwGraph:
    def __init__(self, file_path, xi=0.1):
        
        # The pheromone evaporates rate
        self.xi = xi
        
        # node_num Number of nodes
        # node_dist_mat distance between nodes (matrix)
        # tau_ij Degree of information concentration on the path between nodes

        self.node_num, self.nodes, self.node_dist_mat, self.vehicle_num, self.vehicle_capacity = self.create_instance_from_file(file_path)
        
        self.nnh_travel_path, self.init_pheromone_val, _ = self.nearest_neighbor_heuristic()
        
        self.init_pheromone_val = 1/(self.init_pheromone_val * self.node_num) 

        # Create pheromone matrix
        self.pheromone_mat = np.full((self.node_num, self.node_num), self.init_pheromone_val)


        self.heuristic_info_mat = 1 / self.node_dist_mat

        
    def copy(self, tau_0):
        
        """
            Define a method to create a copy 
            of the current object and initialize some of its attributes
        """
        
        new_graph = copy.deepcopy(self)

        new_graph.tau_0 = tau_0
        new_graph.tau_ij = np.ones((new_graph.node_num, new_graph.node_num)) * tau_0
    
    
    def create_instance_from_file(self, file_path: str) -> int | list[Node] | np.ndarray | int | int :
        
        """
            Create instances from file
        """
        
        # Initialize an empty list to store information about each node
        node_list = []
        # Open the file at file_path for reading
        with open(file_path, 'rt') as f:
            count = 1
            # Iterate over each line in the file
            for line in f:
                # If this is the fifth line, extract the vehicle number and capacity
                if count == 5:
                    vehicle_num, vehicle_capacity = line.split()
                    vehicle_num = int(vehicle_num)
                    vehicle_capacity = int(vehicle_capacity)
                # If this is the tenth line or later, extract information about a node
                elif count >= 10:
                    node_list.append(line.split())
                # Increment the line counter
                count += 1
        # Calculate the total number of nodes in the problem instance
        node_num = len(node_list)        
        # Create a list of Node objects using the information in node_list
        nodes = [Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]), float(item[6])) for item in node_list]
        # Create a 2D numpy array to store the distances between each pair of nodes
        node_dist_mat = np.zeros((node_num, node_num))
        # Iterate over each node in the problem instance
        for i in range(node_num):
            # Get the ith node
            node_a = nodes[i]
            # Set the distance between a node and itself to a very large number to avoid issues with zero distances
            node_dist_mat[i][i] = 1e-8
            # Iterate over the remaining nodes
            for j in range(i+1, node_num):
                # Get the jth node
                node_b = nodes[j]
                # Calculate the distance between node_a and node_b using the VrptwGraph class's calculate_dist method
                node_dist_mat[i][j] = VrptwGraph.calculate_dist(node_a, node_b)
                # Set the distance between node_b and node_a to the same value to ensure symmetry
                node_dist_mat[j][i] = node_dist_mat[i][j]
    
        return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity
    
    @staticmethod
    def calculate_dist(node_a, node_b) -> float :
        
        """
            Calcolate euclidiean distance between node a and b
            
            return : eclidiean distance
        """
        
        return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))

    def local_update_pheromone(self, start_ind, end_ind):
        """
        Updates the pheromone level on the path between two nodes locally. This is typically called after an ant
        moves from one node to another. The update rule uses an evaporation factor to reduce the pheromone level,
        while adding a small amount of new pheromone.

        :param start_ind: The starting node index of the path segment.
        :param end_ind: The ending node index of the path segment.
        """
        # The existing pheromone on the path is reduced by a factor of (1 - xi)
        # and a new pheromone value, initial pheromone value (init_pheromone_val), is added.
        self.pheromone_mat[start_ind][end_ind] = (1 - self.xi) * self.pheromone_mat[start_ind][end_ind] + \
                                                self.xi * self.init_pheromone_val

    def global_update_pheromone(self, best_path, best_path_distance):
        """
        Performs a global update on the pheromone matrix using the best path found in the current iteration.
        The update reduces the pheromone level on all paths by the evaporation factor and increases the pheromone
        level on the paths used in the best path. The amount of pheromone added is inversely proportional to the
        distance of the best path, promoting shorter paths.

        :param best_path: The sequence of nodes representing the best path found.
        :param best_path_distance: The total distance of the best path.
        """
        # Evaporate pheromone on all paths
        self.pheromone_mat = (1 - self.xi) * self.pheromone_mat

        # Increase pheromone on paths used in the best path
        current_ind = best_path[0]
        for next_ind in best_path[1:]:
            # The pheromone increase is inversely proportional to the best path distance
            self.pheromone_mat[current_ind][next_ind] += self.xi / best_path_distance
            current_ind = next_ind

    
    def nearest_neighbor_heuristic(self, max_vehicle_num=None) -> list[int] | float | int:
        
        """
            This function takes a set of nodes with their associated properties
            and returns a solution using the nearest neighbor heuristic algorithm.
            
            return: travel_path, travel_distance, vehicle_num
        """
        
        # Initialize variables
        index_to_visit = list(range(1, self.node_num)) # list of indices of unvisited nodes
        current_index = 0 # starting from the depot
        current_load = 0 # current load on the vehicle
        current_time = 0 # current time of the vehicle
        travel_distance = 0 # total distance traveled by the vehicle(s)
        travel_path = [0] # path of nodes visited by the vehicle(s), starting from the depot
        
        if max_vehicle_num is None: 
            max_vehicle_num = self.node_num # if max_vehicle_num is not specified, set it to the number of nodes
        
        # While there are nodes to visit and there are still available vehicles
        while len(index_to_visit) > 0 and max_vehicle_num > 0:
            # Find the nearest unvisited node from the current node based on distance, load, and time constraints
            nearest_next_index = self._cal_nearest_next_index(index_to_visit, current_index, current_load, current_time)

            # If there is no unvisited node available, return to the depot
            if nearest_next_index is None:
                travel_distance = travel_distance + self.node_dist_mat[current_index][0]

                current_load = 0
                current_time = 0
                travel_path.append(0)
                current_index = 0

                max_vehicle_num = max_vehicle_num - 1 
            
            else:
                
                # Update current load and time based on visiting the next node
                current_load = current_load + self.nodes[nearest_next_index].demand

                dist = self.node_dist_mat[current_index][nearest_next_index]
                wait_time = max(self.nodes[nearest_next_index].ready_time - current_time - dist, 0)
                service_time = self.nodes[nearest_next_index].service_time

                current_time = current_index + dist + wait_time + service_time
                index_to_visit.remove(nearest_next_index)

                # Update travel distance and path
                travel_distance = travel_distance + self.node_dist_mat[current_index][nearest_next_index]
                travel_path.append(nearest_next_index)
                current_index = nearest_next_index
        
        # Go back to the depot to end the path
        travel_distance = travel_distance + self.node_dist_mat[current_index][0]
        travel_path.append(0)

        # Count the number of vehicles used in the path
        vehicle_num = travel_path.count(0)-1
        
        return travel_path, travel_distance, vehicle_num
    
    def _cal_nearest_next_index(self, index_to_visit:int, current_index:int, current_load:float, current_time:float) -> int:
        
        """
            Calculates the nearest next index that can be visited
            based on the current location of a vehicle and the load it is transport.
            
            return: the index of the nearest node that can be visited
        """
        
        nearest_ind = None  # initialize variable to keep track of the nearest node index
        nearest_distance = None  # initialize variable to keep track of the distance to the nearest node

        # loop through each node index in the given list of indices to visit
        for next_index in index_to_visit:
            # check if the vehicle can carry the load required for this node
            if current_load + self.nodes[next_index].demand > self.vehicle_capacity:
                continue  

            # calculate the distance to the node, the wait time before servicing the node,
            # and the time required to service the node
            dist = self.node_dist_mat[current_index][next_index]
            wait_time = max(self.nodes[next_index].ready_time - current_time - dist, 0)
            service_time = self.nodes[next_index].service_time

            # check if the vehicle can return to the depot before the depot's due time
            if current_time + dist + wait_time + service_time + self.node_dist_mat[next_index][0] > self.nodes[0].due_time:
                continue 

            # check if the node due time has already passed
            if current_time + dist > self.nodes[next_index].due_time:
                continue  
            
            # if the distance to this node is shorter than the distance to the current nearest node,
            # update the nearest node index and distance variables
            if nearest_distance is None or self.node_dist_mat[current_index][next_index] < nearest_distance:
                nearest_distance = self.node_dist_mat[current_index][next_index]
                nearest_ind = next_index

       
        return nearest_ind


