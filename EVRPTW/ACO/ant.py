import numpy as np 
from evrptw_config import EvrptwGraph

class Ant:
    def __init__(self, graph: EvrptwGraph, start_index=0):
        """
        Initializes an ant agent for solving the Electric Vehicle Routing Problem with Time Windows (EVRPTW).

        Parameters:
            graph (EvrptwGraph): The EVRPTW graph on which the ant operates.
            start_index (int, optional): The starting node index for the ant, typically the depot. Defaults to 0.

        The ant is initialized with a starting node, and various state variables are set, including its current load, 
        travel time, travel path, arrival times, total travel distance, and fuel level. The ant's path is initially 
        set to include only the start node, and it's prepared to visit all other nodes.
        """
        
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.vehicle_travel_time = 0
        self.travel_path = [start_index] 
        self.arrival_time = [0] 
        self.total_travel_distance = 0
        self.fuel_level = self.graph.tank_capacity 
        self.index_to_visit = list(range(graph.node_num))
        self.index_to_visit.remove(start_index) 

        # Initialize the dynamic heuristic matrix for ACS-DIST with zeros.
        self.eta_k_ij_mat = np.zeros((self.graph.node_num, self.graph.node_num))


    def update_eta_matrix(self) -> (None):
        """
        Updates the eta matrix, which is used for calculating heuristic values for routing decisions.

        This method iterates over all pairs of nodes (excluding self-loops) in the graph and calculates
        the heuristic value based on travel time and penalties for arriving too early or too late
        relative to a node's time windows.
        """
        for i in range(self.graph.node_num):
            for j in range(self.graph.node_num):
                if i != j:  # Exclude self-loops
                    distance_to_j = self.graph.node_dist_mat[i][j]
                    # Calculate travel time to node j
                    travel_time_to_j = distance_to_j / self.graph.velocity  # t_ij (the travel time of arc i,j)

                    # Calculate penalty based on time window constraints
                    e_j = self.graph.nodes[j].ready_time  # Start of time window
                    l_j = self.graph.nodes[j].due_date    # End of time window
                    arrival_time_at_j = self.vehicle_travel_time + travel_time_to_j # ct^k is the accumulated travel time of ant k, t_ij

                    # Determine penalty for arriving too early or too late
                    penalty = 0
                    if arrival_time_at_j < e_j:
                        penalty = e_j - arrival_time_at_j
                    elif arrival_time_at_j > l_j:
                        penalty = arrival_time_at_j - l_j
                    elif e_j <= arrival_time_at_j <= l_j:
                        penalty = 0

            
                    self.eta_k_ij_mat[i][j] = 1 / (travel_time_to_j + penalty)

    def move_to_next_index(self, next_index: int)-> (bool):
        """
        Attempts to move to the specified next index (node) if energy and other constraints are met.
        
        Args:
            next_index (int): The index of the next node to move to.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        # Check whether all conditions are met for moving to the next point
        if not self.check_condition(next_index):
            return False  # If any condition fails, cannot move to the next index

        # If all conditions are satisfied, proceed to update travel state and move to the next index
        self.update_travel_state(next_index)
        self.update_eta_matrix()
        self.current_index = next_index
        return True

    def update_travel_state(self, next_index: int) -> (int): 
        """
        Updates the travel state of the vehicle when moving to a new index.

        This method calculates the distance and travel time to the next node, updates the vehicle's
        travel state, and then handles the visit to the corresponding node type (depot, customer, or station).

        Args:
            next_index (int): The index of the next node to move to.
        """
        # Calculate the distance and travel time to the next index
        distance_to_next_index = self.graph.node_dist_mat[self.current_index][next_index]
        travel_time_to_next_index = distance_to_next_index / self.graph.velocity

        self.total_travel_distance += distance_to_next_index
        self.vehicle_travel_time += travel_time_to_next_index
        self.fuel_level = max(0, self.fuel_level - distance_to_next_index * self.graph.fuel_consumption_rate)
        self.travel_path.append(next_index)
        self.arrival_time.append(self.vehicle_travel_time)

        # Handle the different types of nodes
        current_node = self.graph.nodes[next_index]
        if current_node.is_depot():
            self.handle_depot_visit()
        elif current_node.is_customer():
            self.handle_customer_visit(next_index)
        elif current_node.is_station():
            self.handle_station_visit()

    def handle_depot_visit(self) -> (None):
        """ Handles the visit to the depot. """
        self.fuel_level = self.graph.tank_capacity
        self.vehicle_load = 0
        self.vehicle_travel_time = 0

    def handle_customer_visit(self, customer_index:int) -> (bool):
        """ Handles the vist to the customer """
        customer_node = self.graph.nodes[customer_index]
        demand = customer_node.demand

        if self.vehicle_load + demand > self.graph.load_capacity:
            # Capacity exceeded, cannot service the customer
            return False

        # Calculate the arrival time at the customer
        arrival_time_at_customer = self.calculate_arrival_time(customer_index)

        # Check if the arrival time is within the customer's time window
        within_time_window, _ = self.check_time_window(arrival_time_at_customer, customer_node)

        if not within_time_window:
            return False

        # If within time window, proceed to service the customer
        self.vehicle_load += demand
        self.vehicle_travel_time = arrival_time_at_customer + customer_node.service_time
        self.travel_path.append(customer_index)
        self.index_to_visit.remove(customer_index)

        return True

    def handle_station_visit(self) -> (None):
        """ Handles the visit to the charging station. """

        # Recharge the vehicle's fuel, but not exceeding the tank's capacity
        if self.fuel_level <= self.graph.tank_capacity:
            self.fuel_level += self.graph.charging_rate
        else:
            self.fuel_level = self.graph.tank_capacity

    def has_enough_energy(self, current_index: int, next_index: int) -> (bool):
        """
        Checks if there is enough energy to move from the current index to the next index.

        Args:
            current_index (int): The current index of the ant.
            next_index (int): The index of the next node to move to.

        Returns:
            bool: True if there is enough energy, False otherwise.
        """
        if self.graph.nodes[current_index].is_station() and self.fuel_level < self.graph.tank_capacity:
            self.fuel_level = min(self.fuel_level + self.graph.charging_rate, self.graph.tank_capacity)
            return True
        
        # Calculate the fuel required to move to the next index.
        dist_to_next = self.graph.node_dist_mat[current_index][next_index]
        fuel_required_to_next = dist_to_next * self.graph.fuel_consumption_rate

        # Check if there is enough fuel to move directly to the next index.
        if self.fuel_level >= fuel_required_to_next:
            self.fuel_level -= fuel_required_to_next 
            return True

        # If not enough fuel to go directly, check for the nearest charging station on the path.
        nearest_charge_station, dist_to_station = self.graph.select_closest_station(current_index, next_index)
        if nearest_charge_station is not None:
            if self.fuel_level >= dist_to_station * self.graph.fuel_consumption_rate:
                self.fuel_level -= dist_to_station * self.graph.fuel_consumption_rate
                return nearest_charge_station

        return False  # Not enough fuel to move to the next index or a charging station.

    def check_condition(self, next_index: int) -> bool:
        """
        Checks whether moving to the next point satisfies all constraints.

        Args:
            next_index (int): The index of the next node to consider.

        Returns:
            bool: True if all constraints are met for moving to the next index, False otherwise.
        """
        next_node = self.graph.nodes[next_index]

        # 1. Single visit constraint, applicable only to customers.
        if next_node.is_customer() and next_index in self.travel_path:
            return False  # Already visited, can't visit again.

        # 2. Capacity constraint.
        if not self.check_capacity_constraint(next_index):
            return False  # Capacity constraint not met.

        # 3. Energy (fuel) constraint.
        if not self.has_enough_energy(self.current_index, next_index):
            return False  # Not enough energy to move to the next index.
        
        #4. Time window constraint.
        arrival_time_at_next = self.calculate_arrival_time(next_index)
        if next_node.is_customer():
            time_window_ok, _ = self.check_time_window(arrival_time_at_next, next_node)
            if not time_window_ok:
                return False  # Time window constraint not met.
    
        # 5. Feasibility of returning to the depot.
        if not self.can_return_to_depot_from(next_index, arrival_time_at_next):
            return False  # Can't return to depot in time from this node.

        return True  # All constraints met.

    def calculate_arrival_time(self, next_index:int) -> (float):
        """
        Calculates the arrival time at the next index.

        Args:
            next_index (int): The index of the next node.

        Returns:
            float: Estimated arrival time at the next index.
        """
        dist_to_next = self.graph.node_dist_mat[self.current_index][next_index]
        return self.vehicle_travel_time + dist_to_next / self.graph.velocity

    def check_time_window(self, arrival_time: float, next_node: int) -> tuple[bool, float]:
        """
        Checks if the service at the next node can be started within its time window.

        Args:
            arrival_time (float): The arrival time at the next node.
            next_node (Node): The next node object.

        Returns:
            tuple:
                - bool: True if the service can be started and completed within the time window, False otherwise.
                - float: The amount of time by which the service end time exceeds the node's due date. 
                        If the service is within the time window, this value is 0.
        """
        
        # Wait if necessary and check if service can be started within the time window.
        wait_time = max(next_node.ready_time - arrival_time, 0)
        service_start_time = arrival_time + wait_time
        service_end_time = service_start_time + next_node.service_time

        # Calculate overrun
        time_violation = max(service_end_time - next_node.due_date, 0)

        return service_end_time <= next_node.due_date, time_violation

    def can_return_to_depot_from(self, next_index: int, current_time: float) -> bool: 
        """
        Checks if the vehicle can return to the depot from the next index within the depot's due time.

        Args:
            next_index (int): The index of the next node.
            current_time (float): The current time.

        Returns:
            bool: True if the vehicle can return to the depot in time, False otherwise.
        """
        dist_to_depot = self.graph.node_dist_mat[next_index][0]
        time_to_depot = dist_to_depot / self.graph.velocity
        return current_time + time_to_depot <= self.graph.nodes[0].due_date

    def return_to_depot(self):
        """
        Handles the logic when the ant needs to return to the depot.
        """
        self.total_travel_distance += self.graph.node_dist_mat[self.current_index][0]  # Distance to depot
        self.vehicle_travel_time += self.graph.node_dist_mat[self.current_index][0] / self.graph.velocity  # Travel time to depot
        self.fuel_level = self.graph.tank_capacity  # Refuel at depot
        self.vehicle_load = 0  # Unload any cargo
        self.travel_path.append(0)  # Add depot to the travel path
        self.arrival_time.append(self.vehicle_travel_time)  # Log arrival time at depot
        self.current_index = 0  # Set current position to depot

    def check_capacity_constraint(self, next_index: int) -> bool:
        """
        Checks if the vehicle's load capacity is not exceeded when visiting the next node.

        Args:
            next_index (int): The index of the next node.

        Returns:
            bool: True if the capacity constraint is not violated, False otherwise.
        """
        if next_index < 0 or next_index >= len(self.graph.nodes):
            return False  
        
        next_node = self.graph.nodes[next_index]

        if next_node.is_customer():
            new_load = self.vehicle_load + next_node.demand
            if new_load > self.graph.load_capacity:
                return False  # Capacity constraint violated.
        return True  # No capacity constraint violation.


    def cal_nearest_next_index(self, next_index_list: list) -> int: 
        """
        Calculates the nearest index from the current position among a list of indices.

        Args:
            next_index_list (list): A list of indices to consider.

        Returns:
            int: The index of the nearest node.
        """
        current_ind = self.current_index
        nearest_ind = next_index_list[0]
        min_dist = self.graph.node_dist_mat[current_ind][next_index_list[0]]
        for next_ind in next_index_list[1:]:
            dist = self.graph.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind
        return nearest_ind

    def calculate_feasible_neighbors(self) -> list: 
        """
        Calculate feasible neighbors of the ant, excluding charging stations and the depot,
        and return a list of their indices.

        This function evaluates each node in the graph, except for the depot and charging stations, 
        to determine if it can be visited next by the ant. A node is considered feasible if it has 
        not been visited yet and meets all the routing constraints such as energy requirements, 
        load capacity, and time windows.

        Returns:
            list: A list of indices, each corresponding to a feasible node that the ant can visit next. 
                These indices refer to nodes that are neither charging stations nor the depot and 
                meet all necessary constraints for the next visit.
        """
        feasible_neighbors = []
        for index in self.index_to_visit:
            if 0 <= index < len(self.graph.nodes):
                node = self.graph.nodes[index]
                # Skip charging stations and the depot
                if not node.is_station() and not node.is_depot():
                    # Check if moving to this node meets all the constraints
                    if self.check_condition(index):
                        feasible_neighbors.append(index)
        return feasible_neighbors

    @staticmethod
    def cal_total_travel_distance(graph: EvrptwGraph, travel_path: list) -> float:
        """
        Calculate the total travel distance given a travel path and the VRPTW graph.

        Return: distance (float): the total travel distance of the given travel path.
        """

        # Verify empty path if empty return -1 
        if not travel_path:
            return -1

        # Inizialize total distance and firtst index of path 
        distance = 0
        current_ind = travel_path[0]

        for next_ind in travel_path[1:]:
            # Calcolate distance between current index and next index
            distance += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind

        return distance
    
    def is_feasible(self, path: list, ant: 'Ant') -> bool:
        """
        Verifies if a given path is feasible for the EVRPTW.

        Args:
        - path (list): The path to be checked.
        - ant (Ant): The ant object with methods to check constraints.

        Returns:
        - bool: True if the path is feasible, False otherwise.
        """
        for i in range(len(path) - 1):
            ant.current_index = path[i]
            next_index = path[i + 1]

            if not ant.check_condition(next_index):
                return False  # If any constraint is violated, the path is not feasible

        # Check if the path correctly ends at the depot
        if not path[-1] == ant.graph.depot.idx:
            return False

        return True  # The path is feasible

    def local_search_2opt(self, graph: EvrptwGraph, best_path: list) -> tuple[list, float]:
        """
        Performs a local search optimization on the given path using the swap method.

        This function iterates over pairs of non-depot nodes in the path and attempts to swap them 
        to find a more optimal route. The goal is to minimize the total travel distance. After each swap, 
        it checks the feasibility of the new path. If the new path is feasible and has a shorter distance 
        than the current best path, it updates the best path.

        Args:
            graph: The EVRPTW graph that the path traverses.
            best_path (list): The current best path to be optimized.

        Returns:
            tuple:
                - list: The optimized path that potentially has a shorter distance than the input path.
                - float: The total travel distance of the optimized path.
        """
        best_distance = Ant.cal_total_travel_distance(graph, best_path)
        ant = Ant(graph)

        for i in range(1, len(best_path) - 2):
            for j in range(i + 1, len(best_path) - 1):
                if best_path[i] == 0 or best_path[j] == 0:  # Skip if depot
                    continue

                # Swap elements to create a new path
                new_path = best_path.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]

                # Recalculate the distance for the specific segment of the path that was affected by the swap
                affected_distance = Ant.cal_total_travel_distance(graph, new_path[i - 1:j + 2])

                # Determine if the new segment distance is better than the old one
                old_segment_distance = Ant.cal_total_travel_distance(graph, best_path[i - 1:j + 2])
                if affected_distance < old_segment_distance:
                    # Check the feasibility of the entire new path
                    if self.is_feasible(new_path, ant):
                        # If feasible, update the best path and its corresponding distance
                        best_path = new_path
                        best_distance = Ant.cal_total_travel_distance(graph, best_path)
                        print("\n\nImproved path found during local search.")

        return best_path, best_distance


