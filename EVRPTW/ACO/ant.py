import numpy as np 
import copy 
from evrptw_config import EvrptwGraph

class Ant:
    def __init__(self, graph: EvrptwGraph, start_index=0):
        """
        Initialize the ant with reference to the graph it will traverse, starting index, and initial state variables.
        
        Parameters:
        - graph (EvrptwGraph): The graph representing the EVRPTW problem.
        - start_index (int): The index of the graph where the ant starts, default is 0 (the depot).
        """
        
        self.graph = graph
        self.current_index = start_index
        self.vehicle_load = 0
        self.vehicle_travel_time = 0
        self.travel_path = [start_index]  # Initialize the travel path with the starting index.
        self.arrival_time = [0]  # Initialize the arrival times with the starting time 0.
        self.total_travel_distance = 0
        self.fuel_level = self.graph.tank_capacity  # Initialize the fuel level to the tank's capacity.
        
        # Initialize the list of indices the ant has yet to visit.
        self.index_to_visit = list(range(graph.node_num))
        self.index_to_visit.remove(start_index)  # Remove the starting index as it's the current location.

        # Initialize the dynamic heuristic matrix for ACS-DIST with zeros.
        self.eta_k_ij_mat = np.zeros((self.graph.node_num, self.graph.node_num))
        for i in range(self.graph.node_num):
            for j in range(self.graph.node_num):
                self.eta_k_ij_mat[i][j] = self.acs_dist_heuristic(i, j)

    def acs_dist_heuristic(self, i, j):
        """
        Calculate the dynamic heuristic for the ACS-DIST algorithm based on current travel time,
        the distance between nodes, and the time windows of the nodes.

        Parameters:
        - i (int): The index of the current node.
        - j (int): The index of the next node.

        Returns:
        - eta_k_ij (float): The heuristic value for moving from node i to node j.
        """

        ctk = self.vehicle_travel_time  # Accumulated travel time of ant k
        distance_ij = self.graph.node_dist_mat[i][j]  # Distance from customer i to customer j
        t_ij = distance_ij / self.graph.velocity  # Convert distance to time based on average speed

        e_j = self.graph.nodes[j].ready_time  # Earliest time window of customer j

        # Calculate the delivery time of ant k when traveling to customer j
        dt_j = max(ctk + t_ij, e_j)

        eta_k_ij = 1 / max(1, (dt_j - ctk) * (e_j - ctk))

        return eta_k_ij


    def move_to_next_index(self, next_index):
        """
        Attempts to move to the specified next index (node) if energy and other constraints are met.
        
        Args:
            next_index (int): The index of the next node to move to.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if self.has_enough_energy(self.current_index, next_index):
            # Update travel state and move next index
            self.update_travel_state(next_index)
            self.current_index = next_index
            return True
        else:
            # Find nearest station
            nearest_station_index, dist_to_station = self.graph.select_closest_station(self.current_index, next_index)
            if nearest_station_index is not None and self.fuel_level >= dist_to_station * self.graph.fuel_consumption_rate:
                self.update_travel_state(nearest_station_index)
                self.current_index = nearest_station_index
                # Reload batteru
                self.fuel_level += self.graph.charging_rate
                # After reload check next index
                if self.has_enough_energy(self.current_index, next_index):
                    # Update travel state 
                    self.update_travel_state(next_index)
                    self.current_index = next_index
                    return True
                else:
                    # Even after recharging, the next index cannot be reached.
                    return False
            else:
                # Neither the next index nor a charging station can be reached.
                return False

    def update_travel_state(self, next_index):
        """
        Updates the travel state of the vehicle when moving to a new index.

        Args:
            next_index (int): The index of the next node to move to.
        """
        # Compute distance to next index and update total travel distance
        distance_to_next_index = self.graph.node_dist_mat[self.current_index][next_index]
        self.total_travel_distance += distance_to_next_index

        # Compute travel time to next index
        vehicle_speed = self.graph.velocity
        travel_time_to_next_index = distance_to_next_index / vehicle_speed
        self.vehicle_travel_time += travel_time_to_next_index  # Update accumulated travel time
        # Update fuel level
        self.fuel_level -= distance_to_next_index * self.graph.fuel_consumption_rate

        # Add next index to travel path and record arrival time
        self.travel_path.append(next_index)
        self.arrival_time.append(self.vehicle_travel_time)

        ctk = self.vehicle_travel_time  # Accumulated travel time of ant k

        t_ij = distance_to_next_index / self.graph.velocity  # Convert distance to time based on average speed

        e_j = self.graph.nodes[next_index].ready_time  # Earliest time window of customer j


        # Calculate the delivery time of ant k when traveling to customer j
        dt_j = max(ctk + t_ij, e_j)
        eta_k_ij = 1 / max(1, (dt_j - ctk) * (e_j - ctk))

        # Update the dynamic heuristic matrix for all nodes with respect to the next index.
        for i in range(self.graph.node_num):
            if i != next_index:  # Exclude self-loop
                self.eta_k_ij_mat[i][next_index] = eta_k_ij
                self.eta_k_ij_mat[next_index][i] = eta_k_ij

        # Manages the different types of nodes: depot, customer or charging station.
        if self.graph.nodes[next_index].is_depot():
            self.fuel_level = self.graph.tank_capacity
            self.vehicle_load = 0
            self.vehicle_travel_time = 0

        elif self.graph.nodes[next_index].is_customer():
            potential_load = self.vehicle_load + self.graph.nodes[next_index].demand
            if potential_load <= self.graph.load_capacity:
                self.vehicle_load = potential_load
                wait_time = max(0, self.graph.nodes[next_index].ready_time - self.vehicle_travel_time)
                self.vehicle_travel_time += wait_time + self.graph.nodes[next_index].service_time
                self.index_to_visit.remove(next_index)
            else:
                self.return_to_depot()

        elif self.graph.nodes[next_index].is_station():
            self.fuel_level += self.graph.charging_rate


    def has_enough_energy(self, current_index, next_index):
        """
        Checks if there is enough energy to move from the current index to the next index.

        Args:
            current_index (int): The current index of the ant.
            next_index (int): The index of the next node to move to.

        Returns:
            bool: True if there is enough energy, False otherwise.
        """
        # If the current index is a charging station, recharge to maximum and don't check further.
        if self.graph.nodes[current_index].is_station():
            self.fuel_level += self.graph.charging_rate  # Assume refueling brings the tank to full.
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
                self.fuel_level -= dist_to_station 
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

        # 4. Time window constraint.
        arrival_time_at_next = self.calculate_arrival_time(next_index)
        if next_node.is_customer():
            if not self.check_time_window(arrival_time_at_next, next_node):
                return False  # Time window constraint not met.

        # 5. Feasibility of returning to the depot.
        if not self.can_return_to_depot_from(next_index, arrival_time_at_next):
            return False  # Can't return to depot in time from this node.

        return True  # All constraints met.

    def calculate_arrival_time(self, next_index):
        """
        Calculates the arrival time at the next index.

        Args:
            next_index (int): The index of the next node.

        Returns:
            float: Estimated arrival time at the next index.
        """
        dist_to_next = self.graph.node_dist_mat[self.current_index][next_index]
        return self.vehicle_travel_time + dist_to_next / self.graph.velocity

    def check_time_window(self, arrival_time, next_node):
        """
        Checks if the service at the next node can be started within its time window.

        Args:
            arrival_time (float): The arrival time at the next node.
            next_node (Node): The next node object.

        Returns:
            bool: True if the service can be started and completed within the time window, False otherwise.
        """
        
        # Wait if necessary and check if service can be started within the time window.
        wait_time = max(next_node.ready_time - arrival_time, 0)
        service_start_time = arrival_time + wait_time
        service_end_time = service_start_time + next_node.service_time

        # Calculate overrun
        time_violation = max(service_end_time - next_node.due_date, 0)

        return service_end_time <= next_node.due_date, time_violation

    def can_return_to_depot_from(self, next_index, current_time):
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

    def check_capacity_constraint(self, next_index):
        """
        Checks if the vehicle's load capacity is not exceeded when visiting the next node.

        Args:
            next_index (int): The index of the next node.

        Returns:
            bool: True if the capacity constraint is not violated, False otherwise.
        """
        next_node = self.graph.nodes[next_index]
        if next_node.is_customer():
            new_load = self.vehicle_load + next_node.demand
            if new_load > self.graph.load_capacity:
                return False  # Capacity constraint violated.
        return True  # No capacity constraint violation.

    def cal_next_index_meet_constrains(self):
        """
        Calculates indices of all nodes that meet constraints and are yet to be visited.

        Returns:
            list: A list of indices that meet all constraints.
        """
        next_index_meet_constrains = []
        for next_ind in self.index_to_visit:
            if self.check_condition(next_ind):
                next_index_meet_constrains.append(next_ind)
        return next_index_meet_constrains

    def cal_nearest_next_index(self, next_index_list):
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

    def calculate_feasible_neighbors(self):
        """
        Calculate feasible neighbors of the ant, excluding charging stations and the depot,
        and return a list of their indices.
        """
        feasible_neighbors = []
        for index in self.index_to_visit:
            node = self.graph.nodes[index]
            # Skip charging stations and the depot
            if not node.is_station() and not node.is_depot():
                # Check if moving to this node meets all the constraints
                if self.check_condition(index):
                    feasible_neighbors.append(index)
        return feasible_neighbors

    @staticmethod
    def cal_total_travel_distance(graph: EvrptwGraph, travel_path:list) -> float:
        
        """
            Calculate the total travel distance given a travel path and the VRPTW graph.

            Return: distance (float): the total travel distance of the given travel path.
        """

        # Initialize the total travel distance and the current index with the first index in the travel path
        distance = 0
        current_ind = travel_path[0]

        for next_ind in travel_path[1:]:            
            # Calculate the distance between the current index and the next index
            distance = distance + graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind

        return distance


    @staticmethod
    def cross_exchange(graph: EvrptwGraph, travel_path: list, travel_distance: float):
        half = len(travel_path) // 2
        segment_1 = travel_path[:half]
        segment_2 = travel_path[half:]

        segment_1 = Ant.remove_consecutive_zeros(segment_1)
        segment_2 = Ant.remove_consecutive_zeros(segment_2)
        # Verifica che entrambi i segmenti contengano almeno un elemento
        if not segment_1 or not segment_2:
            #print("Errore: uno dei segmenti è vuoto.")
            return None, None

        # Ensure both segments start and end with the depot's index
        if segment_1[0] != graph.depot.idx:
            segment_1.insert(0, graph.depot.idx)
        if segment_1[-1] != graph.depot.idx:
            segment_1.append(graph.depot.idx)
        if segment_2[0] != graph.depot.idx:
            segment_2.insert(0, graph.depot.idx)
        if segment_2[-1] != graph.depot.idx:
            segment_2.append(graph.depot.idx)

        # Extract the path up to the first customer in segment_1
        path_to_first_customer_first_segment = []
        for node in segment_1:
            path_to_first_customer_first_segment.append(node)
            if graph.nodes[node].is_customer():
                break

        # Reverse segment_1 to find the path to the last customer
        segment_1.reverse()
        path_to_last_customer_first_segment = []
        for node in segment_1:
            path_to_last_customer_first_segment.append(node)
            if graph.nodes[node].is_customer():
                break

        segment_1.reverse()  # Reverse back to original order
        # Nuovo segmento dopo il primo cliente in segment_1
        new_segment_after_first_customer_1 = []
        index_after_first_customer_1 = len(path_to_first_customer_first_segment)

        # Verifica la presenza di un altro cliente subito dopo il primo cliente in segment_1
        if index_after_first_customer_1 < len(segment_1):
            for node in segment_1[index_after_first_customer_1:]:
                new_segment_after_first_customer_1.append(node)
                if graph.nodes[node].is_customer():
                    if node not in path_to_last_customer_first_segment:
                        break
                    else:
                        new_segment_after_first_customer_1 = []
                        break

        segment_1.reverse()
        # Nuovo segmento dopo l'ultimo cliente in segment_1
        new_segment_after_last_customer_1 = []
        index_after_last_customer_1 = len(path_to_last_customer_first_segment)

        # Verifica la presenza di un altro cliente subito dopo l'ultimo cliente in segment_1
        if index_after_last_customer_1 < len(segment_1):
            for node in segment_1[index_after_last_customer_1:]:
                    new_segment_after_last_customer_1.append(node)
                    if graph.nodes[node].is_customer():
                        if node not in path_to_first_customer_first_segment and node not in new_segment_after_first_customer_1:
                            break
                        else:
                            new_segment_after_last_customer_1 = []
                            break

        segment_1.reverse()

        # Crea un segmento aggiuntivo per includere gli indici rimanenti di segment_1
        additional_segment_1 = []

        # Trova gli indici per il segmento aggiuntivo
        if new_segment_after_first_customer_1 and new_segment_after_last_customer_1:
            end_index_of_first = segment_1.index(new_segment_after_first_customer_1[-1])
            start_index_of_last = segment_1.index(new_segment_after_last_customer_1[-1])
            for i in range(end_index_of_first + 1, start_index_of_last):
                additional_segment_1.append(segment_1[i])
        else:
            additional_segment_1 = []

        # Extract the path up to the first customer in segment_2
        path_to_first_customer_second_segment = []
        for node in segment_2:
            path_to_first_customer_second_segment.append(node)
            if graph.nodes[node].is_customer():
                break

        # Reverse segment_2 to find the path to the last customer
        segment_2.reverse()
        path_to_last_customer_second_segment = []
        for node in segment_2:
            path_to_last_customer_second_segment.append(node)
            if graph.nodes[node].is_customer():
                break

        segment_2.reverse()  # Reverse back to original order
        
        # New segment after costumer first costumer segment_2
        new_segment_after_first_customer_2 = []
        index_after_first_customer_2 = len(path_to_first_customer_second_segment)
        if index_after_first_customer_2 < len(segment_2):
            for node in segment_2[index_after_first_customer_2:]:
                new_segment_after_first_customer_2.append(node)
                if graph.nodes[node].is_customer():
                    if node not in path_to_last_customer_second_segment:
                        break
                    else:
                        new_segment_after_first_customer_2 = []
                        break

        segment_2.reverse()
        
        # New segment after last client in segment_2
        new_segment_after_last_customer_2 = []
        index_after_last_customer_2 = len(path_to_last_customer_second_segment)
        if index_after_last_customer_2 < len(segment_2):
            for node in segment_2[index_after_last_customer_2:]:
                new_segment_after_last_customer_2.append(node)
                if graph.nodes[node].is_customer():
                    if node not in path_to_first_customer_second_segment and node not in new_segment_after_first_customer_2:
                        break
                    else:
                        new_segment_after_last_customer_2 = []
                        break

        additional_segment_2 = []

        # Find indexes for the additional segment
        if new_segment_after_first_customer_2 and new_segment_after_last_customer_2:
            end_index_of_first = segment_2.index(new_segment_after_first_customer_2[-1])
            start_index_of_last = segment_2.index(new_segment_after_last_customer_2[-1])
            for i in range(end_index_of_first + 1, start_index_of_last):
                additional_segment_2.append(segment_2[i])
        else:
            additional_segment_2 = []

        segment_2.reverse()

        # Management of empty segments for the first solution
        if new_segment_after_first_customer_2 == [] or new_segment_after_last_customer_2 == [] or new_segment_after_last_customer_1 == [] or new_segment_after_first_customer_1 == []:
            # Or-opt: If all segments after the first and last customer are empty
            solution1 = (path_to_first_customer_first_segment + additional_segment_1 + path_to_last_customer_first_segment)
            solution2 = (path_to_first_customer_second_segment + additional_segment_2 + path_to_last_customer_second_segment)
        else:
            # 2-opt*: If at least one of the segments after the first and last client is not empty
            solution1 = (path_to_first_customer_first_segment + new_segment_after_first_customer_2 + additional_segment_2 + new_segment_after_last_customer_2 + path_to_last_customer_first_segment)
            solution2 = (path_to_first_customer_second_segment + new_segment_after_first_customer_1 + additional_segment_1 + new_segment_after_last_customer_1 + path_to_last_customer_second_segment)

        # Add storage at the end of solutions if necessary
        if solution1[-1] != graph.depot.idx:
            solution1.append(graph.depot.idx)
        if solution2[-1] != graph.depot.idx:
            solution2.append(graph.depot.idx)


        final_solution = solution1 + solution2

        cost_final_solution = Ant.cal_total_travel_distance(graph, final_solution)

        if cost_final_solution < travel_distance:
            final_cost = cost_final_solution
        else:
            final_cost = travel_distance

        return final_solution, final_cost

    def is_new_path_feasible(self, graph: EvrptwGraph, new_path: list):
        """
        Checks if a new path is feasible considering capacity constraints.

        Args:
            graph (EvrptwGraph): The graph representing the problem space.
            new_path (list): The path to be evaluated.

        Returns:
            bool: True if the path is feasible under capacity constraints, False otherwise.
        """
        ant = Ant(graph) 
        for next_index in new_path:
            # Check each node in the new path for capacity constraint violations
            if not ant.check_capacity_constraint(next_index):
                return False  # Path is not feasible due to capacity constraint violation

        return True  # Path is feasible under capacity constraints

    @staticmethod
    def remove_consecutive_zeros(segment):
        """
        Removes consecutive zeros from a given list (segment). Only the first zero in a 
        series of consecutive zeros is kept, and the rest are removed.

        Args:
        segment (list): The list from which consecutive zeros need to be removed.

        Returns:
        list: A new list with consecutive zeros removed.
        """

        new_segment = []  # Initialize an empty list to store the processed segment

        # Loop through each element in the original segment
        for i in range(len(segment)):
            # Append the first element, any non-zero element, or a zero that doesn't follow another zero
            if i == 0 or segment[i] != 0 or (segment[i] == 0 and segment[i - 1] != 0):
                new_segment.append(segment[i])

        return new_segment  

    def check_energy_new_path(self, path):
        """
        Checks and repairs a path for energy constraints by inserting charging stations as needed.

        Args:
            path (list): The path to be evaluated and potentially modified.

        Returns:
            bool: True if the path is feasible under energy constraints after modification, False otherwise.
        """
        i = 0
        while i < len(path) - 1:
            # Check each segment of the path for energy constraints
            if not self.has_enough_energy(path[i], path[i + 1]):
                # If energy constraint is violated, find the nearest charging station and insert it into the path
                charging_station_idx, _ = self.graph.select_closest_station(path[i], path[i + 1])
                if charging_station_idx is not None:
                    if path[i+1] == charging_station_idx:
                        alternative_station_id = self.find_alternative_charging_station(path, path[i], charging_station_idx)
                        path.insert(i+1, alternative_station_id)
                    else:
                        path.insert(i + 1, charging_station_idx)
                        i += 1  
                else:
                    return False  

            i += 1 

        return True  

    def find_alternative_charging_station(self, path, current_index, excluded_station_idx):
        """
        Finds an alternative charging station that is reachable and different from the given excluded station.
        If no such station is found, returns the index of the depot as a last resort.

        Args:
        - path (list): The current path of the vehicle.
        - current_index (int): The current index in the path.
        - excluded_station_idx (int): The index of the charging station to be excluded.

        Returns:
        - int: The index of the alternative charging station, or the depot index if no such station is found.
        """
        if current_index < 0 or current_index >= len(path):
            # If the current_index is out of bounds, handle the error or return the depot index
            return self.graph.depot.idx

        for station_index, node in enumerate(self.graph.nodes):
            if node.is_station() and station_index != excluded_station_idx:
                # Ensure that the next station is not the same as the current one
                if current_index + 1 < len(path) and station_index == path[current_index + 1]:
                    continue
                if self.has_enough_energy(path[current_index], station_index):
                    return station_index

        # If no alternative station is found, return the depot index as a last resort
        return self.graph.depot.idx


    def local_search_procedure(self):
        """
        Executes a local search procedure to optimize the current travel path.
        The method iteratively applies cross-exchange operations and accepts new paths
        if they are feasible and satisfy energy constraints. The search terminates 
        either after a fixed number of improvements or when an infeasible or 
        non-improving solution is encountered.

        The travel path of the ant is updated with the best found path.
        """
       
        new_path = copy.deepcopy(self.travel_path)
        new_path_distance = Ant.cal_total_travel_distance(self.graph, self.travel_path)
        
        improvement_count = 0
        max_no_improvement = 100 

        # Continue the search until the maximum number of improvements is reached
        while improvement_count < max_no_improvement:
            # Apply cross-exchange operation to find a new path
            temp_path, temp_distance = Ant.cross_exchange(self.graph, new_path, new_path_distance)

            # Proceed if a new path is found
            if temp_path is not None:
                if self.is_new_path_feasible(self.graph, new_path):
                    if self.check_energy_new_path(new_path):
                        # Accept the new path and update the travel distance
                        new_path = temp_path
                        new_path_distance = temp_distance
                        improvement_count += 1   
                    else:
                        # Break the loop if energy constraints are not met
                        break
                else:
                    # Break the loop if the new path is not feasible
                    break
            else:
                # Break the loop if no new path is found
                break

        # Update the ant's travel path and total travel distance with the optimized path
        self.travel_path = new_path
        self.total_travel_distance = new_path_distance