import numpy as np
import copy
from vrptw_config import VrptwGraph
from threading import Event



class Ant:
    def __init__(self, graph: VrptwGraph, start_index=0):
        """
        Initialize an ant used in ACO for solving the VRPTW.
        
        :param graph: Instance of the VrptwGraph representing the problem space.
        :param start_index: The starting node index for the ant, typically the depot.
        """
        self.graph = graph  # The problem graph
        self.current_index = start_index  # Current node index
        self.vehicle_load = 0  # Current load of the vehicle
        self.vehicle_travel_time = 0  # Current travel time of the vehicle
        self.travel_path = [start_index]  # Path taken by the vehicle
        self.arrival_time = [0]  # Arrival times at each node

        # List of node indices that are yet to be visited
        self.index_to_visit = list(range(graph.node_num))
        self.index_to_visit.remove(start_index)  # Remove the start node

        self.total_travel_distance = 0  # Total travel distance of the ant

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
        - average_speed (float): The average speed used to convert distance to time.
        
        Returns:
        - eta_k_ij (float): The heuristic value for moving from node i to node j.
        """

        average_speed = 1.0
        
        ctk = self.vehicle_travel_time  # Accumulated travel time of ant k
        distance_ij = self.graph.node_dist_mat[i][j]  # Distance from customer i to customer j
        t_ij = distance_ij / average_speed  # Convert distance to time based on average speed

        e_j = self.graph.nodes[j].ready_time  # Earliest time window of customer j

        # Calculate the delivery time of ant k when traveling to customer j
        dt_j = max(ctk + t_ij, e_j)
        
        eta_k_ij = 1 / max(1, (dt_j - ctk) * (e_j - ctk))

        return eta_k_ij

    def clear(self):
        """
        Clears the current path and nodes to visit. This is typically used before starting a new search iteration.
        """
        self.travel_path.clear()
        self.index_to_visit.clear()

    def move_to_next_index(self, next_index):
        """
        Move the ant to the next node index and update the path, load, and travel time.
        
        :param next_index: The next node index the ant will move to.
        """
        # Update the ant's path and total travel distance
        self.travel_path.append(next_index)
        self.total_travel_distance += self.graph.node_dist_mat[self.current_index][next_index]

        # Calculate distance to next index and update arrival time
        dist = self.graph.node_dist_mat[self.current_index][next_index]
        self.arrival_time.append(self.vehicle_travel_time + dist)

        if self.graph.nodes[next_index].is_depot:
            # If the next node is the depot, reset the vehicle load and travel time
            self.vehicle_load = 0
            self.vehicle_travel_time = 0
        else:
            # Update vehicle load, travel time, and remove the node from the list of nodes to visit
            self.vehicle_load += self.graph.nodes[next_index].demand
            self.vehicle_travel_time += dist + \
                max(self.graph.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0) + \
                self.graph.nodes[next_index].service_time
            self.index_to_visit.remove(next_index)

        self.current_index = next_index  # Update the current node index

    def index_to_visit_empty(self):
        """
        Check if there are no more nodes left to visit.
        
        :return: Boolean indicating if the list of nodes to visit is empty.
        """
        return len(self.index_to_visit) == 0

    def get_active_vehicles_num(self):
        """
        Calculate the number of active vehicles based on the number of times the depot is revisited.
        
        :return: The number of active vehicles.
        """
        return self.travel_path.count(0) - 1

    def check_condition(self, next_index) -> bool:
        """
        Check if moving to the next node satisfies the problem constraints such as load and time windows.
        
        :param next_index: The next node index to check.
        :return: Boolean indicating if the move is feasible.
        """
        # Check if the vehicle load exceeds capacity after taking the next node's demand
        if self.vehicle_load + self.graph.nodes[next_index].demand > self.graph.vehicle_capacity:
            return False

        # Calculate the distance to the next node and waiting time if early
        dist = self.graph.node_dist_mat[self.current_index][next_index]
        wait_time = max(self.graph.nodes[next_index].ready_time - self.vehicle_travel_time - dist, 0)
        service_time = self.graph.nodes[next_index].service_time

        # Check if the vehicle can return to the depot before due time after servicing the next node
        if self.vehicle_travel_time + dist + wait_time + service_time + \
           self.graph.node_dist_mat[next_index][0] > self.graph.nodes[0].due_time:
            return False

        # Check if the vehicle arrives at the next node before its due time
        if self.vehicle_travel_time + dist > self.graph.nodes[next_index].due_time:
            return False

        return True

    def cal_next_index_meet_constrains(self):
        """
        Finds all reachable customers from the current position that meet the constraints.
        
        :return: A list of indices that meet the travel constraints.
        """
        next_index_meet_constrains = []
        # Loop through all nodes that have not been visited yet
        for next_ind in self.index_to_visit:
            # Check if the next node meets the load and time window constraints
            if self.check_condition(next_ind):
                next_index_meet_constrains.append(next_ind)
        return next_index_meet_constrains

    def cal_nearest_next_index(self, next_index_list):
        """
        From the list of possible next customers, select the one closest to the current position.
        
        :param next_index_list: List of indices of customers to be considered.
        :return: The index of the nearest customer.
        """
        current_ind = self.current_index
        nearest_ind = next_index_list[0]
        min_dist = self.graph.node_dist_mat[current_ind][nearest_ind]

        # Find the nearest customer by comparing distances
        for next_ind in next_index_list[1:]:
            dist = self.graph.node_dist_mat[current_ind][next_ind]
            if dist < min_dist:
                min_dist = dist
                nearest_ind = next_ind

        return nearest_ind

    @staticmethod
    def cal_total_travel_distance(graph: VrptwGraph, travel_path):
        """
        Calculate the total travel distance for a given path.
        
        :param graph: The graph instance representing the VRPTW.
        :param travel_path: The path for which the total distance is calculated.
        :return: The total distance of the travel path.
        """
        distance = 0
        current_ind = travel_path[0]
        # Sum the distances between consecutive nodes in the path
        for next_ind in travel_path[1:]:
            distance += graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind
        return distance


    def try_insert_on_path(self, node_id, stop_event: Event):
        """
        Attempts to insert a node (node_id) into the current path (travel_path) of the ant.
        The insertion must not violate the constraints of load, time, and travel distance.
        If multiple insertion points are available, the optimal one is chosen based on the shortest resulting path.
        
        :param node_id: The ID of the node to try to insert.
        :param stop_event: An Event object that, if set, will stop the procedure (used in threading).
        :return: The index at which the node is best inserted, or None if no valid insertion point is found.
        """
        best_insert_index = None
        best_distance = None

        for insert_index in range(len(self.travel_path)):

            if stop_event.is_set():
                # print('[try_insert_on_path]: receive stop event')
                return

            if self.graph.nodes[self.travel_path[insert_index]].is_depot:
                continue

            # Find the nearest depot preceding insert_index
            front_depot_index = insert_index
            while front_depot_index >= 0 and not self.graph.nodes[self.travel_path[front_depot_index]].is_depot:
                front_depot_index -= 1
            front_depot_index = max(front_depot_index, 0)

            # check_ant from front_depot_index
            check_ant = Ant(self.graph, self.travel_path[front_depot_index])

            for i in range(front_depot_index+1, insert_index):
                check_ant.move_to_next_index(self.travel_path[i])
                
            if check_ant.check_condition(node_id):
                check_ant.move_to_next_index(node_id)
            else:
                continue

            for next_ind in self.travel_path[insert_index:]:
                if stop_event.is_set():
                    # print('[try_insert_on_path]: receive stop event')
                    return

                if check_ant.check_condition(next_ind):
                    check_ant.move_to_next_index(next_ind)
                    if self.graph.nodes[next_ind].is_depot:
                        temp_front_index = self.travel_path[insert_index-1]
                        temp_back_index = self.travel_path[insert_index]

                        check_ant_distance = self.total_travel_distance - self.graph.node_dist_mat[temp_front_index][temp_back_index] + \
                                             self.graph.node_dist_mat[temp_front_index][node_id] + self.graph.node_dist_mat[node_id][temp_back_index]

                        if best_distance is None or check_ant_distance < best_distance:
                            best_distance = check_ant_distance
                            best_insert_index = insert_index
                        break
                else:
                    break

        return best_insert_index

    def insertion_procedure(self, stop_even: Event):
        """
        For each unvisited node, the method attempts to find a suitable insertion point into the ant's current path.
        It ensures that the insertion doesn't violate the constraints of load, time, and distance.
        This process is repeated until no more nodes can be successfully inserted.
        
        :param stop_even: An Event object to signal the process to stop (used in threading).
        """
        if self.index_to_visit_empty():
            return

        success_to_insert = True
        while success_to_insert:
            success_to_insert = False
            ind_to_visit = np.array(copy.deepcopy(self.index_to_visit))
            demand = np.zeros(len(ind_to_visit))
            for i, ind in zip(range(len(ind_to_visit)), ind_to_visit):
                demand[i] = self.graph.nodes[ind].demand

            arg_ind = np.argsort(demand)[::-1]
            ind_to_visit = ind_to_visit[arg_ind]

            for node_id in ind_to_visit:
                if stop_even.is_set():
                    # print('[insertion_procedure]: receive stop event')
                    return

                best_insert_index = self.try_insert_on_path(node_id, stop_even)
                if best_insert_index is not None:
                    self.travel_path.insert(best_insert_index, node_id)
                    self.index_to_visit.remove(node_id)
                    # print('[insertion_procedure]: success to insert %d(node id) in %d(index)' % (node_id, best_insert_index))
                    success_to_insert = True

            del demand
            del ind_to_visit
        if self.index_to_visit_empty():
            print('[insertion_procedure]: success in insertion')

        self.total_travel_distance = Ant.cal_total_travel_distance(self.graph, self.travel_path)

    @staticmethod
    def local_search_once(graph: VrptwGraph, travel_path: list, travel_distance: float, i_start, stop_event: Event):
        """
        Performs one iteration of local search on the given path. It attempts to improve the path by exchanging segments of the route.
        If a better path is found (shorter distance), it is returned along with the new total distance and the start index for the next iteration.
        
        :param graph: The VRPTW graph instance.
        :param travel_path: The current travel path of the ant.
        :param travel_distance: The total distance of the current travel path.
        :param i_start: The start index for the current local search iteration.
        :param stop_event: An Event object that, if set, terminates the search early.
        :return: A tuple of the new path, its total distance, and the start index for the next iteration; or None if no improvement is found.
        """

      
        depot_ind = []
        for ind in range(len(travel_path)):
            if graph.nodes[travel_path[ind]].is_depot:
                depot_ind.append(ind)

        for i in range(i_start, len(depot_ind)):
            for j in range(i + 1, len(depot_ind)):

                if stop_event.is_set():
                    return None, None, None

                for start_a in range(depot_ind[i - 1] + 1, depot_ind[i]):
                    for end_a in range(start_a, min(depot_ind[i], start_a + 6)):
                        for start_b in range(depot_ind[j - 1] + 1, depot_ind[j]):
                            for end_b in range(start_b, min(depot_ind[j], start_b + 6)):
                                if start_a == end_a and start_b == end_b:
                                    continue
                                new_path = []
                                new_path.extend(travel_path[:start_a])
                                new_path.extend(travel_path[start_b:end_b + 1])
                                new_path.extend(travel_path[end_a:start_b])
                                new_path.extend(travel_path[start_a:end_a])
                                new_path.extend(travel_path[end_b + 1:])

                                depot_before_start_a = depot_ind[i - 1]

                                depot_before_start_b = depot_ind[j - 1] + (end_b - start_b) - (end_a - start_a) + 1
                                if not graph.nodes[new_path[depot_before_start_b]].is_depot:
                                    raise RuntimeError('error')

                    
                                success_route_a = False
                                check_ant = Ant(graph, new_path[depot_before_start_a])
                                for ind in new_path[depot_before_start_a + 1:]:
                                    if check_ant.check_condition(ind):
                                        check_ant.move_to_next_index(ind)
                                        if graph.nodes[ind].is_depot:
                                            success_route_a = True
                                            break
                                    else:
                                        break

                                check_ant.clear()
                                del check_ant

                                success_route_b = False
                                check_ant = Ant(graph, new_path[depot_before_start_b])
                                for ind in new_path[depot_before_start_b + 1:]:
                                    if check_ant.check_condition(ind):
                                        check_ant.move_to_next_index(ind)
                                        if graph.nodes[ind].is_depot:
                                            success_route_b = True
                                            break
                                    else:
                                        break
                                check_ant.clear()
                                del check_ant

                                if success_route_a and success_route_b:
                                    new_path_distance = Ant.cal_total_travel_distance(graph, new_path)
                                    if new_path_distance < travel_distance:
                                       
                   
                                        for temp_ind in range(1, len(new_path)):
                                            if graph.nodes[new_path[temp_ind]].is_depot and graph.nodes[new_path[temp_ind - 1]].is_depot:
                                                new_path.pop(temp_ind)
                                                break
                                        return new_path, new_path_distance, i
                                else:
                                    new_path.clear()

        return None, None, None

    def local_search_procedure(self, stop_event: Event):
        """
        Conducts a local search on the ant's travel path which has visited all nodes in the graph.
        The search uses a 'cross' method to try and improve the path.
        The local search continues for a predefined number of iterations or until no further improvements can be found.
        
        :param stop_event: An Event object to signal the process to stop (used in threading).
        """
        new_path = copy.deepcopy(self.travel_path)
        new_path_distance = self.total_travel_distance
        times = 100
        count = 0
        i_start = 1
        while count < times:
            temp_path, temp_distance, temp_i = Ant.local_search_once(self.graph, new_path, new_path_distance, i_start, stop_event)
            if temp_path is not None:
                count += 1

                del new_path, new_path_distance
                new_path = temp_path
                new_path_distance = temp_distance

                
                i_start = (i_start + 1) % (new_path.count(0)-1)
                i_start = max(i_start, 1)
            else:
                break

        self.travel_path = new_path
        self.total_travel_distance = new_path_distance
        #print('[local_search_procedure]: local search finished')