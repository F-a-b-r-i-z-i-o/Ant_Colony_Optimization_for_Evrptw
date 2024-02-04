from target import Node
import numpy as np
import re


class EvrptwGraph:
    def __init__(self, file_path, rho=0.1):
        """
        Initializes an EVRPTW graph from a given instance file.

        Parameters:
            file_path (str): Path to the EVRPTW instance file containing node and vehicle data.
            rho (float, optional): Pheromone evaporation rate for the ACO algorithm. Defaults to 0.1.

        The constructor performs the following initializations:
        - Reads data from the instance file to set up nodes, vehicle properties, and other parameters.
        - Calculates initial pheromone levels and sets up a pheromone matrix.
        - Implements the nearest neighbor heuristic to generate a preliminary solution.
        """
        self.rho = rho
        (
            self.node_num,
            self.nodes,
            self.node_dist_mat,
            self.fuel_stations,
            self.depot,
            self.tank_capacity,
            self.load_capacity,
            self.fuel_consumption_rate,
            self.charging_rate,
            self.velocity,
        ) = self._read_instance(file_path)

        self.nnh_travel_path, self.Cnn, self.vehicles = (
            self.nearest_neighbor_heuristic()
        )

        self.tau_0 = 1 / (self.node_num * self.Cnn)

        # Create pheromone matrix
        self.pheromone_mat = np.full((self.node_num, self.node_num), self.tau_0)

    def global_update_pheromone(
        self, best_path: list, best_path_distance: float
    ) -> None:
        """
        Global update pheromone for the best-so-far ant.

        Parameters:
        - best_path (list): The best path found so far.
        - best_path_distance (float): The total distance of the best path.
        """

        # Only evaporate pheromone on arcs that are part of the best path
        for i in range(len(best_path) - 1):
            current_ind = best_path[i]
            next_ind = best_path[i + 1]
            self.pheromone_mat[current_ind][next_ind] *= 1 - self.rho
            self.pheromone_mat[current_ind][next_ind] += self.rho / best_path_distance

    @staticmethod
    def calculate_distance_matrix(nodes: list) -> np.ndarray:
        """
        Calculate the Euclidean distance matrix for a list of nodes.

        Parameters:
        - nodes (list): A list of Node objects.

        Returns:
        - numpy.ndarray: A distance matrix where element [i, j] represents
                        the distance between nodes i and j.
        """
        coordinates = np.array([(node.x, node.y) for node in nodes])
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff**2, axis=2))
        dist_mat[dist_mat == 0] = 1e-9  # Avoid division by zero for distances of 0
        return dist_mat

    @staticmethod
    def _read_instance(file_path: str) -> tuple:
        """
        Reads the EVRPTW instance from the given file and extracts necessary information.

        Parameters:
        - file_path (str): Path to the instance file.

        Returns:
        - tuple: Contains the number of nodes, list of nodes, distance matrix,
                list of fuel stations, depot, tank capacity, load capacity,
                fuel consumption rate, charging rate, and vehicle velocity.
        """

        with open(file_path, "rt") as f:
            lines = f.readlines()
            nodes = []
            fuel_stations = []
            depot = None
            tank_capacity = 0.0
            load_capacity = 0.0
            fuel_consumption_rate = 0.0
            charging_rate = 0.0
            velocity = 0.0

            # Iterating over the lines in the file to extract information
            for line in lines[1:]:
                stl = line.split()

                # Extracting node information
                if len(stl) == 8:
                    idx = int(stl[0][1:])
                    new_node = Node(
                        idx,
                        stl[0],
                        stl[1],
                        float(stl[2]),
                        float(stl[3]),
                        float(stl[4]),
                        float(stl[5]),
                        float(stl[6]),
                        float(stl[7]),
                    )
                    nodes.append(new_node)

                    if stl[1] == "d":
                        depot = new_node
                    elif stl[1] == "f":
                        fuel_stations.append(new_node)

                # Extracting vehicle related information
                elif "Vehicle fuel tank capacity" in line:
                    tank_capacity = float(re.search(r"\d+\.\d+", line).group())
                elif "Vehicle load capacity" in line:
                    load_capacity = float(re.search(r"\d+\.\d+", line).group())
                elif "fuel consumption rate" in line:
                    fuel_consumption_rate = float(re.search(r"\d+\.\d+", line).group())
                elif "inverse refueling rate" in line:
                    charging_rate = float(re.search(r"\d+\.\d+", line).group())
                elif "average Velocity" in line:
                    velocity = float(re.search(r"\d+\.\d+", line).group())

        # Calculating the distance matrix for the nodes
        node_num = len(nodes)

        # Use calculate_distance_matrix to create the distance matrix
        node_dist_mat = EvrptwGraph.calculate_distance_matrix(nodes)

        return (
            node_num,
            nodes,
            node_dist_mat,
            fuel_stations,
            depot,
            tank_capacity,
            load_capacity,
            fuel_consumption_rate,
            charging_rate,
            velocity,
        )

    def select_closest_station(self, i: int, j: int) -> (int, float):
        """
        Select the charging station that is closest in terms of total distance to the nodes i and j.

        Parameters:
        - i (int): Index of the first node.
        - j (int): Index of the second node.

        Returns:
        - tuple: A tuple containing:
            - int: Index of the selected charging station. If no station is found, returns -1.
            - float: The total distance to the station. Returns None if no station is found.
        """

        charging_station_indices = [
            idx for idx, node in enumerate(self.nodes) if node.is_station()
        ]
        min_total_distance = float("inf")
        selected_station_idx = None

        for station_idx in charging_station_indices:
            distance_to_i = self.node_dist_mat[i][station_idx]
            distance_to_j = self.node_dist_mat[station_idx][j]
            total_distance = distance_to_i + distance_to_j

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                selected_station_idx = station_idx

        if selected_station_idx is not None:
            return selected_station_idx, min_total_distance
        else:
            return -1, None

    def get_coordinates_from_path(self, path: list) -> list:
        """
        Returns the (x, y) coordinates for each node in the given path.

        Parameters:
        - path (list): A list of node indices representing the path.

        Returns:
        - list: A list of tuples, each containing the (x, y) coordinates of a node in the path.
        """
        coordinates = []
        for node_idx in path:
            node = self.nodes[node_idx]
            coordinates.append((node.x, node.y))
        return coordinates

    def create_node_type_map(self) -> dict:
        """
        Creates a map that associates each node's index with its type.

        Returns:
        dict: A map associating each node's index (int) with its type (str).
        """
        return {node.idx: node.node_type for node in self.nodes}

    def nearest_neighbor_heuristic(self, max_vehicle_num: int = None) -> tuple:
        """
        Heuristic to generate a preliminary travel path using the nearest neighbor approach.

        Parameters:
        - max_vehicle_num (int, optional): The maximum number of vehicles to consider.

        Returns:
        tuple: A tuple containing the travel path (list), the total travel distance (float),
               and the number of vehicles used (int).
        """
        index_to_visit = list(
            range(1, self.node_num)
        )  # Initialize with all nodes except depot (0)
        current_index = 0
        current_load = 0
        current_time = 0
        travel_distance = 0
        travel_path = [0]
        current_battery = self.tank_capacity

        if max_vehicle_num is None:
            max_vehicle_num = self.node_num

        while len(index_to_visit) > 0 and max_vehicle_num > 0:

            if (
                self.nodes[current_index].is_station()
                and current_battery <= self.tank_capacity
            ):
                # Vehicle is currently at a charging station but hasn't fully charged
                current_battery += self.charging_rate
            else:
                if current_battery < self.tank_capacity:
                    nearest_station_index, _ = self.select_closest_station(
                        current_index, 0
                    )
                    if (
                        nearest_station_index is not None
                        and nearest_station_index != current_index
                    ):
                        # Go to the nearest charging station
                        distance_to_station = self.node_dist_mat[current_index][
                            nearest_station_index
                        ]
                        travel_distance += distance_to_station
                        travel_path.append(nearest_station_index)

                        # Add travel time to station
                        current_time += distance_to_station / self.velocity

                        # Complete battery recharge
                        current_battery += self.charging_rate
                        current_index = nearest_station_index
                        continue

            nearest_next_index = self._cal_nearest_next_index(
                index_to_visit, current_index, current_battery, current_time
            )
            if nearest_next_index is not None:
                distance_to_next = self.node_dist_mat[current_index][nearest_next_index]
                battery_usage = distance_to_next * self.fuel_consumption_rate

                if current_battery >= battery_usage:
                    if self.nodes[nearest_next_index].is_customer():
                        demand = self.nodes[nearest_next_index].demand
                        if current_load <= self.load_capacity:
                            # Update state to serve the node
                            current_load += demand
                            current_battery -= battery_usage
                            travel_distance += distance_to_next
                            travel_path.append(nearest_next_index)
                            current_index = nearest_next_index
                            index_to_visit.remove(nearest_next_index)
                        else:
                            # Can't serve this customer due to capacity or time window
                            continue
                    else:
                        # For non-customer nodes, just move
                        current_battery -= battery_usage
                        travel_distance += distance_to_next
                        travel_path.append(nearest_next_index)
                        current_time += distance_to_next / self.velocity
                        current_index = nearest_next_index
                        index_to_visit.remove(nearest_next_index)

                else:
                    # Not enough battery to reach next node, find and move to the nearest charging station
                    nearest_station_index, _ = self.select_closest_station(
                        current_index, 0
                    )
                    if (
                        nearest_station_index is not None
                        and nearest_station_index != current_index
                    ):
                        # Go to the nearest charging station
                        distance_to_station = self.node_dist_mat[current_index][
                            nearest_station_index
                        ]
                        travel_distance += distance_to_station
                        travel_path.append(nearest_station_index)

                        # Add travel time to station and adjust battery
                        current_time += distance_to_station / self.velocity
                        current_battery = (
                            self.tank_capacity
                        )  # Assuming immediate full recharge for simplicity
                        current_index = nearest_station_index
                        continue
                    else:
                        # If no charging station is available, return to depot
                        distance_to_depot = self.node_dist_mat[current_index][0]
                        travel_distance += distance_to_depot
                        current_index = 0
                        current_battery = self.tank_capacity  # Recharge at depot
                        current_time = 0
                        current_load = 0
                        max_vehicle_num -= 1
            else:
                distance_to_depot = self.node_dist_mat[current_index][0]
                travel_distance += distance_to_depot
                current_battery -= distance_to_depot * self.fuel_consumption_rate
                current_index = 0
                current_battery = self.tank_capacity  # Recharge at depot
                current_time = 0
                current_load = 0
                # Return to depot and start with a new vehicle
                max_vehicle_num -= 1

        travel_path.append(0)
        vehicle_num = travel_path.count(0) - 1
        # print(travel_path, travel_distance)
        return travel_path, travel_distance, vehicle_num

    def _cal_nearest_next_index(
        self,
        index_to_visit: list,
        current_index: int,
        current_battery: float,
        current_time: float,
    ) -> int:
        """
        Calculates the nearest next index that the vehicle can visit from the current index, considering battery and time constraints.

        Args:
        - index_to_visit (list): List of node indices that are yet to be visited.
        - current_index (int): The current node index.
        - current_battery (float): The current battery level.
        - current_time (float): The current time.

        Returns:
        int: The index of the nearest next node that can be visited. Returns None if no such node is found.
        """

        nearest_ind = None
        nearest_distance = float("inf")

        for next_index in index_to_visit:
            dist = self.node_dist_mat[current_index][next_index]
            battery_usage = dist * self.fuel_consumption_rate

            # Check if there is enough battery to reach the next node.
            if current_battery >= battery_usage:
                if self.nodes[next_index].is_customer():
                    ready_time = self.nodes[next_index].ready_time
                    due_date = self.nodes[next_index].due_date
                    wait_time = max(ready_time - current_time - dist, 0)
                    service_time = self.nodes[next_index].service_time
                    current_time += dist + wait_time + service_time

                    # Check whether the vehicle can serve the customer within its time window.
                    if service_time >= due_date:
                        continue

                # Calculate the distance back to the nearest charging station or to the depot after visiting the node.
                nearest_station_index, _ = self.select_closest_station(next_index, 0)
                dist_to_nearest_station = self.node_dist_mat[next_index][
                    nearest_station_index
                ]
                total_required_battery = (
                    battery_usage + dist_to_nearest_station * self.fuel_consumption_rate
                )

                if current_battery >= total_required_battery:
                    if dist < nearest_distance:
                        nearest_distance = dist
                        nearest_ind = next_index

        return nearest_ind
