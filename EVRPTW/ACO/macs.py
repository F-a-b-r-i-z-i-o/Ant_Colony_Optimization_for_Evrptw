import numpy as np
import random
from evrptw_config import EvrptwGraph
from ant import Ant
import time

class MultipleAntColonySystem:
    """
    Implements a Multiple Ant Colony System for solving the Electric Vehicle Routing Problem
    with Time Windows (EVRPTW).

    Attributes:
        graph (EvrptwGraph): The graph representing the EVRPTW problem.
        ants_num (int): Number of ants in the system.
        max_iter (int): Maximum number of iterations.
        alpha (float): Parameter influencing the importance of pheromone.
        beta (float): Parameter influencing the importance of heuristic information.
        q0 (float): Parameter for pseudo-random proportional decision rule.
        k1 (float): Coefficient for time window penalty.
        k2 (float): Coefficient for the number of vehicles penalty.
    """

    def __init__(
            self, graph: EvrptwGraph, ants_num=10, max_iter=400, alpha=1, beta=2, q0=0.9, k1=2, k2=3
    ):
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.beta = beta
        self.alpha = alpha
        self.q0 = q0
        self.k1 = k1
        self.k2 = k2


    def macs(self):
        C_test = float("inf")
        T_best = list(range(self.graph.node_num))
        m = (len(self.graph.nodes))

        start_time = time.time()  # Record the starting time
        max_execution_time = 7200
        c_test_history = []
        time_history = []

        while time.time() - start_time < max_execution_time:
            while True:
                T_dist, C_dist = self._ACS_DIST(T_best, m)


                if C_test > C_dist:
                    C_test = C_dist
                    c_test_history.append(C_test)
                    time_history.append(time.time() - start_time)

                print(f"Distance: {C_test} -- Elapsed Time: {time.time() - start_time:.2f} seconds")

                break

        print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
        return C_test, c_test_history, time_history

    def _construct_solution(self, sigma: int):
        """
        Constructs a solution (path) for a given number of vehicles (sigma) based on the
        ant colony system and various constraints, including energy limits.

        Returns:
        - T_k (list): The constructed travel path.
        """

        T_k = [self.graph.depot.idx]  # Inizialize path from depot
        vehicles = 1
        visited_customers = set()
        all_customers = set(i for i, node in enumerate(self.graph.nodes) if node.is_customer())
        ants = [Ant(self.graph) for _ in range(self.ants_num)]
        cumulative_time_window_penalty = 0
        penalty_rate = 1.0

        while vehicles <= sigma and visited_customers != all_customers:
            i = T_k[-1]  # Current Node
            ant = self.roulette_wheel_selection(ants)  # Select ant
            feasible_nodes = set(ant.calculate_feasible_neighbors()) - visited_customers
            remaining_customers = all_customers - feasible_nodes - visited_customers

            if feasible_nodes:
                j = self.select_next_index(ant, feasible_nodes)
            elif remaining_customers:
                j = self.select_next_index(ant, remaining_customers)
            else:
                # If there are no more clients to visit, return to the depot.
                j = self.graph.depot.idx
                vehicles += 1  # Start a new route with another vehicle.

            if j != self.graph.depot.idx:
                arrival_time_at_j = ant.calculate_arrival_time(j)
                next_node = self.graph.nodes[j]

                within_time_window, time_violation = ant.check_time_window(arrival_time_at_j, next_node)

                if not within_time_window:
                    # Calculates the penalty for exceeding the time window.
                    penalty = time_violation * penalty_rate
                    cumulative_time_window_penalty += penalty

                # Checks whether a charging station is needed to reach the next node j
                if not ant.has_enough_energy(i, j):
                    s, _ = self.graph.select_closest_station(i, j)
                    if T_k[-1] != s:
                        T_k.append(s)
                        ant.move_to_next_index(s)
                        self.graph.local_update_pheromone(i, s)
                else:
                    T_k.append(j)
                    visited_customers.add(j)
                    ant.move_to_next_index(j)
                    self.graph.local_update_pheromone(i, j)

        if T_k[-1] != self.graph.depot.idx:  # Check depot entry at the end
            i = T_k[-1]
            j = self.graph.depot.idx
            if not ant.has_enough_energy(i, j):
                # If not, select the closest charging station from the current node (i) to the depot (j)
                s, _ = self.graph.select_closest_station(i, j)
                if T_k[-1] != s:
                    # If the last node in the path is not the charging station, append it
                    T_k.append(s)
                    ant.move_to_next_index(s)
                T_k.append(j)
                ant.move_to_next_index(j)

            else:
                T_k.append(j)
                ant.move_to_next_index(j)
        return T_k, cumulative_time_window_penalty


    def evaluate_solution(self, travel_path: list):
        """
        Evaluate the total distance of a given travel path.

        Returns:
        - float: The total distance of the provided travel path.
        """

        distance = 0
        current_ind = travel_path[0]
        for next_ind in travel_path[1:]:
            distance += self.graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind

        # Add the distance from the last customer back to the depot (assuming the starting index represents the depot).
        distance += self.graph.node_dist_mat[current_ind][travel_path[0]]

        return distance

    def _ACS_DIST(self, T_best: list, m: int):
        """
        Implements the Ant Colony System (ACS) algorithm to find an optimal solution for dist
        based on pheromone updating rules and heuristic information.

        Returns:
        - tuple: The best path (list of nodes) and its associated cost.
        """

        sigma = m
        T_dist = T_best
        C_dist = self.evaluate_solution(T_dist)

        # Initialize pheromone levels on all edges to a base level (tau_0)
        for i in range(self.graph.node_num):
            for j in range(self.graph.node_num):
                self.graph.pheromone_mat[i][j] = self.graph.tau_0

        iteration = 0
        while iteration < self.max_iter:
            # Create ants to explore the solution space
            ants = [Ant(self.graph) for _ in range(self.ants_num)]
            for ant in ants:
                # Construct a solution for the current ant
                T_k, penalty = self._construct_solution(sigma)
                veichle = self.get_active_vei(T_k)
                # ant.travel_path = T_k

                # Apply local search to improve the solution
                # ant.local_search_procedure()

                # Evaluate the cost of the solution wiht penalty tw and veichle
                C_k = self.evaluate_solution(T_k) + (penalty * self.k1) + (veichle * self.k2)

                # Update the best solution if the current solution is better
                if C_k > C_dist:
                    C_dist = C_k
                    T_dist = T_k

                # Globally update pheromone levels based on the best solution
                self.graph.global_update_pheromone(T_dist, C_dist)

            iteration += 1


        return T_dist, C_dist

    def select_next_index(self, ant, nodes):
        """
        Selects the next index (node) for an ant to visit based on the node type.
        If the nodes are feasible nodes, it calculates transition probabilities.
        If the nodes are remaining customers, it selects randomly.

        Parameters:
        - ant (Ant): The ant which is currently moving.
        - nodes (set): A set of node indices that are feasible or remaining customers.

        Returns:
        - int: The index of the next node for the ant to move to.
        """
        current_index = ant.current_index
        feasible_nodes = set(ant.calculate_feasible_neighbors())
        next_index = None
        if nodes.issubset(feasible_nodes):
            # Nodes are feasible, calculate transition probabilities
            transition_probabilities = np.zeros(len(nodes))
            for idx, j in enumerate(nodes):
                # Then, use it in the transition probability calculation
                transition_probabilities[idx] = (
                        np.power(self.graph.pheromone_mat[current_index][j], self.alpha) * np.power(
                    ant.eta_k_ij_mat[current_index][j], self.beta))

            # Normalize the probabilities for the feasible nodes only
            sum_probabilities = np.sum(transition_probabilities)
            if sum_probabilities > 0:
                normalized_probabilities = transition_probabilities / sum_probabilities
                if np.random.rand() < self.q0:
                    # Exploit by choosing the neighbor with the highest probability
                    max_prob_index = np.argmax(normalized_probabilities)
                    next_index = list(nodes)[max_prob_index]
                else:
                    # Explore by choosing the neighbor based on the stochastic acceptance
                    next_index = MultipleAntColonySystem.stochastic_accept(list(nodes), normalized_probabilities)
        else:
            # Nodes are remaining customers, select randomly
            next_index = np.random.choice(list(nodes))

        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Stochastic acceptance rule for selecting the next index to visit based on transition probabilities.

        Parameters:
        - index_to_visit (list): List of indices to potentially visit.
        - transition_prob (list): List of transition probabilities corresponding to each index.

        Returns:
        - int: The chosen index.
        """

        # Calculate N and max fitness value
        N = len(index_to_visit)

        # Normalize the transition probabilities
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # O(1) selection
        while True:
            # Randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]

    def roulette_wheel_selection(self, ants: list):
        """
        Perform roulette wheel selection to pick an ant based on the quality of its solution.

        The idea behind roulette wheel selection is that ants with better solutions (shorter paths)
        have a higher chance of getting selected.

        Returns:
        - Ant: The selected ant based on roulette wheel selection.
        """

        # Calculate the total distance traveled by each ant
        path_lengths = [self.evaluate_solution(ant.travel_path) for ant in ants]

        # Convert path lengths to fitness values. Here, shorter paths will have a higher fitness value.
        fitness_values = [1.0 / length for length in path_lengths]

        # Compute the cumulative sum of the fitness values using numpy for efficiency.
        cumulative_fitness = np.cumsum(fitness_values)

        # Select a random value in the range [0, total fitness value]
        rand_value = random.uniform(0, cumulative_fitness[-1])

        # Use binary search to find the ant corresponding to the random value
        chosen_idx = np.searchsorted(cumulative_fitness, rand_value)

        return ants[chosen_idx]

    def get_active_vei(self, path: list):
        """
        Calculates the number of active vehicles used in a given path.

        The method counts the occurrences of the depot (represented as '0' in the path)
        to determine the number of vehicles used. Each occurrence indicates the start
        or end of a vehicle's route.

        Args:
        - path (list): The path containing node indices, including the depot (0).

        Returns:
        - int: The number of active vehicles used in the path.
        """
        vei = path.count(0) - 1
        return vei


