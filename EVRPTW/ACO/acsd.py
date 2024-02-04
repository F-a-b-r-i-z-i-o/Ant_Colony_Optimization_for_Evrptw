import numpy as np
import random
from evrptw_config import EvrptwGraph
from ant import Ant
import time


class AntColonySystem:
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
        self,
        graph: EvrptwGraph,
        ants_num=10,
        max_iter=400,
        alpha=1,
        beta=2,
        q0=0.9,
        k1=0,
        k2=0,
        apply_local_search=False,
    ):
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.beta = beta
        self.alpha = alpha
        self.q0 = q0
        self.k1 = k1
        self.k2 = k2
        self.apply_local_search = apply_local_search

    def _construct_solution(self, sigma: int) -> tuple[list, float]:
        """
        Constructs a solution (path) for a given number of vehicles (sigma) based on the
        ant colony system and various constraints, including energy limits.

        The function iteratively builds a path by selecting nodes (customers, depot, charging stations)
        based on the feasibility and constraints like energy limits and time windows. It utilizes
        multiple vehicles if necessary, as specified by the parameter sigma.

        Args:
        - sigma (int): The number of vehicles available for constructing the solution.

        Returns:
        - tuple:
            - list: T_k, the constructed travel path that includes customers, depot, and possibly charging stations.
            - float: cumulative_time_window_penalty, the total penalty incurred due to time window violations
                    over the entire constructed path.
        """

        T_k = [self.graph.depot.idx]  # Inizialize path from depot
        vehicles = 1
        visited_customers = set()
        all_customers = set(
            i for i, node in enumerate(self.graph.nodes) if node.is_customer()
        )
        ants = [Ant(self.graph) for _ in range(self.ants_num)]
        cumulative_time_window_penalty = 0
        penalty_rate = 1.0

        while vehicles <= sigma and visited_customers != all_customers:
            i = T_k[-1]  # Current Node
            ant = self.roulette_wheel_selection(ants)  # Select ant
            ant.travel_path.clear()
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

                within_time_window, time_violation = ant.check_time_window(
                    arrival_time_at_j, next_node
                )

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
                else:
                    T_k.append(j)
                    visited_customers.add(j)
                    ant.move_to_next_index(j)

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

    def evaluate_solution(self, travel_path: list) -> float:
        """
        Evaluate the total distance of a given travel path.

        Returns:
        - float: The total distance of the provided travel path.
        """

        # Check if the graph is properly initialized
        if self.graph is None:
            raise ValueError("Graph not initialized in MultipleAntColonySystem")

        # Check if the path is empty and return 0 if it is
        if not travel_path:
            return 0

        # Initialize the distance variable
        distance = 0
        current_ind = travel_path[0]

        # Calculate the distance between consecutive nodes in the path
        for next_ind in travel_path[1:]:
            distance += self.graph.node_dist_mat[current_ind][next_ind]
            current_ind = next_ind

        # Add the distance from the last node in the path back to the depot
        distance += self.graph.node_dist_mat[current_ind][travel_path[0]]

        return distance

    def _ACS_DIST_G(self, apply_local_search=True) -> tuple:
        """
        Performs the Ant Colony System algorithm for the Electric Vehicle Routing Problem with Time Windows (EVRPTW).
        This function iterates over a set number of ants, allowing each to construct a solution path.
        The best path found is then globally updated.

        Returns:
            tuple: Contains the final best path distance, the best path, and various histories
                (final cost, time, penalties, improvements).
        """
        C_final = float("inf")
        T_best = None
        m = self.graph.vehicles
        start_time = time.time()
        C_final_history = []
        time_history = []
        penalty_history = []
        improvement_iter_history = []
        improved_path = []

        # Initialize pheromone levels
        for i in range(self.graph.node_num):
            for j in range(self.graph.node_num):
                self.graph.pheromone_mat[i][j] = self.graph.tau_0

        for iteration in range(self.max_iter):
            ants = [Ant(self.graph) for _ in range(self.ants_num)]

            local_best_cost = float("inf")
            local_best_path = None
            local_best_penalty = None

            for ant in ants:
                T_ant, initial_penalty = self._construct_solution(m)

                # Apply local search if enabled
                if apply_local_search:
                    T_ant, improved_path_distance = ant.local_search_2opt(
                        self.graph, T_ant
                    )
                else:
                    improved_path_distance = self.evaluate_solution(T_ant)

                num_vehicles = self.get_active_vei(T_ant)
                C_ant = (
                    improved_path_distance
                    + (initial_penalty * self.k1)
                    + (num_vehicles * self.k2)
                )

                if C_ant < local_best_cost:
                    local_best_cost = C_ant
                    local_best_path = T_ant
                    local_best_penalty = initial_penalty

            # Update global best path if a new local best is found
            if local_best_cost < C_final:
                elapsed_time = time.time() - start_time
                C_final = local_best_cost
                T_best = local_best_path
                C_final_history.append(C_final)
                time_history.append(elapsed_time)
                penalty_history.append(local_best_penalty)
                improved_path.append(T_best)
                improvement_iter_history.append(iteration)

                print(
                    f"[Iteration {iteration}]: Found improved path with cost {C_final}. Time: {time_history[-1]:.3f} seconds"
                )
                self.graph.global_update_pheromone(T_best, C_final)

        total_time = time.time() - start_time
        print(
            f"Final best path distance: {C_final}. Total time: {total_time:.3f} seconds"
        )

        return (
            C_final,
            C_final_history,
            time_history,
            penalty_history,
            improvement_iter_history,
            num_vehicles,
            improved_path,
        )

    def select_next_index(self, ant: Ant, nodes: list) -> int:
        """
        Selects the next index for an ant to visit based on the transition probabilities calculated from
        the pheromone trail and heuristic information.

        The selection strategy incorporates both exploitation (choosing the best option) and exploration
        (randomly choosing based on probability distribution) to balance between finding optimal paths
        and exploring new paths.

        Args:
            ant (Ant): The ant object which is currently constructing a path.
            nodes (list): A list of node indices that the ant can potentially visit next.

        Returns:
            int: The index of the next node the ant will move to. Returns None if there are no feasible nodes
        """
        if not nodes:
            return None

        current_index = ant.current_index
        feasible_nodes = set(ant.calculate_feasible_neighbors())
        next_index = None

        if nodes.issubset(feasible_nodes):
            # Calculate transition probabilities
            transition_probabilities = np.zeros(len(nodes))
            for idx, j in enumerate(nodes):
                transition_probabilities[idx] = np.power(
                    self.graph.pheromone_mat[current_index][j], self.alpha
                ) * np.power(ant.eta_k_ij_mat[current_index][j], self.beta)
            # Normalize the probabilities
            sum_probabilities = np.sum(transition_probabilities)
            normalized_probabilities = (
                transition_probabilities / sum_probabilities
                if sum_probabilities > 0
                else None
            )

            if normalized_probabilities is not None:
                if np.random.rand() < self.q0:
                    # Exploitation: Choose the next node with the highest probability
                    max_prob_index = np.argmax(normalized_probabilities)
                    next_index = list(nodes)[max_prob_index]
                else:
                    # Exploration: Choose the next node based on the probability distribution
                    next_index = AntColonySystem.stochastic_accept(
                        list(nodes), normalized_probabilities
                    )
            else:
                next_index = np.random.choice(list(nodes), p=normalized_probabilities)
        else:
            # If sum of probabilities is zero, select randomly
            next_index = np.random.choice(list(nodes))

        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit: list, transition_prob: list) -> int:
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

    def roulette_wheel_selection(self, ants: list) -> Ant:
        """
        Perform roulette wheel selection to pick an ant based on the quality of its solution.

        The idea behind roulette wheel selection is that ants with better solutions (shorter paths)
        have a higher chance of getting selected.

        Returns:
        - Ant: The selected ant based on roulette wheel selection.
        """

        # Calculate the total distance traveled by each ant
        path_lengths = [self.evaluate_solution(ant.travel_path) for ant in ants]

        # Convert path lengths to fitness values, handling zero or very small lengths
        fitness_values = []
        for length in path_lengths:
            if length > 0:
                fitness_value = 1.0 / length
            else:
                # Assign a small fitness value to avoid division by zero
                fitness_value = (
                    1e-8  # This can be adjusted based on your algorithm's needs
                )
            fitness_values.append(fitness_value)

        # Compute the cumulative sum of the fitness values using numpy for efficiency
        cumulative_fitness = np.cumsum(fitness_values)

        # Select a random value in the range [0, total fitness value]
        rand_value = random.uniform(0, cumulative_fitness[-1])

        # Use binary search to find the ant corresponding to the random value
        chosen_idx = np.searchsorted(cumulative_fitness, rand_value)

        return ants[chosen_idx]

    def get_active_vei(self, path: list) -> int:
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
