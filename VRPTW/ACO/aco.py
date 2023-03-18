import numpy as np
import random
from vrptw_config import VrptwGraph
from ant import Ant
import time
from threading import Event
class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, alpha=1, beta=2, q0=0.9):
        """
        Initializes the basic Ant Colony Optimization (ACO) algorithm.
        
        :param graph: The graph representing the problem instance, including node positions and service time information.
        :param ants_num: The number of ants to be used in the algorithm.
        :param max_iter: The maximum number of iterations to perform.
        :param alpha: The parameter controlling the influence of pheromone trails.
        :param beta: The parameter controlling the importance of heuristic information.
        :param q0: The probability of directly choosing the most probable next node.
        """
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.alpha = alpha
        self.max_load = graph.vehicle_capacity  # The maximum capacity of each vehicle.
        self.beta = beta
        self.q0 = q0
        self.best_path_distance = None  # Best path distance found so far.
        self.best_path = None  # The best path found so far.
        self.best_vehicle_num = None  # The number of vehicles used in the best path.

    def _basic_aco(self):
        """
        Executes the basic ACO algorithm with local search.
        """
        start_time_total = time.time()

        for iter in range(self.max_iter):
            ants = [Ant(self.graph) for _ in range(self.ants_num)]  # Initialize ants for this iteration.

            for k in range(self.ants_num):
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    # Double-check if adding the next node satisfies the constraints.
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0  # Fallback to the depot if no valid next node is found.

                    ants[k].move_to_next_index(next_index)  # Move ant to the next node.
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)

                ants[k].move_to_next_index(0)  # Move ant back to the depot.
                self.graph.local_update_pheromone(ants[k].current_index, 0)
                
            # Compute the distance of all ants' paths.
            paths_distance = np.array([ant.total_travel_distance for ant in ants])
            # Record the current best path.
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1

                print('\n')
                print('[iteration %d]: find a improved path, its distance is %f' % (iter, self.best_path_distance))
                print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))

            # Update the pheromone trail globally.
            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)

        print('\n')
        print('final best path distance is %f, number of vehicle is %d' % (self.best_path_distance, self.best_vehicle_num))
        print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))

    def select_next_index(self, ant):
        """
        Selects the next node for an ant to visit.
        
        :param ant: The ant instance that is selecting the next node.
        :return: The index of the next node to visit.
        """
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        # Calculate the transition probabilities based on pheromone and heuristic information.
        transition_prob = np.power(self.graph.pheromone_mat[current_index][index_to_visit], self.alpha) * \
                          np.power(ant.eta_k_ij_mat[current_index][index_to_visit], self.beta)
        transition_prob /= np.sum(transition_prob)

        # Decide whether to pick the next node based on the highest probability or randomly.
        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)

        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Performs roulette wheel selection (stochastic acceptance) to select the next node.
        
        :param index_to_visit: A list of indices of nodes that have not been visited yet.
        :param transition_prob: The transition probabilities for each node.
        :return: The selected node index.
        """
        N = len(index_to_visit)
        sum_tran_prob = np.sum(transition_prob)
        norm_transition_prob = transition_prob / sum_tran_prob

        # Perform the selection in O(1) time.
        while True:
            ind = int(N * random.random())
            if random.random() <= norm_transition_prob[ind]:
                return index_to_visit[ind]
