import os
import sys
import numpy as np
# Set up the script directory and the absolute file path
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '..', 'ACO', 'paper_total_instances', 'c103C5.txt')
file_path = os.path.abspath(file_path)

# Add the ACO directory to sys.path to ensure imports work
aco_dir = os.path.join(script_dir, '..', 'ACO')
sys.path.insert(0, aco_dir)

from target import Node
from evrptw_config import EvrptwGraph
from ant import Ant
import pytest


@pytest.fixture
def example_graph():
    nodes = [
        Node(0, "D0", "d", 0, 0, 0, 0, 0, 0),  # Depot
        Node(1, "C1", "c", 1, 1, 5, 8, 10, 1),  # Customer with time window and demand
        Node(2, "S1", "f", 2, 2, 0, 0, 0, 0)  # Charging station
    ]
    graph = EvrptwGraph(file_path)
    graph.nodes = nodes
    graph.node_num = len(nodes)
    graph.node_dist_mat = np.array([
        [0, 1.5, 3],
        [1.5, 0, 1.5],
        [3, 1.5, 0]
    ])
    graph.tank_capacity = 10
    graph.load_capacity = 50
    graph.velocity = 1
    graph.fuel_consumption_rate = 1
    graph.charging_rate = 5
    # Ensure the depot's due date allows for the return from any node
    graph.nodes[0].due_date = 100  # Set a sufficiently large due date for the depot
    return graph


def test_successful_move_to_customer(example_graph):
    ant = Ant(example_graph)
    assert ant.move_to_next_index(1) is True
    assert ant.current_index == 1
    assert ant.vehicle_load == 5  # Assuming the demand at customer node is 5

def test_move_to_charging_station(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 5  # Set fuel level low
    assert ant.move_to_next_index(2) is True
    assert ant.current_index == 2  # Ant should move to the charging station

def test_failed_move_insufficient_energy(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 0.5  # Set fuel very low, not enough to move anywhere
    assert ant.move_to_next_index(1) is False  # Cannot reach customer
    assert ant.move_to_next_index(2) is False  # Cannot reach charging station

def test_failed_move_exceeds_load_capacity(example_graph):
    ant = Ant(example_graph)
    # Set the vehicle load to 46 or more, so adding the customer's demand exceeds the load capacity
    ant.vehicle_load = 46
    # Attempt to move to customer node 1 should fail as it would exceed load capacity
    assert ant.move_to_next_index(1) is False

def test_move_to_customer(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = 10
    initial_travel_distance = ant.total_travel_distance
    initial_travel_time = ant.vehicle_travel_time

    ant.update_travel_state(1)  # Move to customer node 1

    assert ant.vehicle_load == 15  # Load should increase by customer's demand (5)
    assert ant.total_travel_distance > initial_travel_distance  # Travel distance should increase
    assert ant.vehicle_travel_time > initial_travel_time  # Travel time should increase
    assert 1 not in ant.index_to_visit  # Customer node should be removed from index_to_visit

def test_move_to_charging_station(example_graph):
    ant = Ant(example_graph)
    initial_fuel_level = ant.fuel_level

    ant.update_travel_state(2)  # Move to charging station node 2

    assert ant.fuel_level >= initial_fuel_level  # Fuel level should increase
    assert ant.vehicle_load == 0  # Load should not change

def test_move_to_depot(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = 10
    ant.fuel_level = 5

    ant.update_travel_state(0)  # Move to depot node 0

    assert ant.fuel_level == example_graph.tank_capacity  # Fuel level should be reset to tank capacity
    assert ant.vehicle_load == 0  # Load should be reset to 0
    assert ant.vehicle_travel_time == 0  # Travel time should be reset to 0

def test_sufficient_energy_to_move(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 10  # Set a sufficient fuel level
    assert ant.has_enough_energy(0, 1) is True  # Assuming the distance to node 1 is within fuel range

def test_insufficient_energy_no_station(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 1  # Set a low fuel level
    assert ant.has_enough_energy(0, 1) is False  # Assuming there's no reachable charging station

def test_current_index_at_charging_station(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 1  # Set a low fuel level
    current_index = 2  # Assuming node 2 is a charging station
    assert ant.has_enough_energy(current_index, 1) is True  # Should return True as it assumes refueling

def test_single_visit_constraint(example_graph):
    ant = Ant(example_graph)
    ant.travel_path = [1]  # Assume node 1 has already been visited
    assert ant.check_condition(1) is False  # Attempt to revisit node 1

def test_capacity_constraint(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = example_graph.load_capacity
    assert ant.check_condition(1) is False  # Next move exceeds load capacity

def test_energy_constraint(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 0  # No fuel left
    assert ant.check_condition(1) is False  # Insufficient energy to move

def test_time_window_constraint(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_travel_time = 100  # Set a high travel time
    assert ant.check_condition(1) is False  # Cannot serve within time window

def test_return_to_depot_feasibility(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_travel_time = example_graph.nodes[0].due_date  # Set travel time to depot's due time
    assert ant.check_condition(1) is False  # Cannot return to depot in time

def test_all_constraints_met(example_graph):
    ant = Ant(example_graph)
    ant.fuel_level = 10  # Sufficient fuel
    ant.vehicle_load = 0  # No load
    assert ant.check_condition(1) is True  # All constraints are met

def test_arrival_time_calculation(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_travel_time = 5  # Assume a travel time of 5 units
    ant.current_index = 0  # Starting from index 0
    next_index = 1  # Moving to index 1

    # Calculate expected arrival time
    distance_to_next_index = example_graph.node_dist_mat[0][1]  # Distance from index 0 to 1
    expected_arrival_time = ant.vehicle_travel_time + distance_to_next_index / example_graph.velocity

    assert ant.calculate_arrival_time(next_index) == expected_arrival_time

def test_check_time_window(example_graph):
    ant = Ant(example_graph)
    arrival_time = 5  # Example arrival time
    next_node = example_graph.nodes[1]  # Example node

    # Assuming next_node has specific ready_time and due_date
    assert ant.check_time_window(arrival_time, next_node) == (True, 0)  # Adjust according to expected values


def test_return_to_depot(example_graph):
    ant = Ant(example_graph)
    ant.current_index = 1  # Starting from an example index
    ant.return_to_depot()

    assert ant.current_index == 0  # Ant should now be at the depot
    assert ant.vehicle_load == 0  # Load should be reset
    assert ant.fuel_level == example_graph.tank_capacity  # Fuel should be refilled


def test_can_return_to_depot_in_time(example_graph):
    ant = Ant(example_graph)
    current_time = 90  # Set a current time where returning to the depot is possible
    next_index = 1  # Choose a node close to the depot

    # Test if the ant can return to the depot from the next index in time
    assert ant.can_return_to_depot_from(next_index, current_time) == True

def test_cannot_return_to_depot_in_time(example_graph):
    ant = Ant(example_graph)
    current_time = 98  # Set a current time very close to the depot's due time
    next_index = 2  # Choose a node farther from the depot

    # Test if the ant cannot return to the depot from the next index in time
    assert ant.can_return_to_depot_from(next_index, current_time) == False

def test_check_capacity_constraint(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = 20  # Set a current load
    assert ant.check_capacity_constraint(1) == True  # Assuming node 1's demand does not exceed capacity
    ant.vehicle_load = 49  # Set load close to capacity
    assert ant.check_capacity_constraint(1) == False  # Assuming node 1's demand exceeds capacity

def test_cal_nearest_next_index(example_graph):
    ant = Ant(example_graph)
    ant.current_index = 0  # Starting from depot
    next_index_list = [1, 2]
    assert ant.cal_nearest_next_index(next_index_list) == 1  # Assuming node 1 is closer to the depot than node 2

def test_calculate_feasible_neighbors(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = 20  # Set a current load
    feasible_neighbors = ant.calculate_feasible_neighbors()
    # Assuming the feasible neighbors meet all constraints and exclude charging stations and the depot
    assert all(index not in [0, 2] for index in feasible_neighbors)

def test_cal_total_travel_distance_non_empty_path(example_graph):
    travel_path = [0, 1, 2]  # A sample travel path
    # Manually calculate the expected total distance
    expected_distance = example_graph.node_dist_mat[0][1] + example_graph.node_dist_mat[1][2]
    assert Ant.cal_total_travel_distance(example_graph, travel_path) == expected_distance

def test_is_feasible_with_feasible_path(example_graph):
    ant = Ant(example_graph)
    feasible_path = [0, 1, 2, 0]  # Assumendo che questo sia un percorso fattibile
    assert ant.is_feasible(feasible_path, ant) is True

def test_is_feasible_with_path_exceeding_load_capacity(example_graph):
    ant = Ant(example_graph)
    ant.vehicle_load = example_graph.load_capacity
    infeasible_path_due_to_load = [0, 1, 0]  

    assert not ant.is_feasible(infeasible_path_due_to_load, ant)


def test_is_feasible_not_ending_at_depot(example_graph):
    ant = Ant(example_graph)
    path_not_ending_at_depot = [0, 1, 2]  # Does not end at depot
    assert ant.is_feasible(path_not_ending_at_depot, ant) is False

def test_local_search_swap_improvement(example_graph):
    ant = Ant(example_graph)
    initial_path = [0, 1, 2, 0]  # Sample initial path
    improved_path, improved_distance = ant.local_search_2opt(example_graph, initial_path)
    initial_distance = Ant.cal_total_travel_distance(example_graph, initial_path)

    assert improved_distance <= initial_distance
    assert ant.is_feasible(improved_path, ant)

def test_local_search_swap_no_improvement(example_graph):
    ant = Ant(example_graph)
    path = [0, 1, 2, 0]  # Path where no improvement is possible
    new_path, new_distance = ant.local_search_2opt(example_graph, path)
    initial_distance = Ant.cal_total_travel_distance(example_graph, path)

    assert new_distance == initial_distance
    assert new_path == path

def test_local_search_swap_skips_depot(example_graph):
    ant = Ant(example_graph)
    path = [0, 1, 2, 0]
    new_path, _ = ant.local_search_2opt(example_graph, path)

    # Verify that depot (node 0) is not swapped
    assert new_path[0] == 0 and new_path[-1] == 0
