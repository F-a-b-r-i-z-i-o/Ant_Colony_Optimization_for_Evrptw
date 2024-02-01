import os
import sys
import numpy as np

# Set up the script directory and the absolute file path for test instances
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
from acsd import AntColonySystem


# Fixture for creating an example graph for testing
@pytest.fixture
def example_graph():
    # Setting up nodes for the graph (depot, customer, and charging station)
    nodes = [
        Node(0, "D0", "d", 0, 0, 0, 0, 0, 0),  # Depot
        Node(1, "C1", "c", 1, 1, 5, 8, 10, 1),  # Customer with time window and demand
        Node(2, "S1", "f", 2, 2, 0, 0, 0, 0)   # Charging station
    ]
    # Creating the graph and setting its properties
    graph = EvrptwGraph(file_path)
    graph.nodes = nodes
    graph.node_num = len(nodes)
    graph.node_dist_mat = np.array([[0, 1.5, 3], [1.5, 0, 1.5], [3, 1.5, 0]])
    graph.tank_capacity = 10
    graph.load_capacity = 50
    graph.velocity = 1
    graph.fuel_consumption_rate = 1
    graph.charging_rate = 5
    graph.nodes[0].due_date = 100  # Large due date for depot
    return graph

# Test for constructing a solution path
def test_construct_solution(example_graph):
    macs_system = AntColonySystem(example_graph)
    path, penalty = macs_system._construct_solution(3)
    # Verifying that the path starts and ends at the depot
    assert path[0] == path[-1] == example_graph.nodes[0].idx
    # Verifying that all visited nodes are part of the graph
    visited_nodes = set(path)
    all_nodes = set(range(len(example_graph.nodes)))
    assert visited_nodes.issubset(all_nodes)

# Test for evaluating a solution with an empty path
def test_evaluate_solution_empty(example_graph):
    macs_system = AntColonySystem(example_graph)
    # Expecting the distance for an empty path to be 0
    assert macs_system.evaluate_solution([]) == 0

# Test for evaluating a non-empty solution path
def test_evaluate_solution_non_empty(example_graph):
    macs_system = AntColonySystem(example_graph)
    path = [0, 1, 2, 0]
    # Expected distance for the given path
    expected_distance = 6.0
    assert macs_system.evaluate_solution(path) == expected_distance

# Test for the ACS algorithm (_ACS_DIST_G)
def test_acs_dist_g(example_graph):
    macs_system = AntColonySystem(example_graph)
    # Running the ACS algorithm and verifying that it finds a finite solution
    C_final, *_ = macs_system._ACS_DIST_G()
    assert C_final != float('inf')

# Test for the roulette wheel selection mechanism
def test_roulette_wheel_selection(example_graph):
    macs_system = AntColonySystem(example_graph)
    # Creating Ant objects with valid travel paths
    ants = [Ant(example_graph) for _ in range(4)]
    # Setting a sample travel path for each ant
    for ant in ants:
        ant.travel_path = [0, 1, 2, 0]
    # Performing roulette wheel selection and verifying the selected ant is from the list
    selected_ant = macs_system.roulette_wheel_selection(ants)
    assert selected_ant in ants

# Test for calculating the number of active vehicles in a path
def test_get_active_vei(example_graph):
    macs_system = AntColonySystem(example_graph)
    # Example path using two vehicles
    path = [0, 1, 2, 0, 3, 0]
    num_vehicles = macs_system.get_active_vei(path)
    # Verifying that two vehicles are used in the path
    assert num_vehicles == 2