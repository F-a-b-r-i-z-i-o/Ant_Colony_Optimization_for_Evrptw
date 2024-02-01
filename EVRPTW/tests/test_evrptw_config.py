import os
import sys
import numpy as np
import tempfile
# Set up the script directory and the absolute file path
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '..', 'ACO', 'paper_total_instances', 'c103C5.txt')
file_path = os.path.abspath(file_path)

# Add the ACO directory to sys.path to ensure imports work
aco_dir = os.path.join(script_dir, '..', 'ACO')
sys.path.insert(0, aco_dir)

from target import Node
from evrptw_config import EvrptwGraph
import pytest

# Sample content of an EVRPTW instance file
SAMPLE_INSTANCE = """
D0 d 0 0 0 0 0 0
F1 f 1 1 0 0 0 0
C1 c 2 2 1 10 20 30
Vehicle fuel tank capacity: 100.0
Vehicle load capacity: 200.0
Vehicle fuel consumption rate: 1.0
Vehicle inverse refueling rate: 2.0
Vehicle average Velocity: 60.0
"""
# Fixture for creating an example graph
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
    return graph

def test_global_update_pheromone(example_graph):
    # Test the global pheromone update method with a given best path and distance
    best_path = [0, 1, 2, 3, 4]
    best_path_distance = 10

    # Perform the global pheromone update
    example_graph.global_update_pheromone(best_path, best_path_distance)

    # Verify the pheromone levels are updated correctly for each edge in the best path
    for i in range(len(best_path) - 1):
        current_ind = best_path[i]
        next_ind = best_path[i + 1]
        # Calculate the expected pheromone level for the edge
        updated_pheromone = example_graph.tau_0 * (1 - example_graph.rho) + example_graph.rho / best_path_distance
        # Assert that the pheromone level matches the expected value
        assert example_graph.pheromone_mat[current_ind][next_ind] == pytest.approx(updated_pheromone)


def test_read_instance():
    # Create a temporary file with the sample instance data
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
        temp_file.write(SAMPLE_INSTANCE)
        temp_file.seek(0)
        file_path = temp_file.name

    # Unpack the tuple returned by _read_instance
    node_num, nodes, node_dist_mat, fuel_stations, depot, tank_capacity, load_capacity, fuel_consumption_rate, charging_rate, velocity = EvrptwGraph._read_instance(file_path)

    # Verify that the extracted values match the expected values
    assert node_num == 3  # Adjust according to the SAMPLE_INSTANCE content
    assert len(fuel_stations) == 1
    assert depot.string_id == "D0"
    assert tank_capacity == 100.0
    assert load_capacity == 200.0
    assert fuel_consumption_rate == 1.0
    assert charging_rate == 2.0
    assert velocity == 60.0
    
@pytest.fixture
def graph_for_distance_matrix():
    nodes = [
        Node(0, "A", "a", 0, 0, 0, 0, 0, 0),
        Node(1, "B", "b", 3, 0, 0, 0, 0, 0),
        Node(2, "C", "c", 0, 4, 0, 0, 0, 0)
    ]
    return nodes

def test_calculate_distance_matrix(graph_for_distance_matrix):
    distance_matrix = EvrptwGraph.calculate_distance_matrix(graph_for_distance_matrix)
    assert np.array_equal(distance_matrix, np.array([
        [1e-9, 3.0, 4.0],
        [3.0, 1e-9, 5.0],
        [4.0, 5.0, 1e-9]
    ]))


@pytest.fixture
def graph_with_stations():
    nodes = [
        Node(0, "C0", "c", 0, 0, 0, 0, 0, 0),
        Node(1, "S1", "f", 1, 1, 0, 0, 0, 0),  # Station
        Node(2, "C2", "c", 2, 2, 0, 0, 0, 0),
        Node(3, "S3", "f", 3, 3, 0, 0, 0, 0)   # Another Station
    ]
    graph = EvrptwGraph(file_path)  # Assuming the constructor can be called without file_path
    graph.nodes = nodes
    graph.node_num = len(nodes)
    graph.node_dist_mat = np.array([
        [0, 1.0, 2.0, 3.0],
        [1.0, 0, 1.0, 2.0],
        [2.0, 1.0, 0, 1.0],
        [3.0, 2.0, 1.0, 0]
    ])
    return graph

def test_select_closest_station_with_stations(graph_with_stations):
    station_idx, total_distance = graph_with_stations.select_closest_station(0, 2)
    assert station_idx == 1  # Expected closest station index
    assert total_distance == 2.0  # Expected total distance to the station

def test_select_closest_station_no_stations():
    graph_without_stations = EvrptwGraph(file_path)
    graph_without_stations.nodes = [
        Node(0, "C0", "c", 0, 0, 0, 0, 0, 0),
        Node(1, "C1", "c", 1, 1, 0, 0, 0, 0)
    ]
    graph_without_stations.node_dist_mat = np.array([
        [0, 1.0],
        [1.0, 0]
    ])

    station_idx, total_distance = graph_without_stations.select_closest_station(0, 1)
    assert station_idx == -1  # Expecting -1 as no station is found
    assert total_distance is None  # Expecting None as no station is found

@pytest.fixture
def simple_graph():
    nodes = [
        Node(0, "D0", "d", 0, 0, 0, 0, 0, 0),  # Depot
        Node(1, "C1", "c", 1, 1, 1, 0, 0, 0),  # Customer
        Node(2, "S1", "f", 2, 2, 0, 0, 0, 0)   # Charging station
    ]
    graph = EvrptwGraph(file_path)  # Assuming the constructor can be called without parameters
    graph.nodes = nodes
    graph.node_num = len(nodes)
    graph.node_dist_mat = np.array([
        [0, 1.0, 2.0],
        [1.0, 0, 1.0],
        [2.0, 1.0, 0]
    ])
    graph.tank_capacity = 30
    graph.load_capacity = 5
    graph.velocity = 1
    graph.fuel_consumption_rate = 0.5
    graph.charging_rate = 1
    return graph


def test_nearest_neighbor_heuristic_basic(simple_graph):
    travel_path, travel_distance, vehicle_num = simple_graph.nearest_neighbor_heuristic()

    # Adjust expected path and distance based on the logic of the method and the setup of simple_graph
    expected_path = [0, 2, 0]  # Adjust if needed
    expected_distance = 4.0  # Adjust if needed

    assert travel_path == expected_path
    assert travel_distance == expected_distance
    assert vehicle_num == 1  # Assuming one vehicle is enough for this simple scenario


def test_cal_nearest_next_index(simple_graph):
    index_to_visit = [1, 2]  # Adjust indices based on test_graph's setup
    current_index = 0
    current_battery = simple_graph.tank_capacity
    current_time = 0

    expected_index = 2  # Adjust this based on your test_graph's setup
    next_index = simple_graph._cal_nearest_next_index(index_to_visit, current_index, current_battery, current_time)

    assert next_index == expected_index



