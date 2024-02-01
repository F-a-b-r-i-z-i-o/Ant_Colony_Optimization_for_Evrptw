# **Module: Evrptw Config**

This module contains the `EvrptwGraph` class, which is designed to solve the Electric Vehicle Routing Problem with Time Windows (EVRPTW).

## Class: EvrptwGraph

### Attributes

- `node_num` (int): Total number of nodes within the graph.
- `nodes` (list of Node): List of `Node` instances representing the nodes within the EVRPTW graph.
- `node_dist_mat` (numpy.ndarray): Matrix representing the distances between nodes.
- `fuel_stations` (list of Node): List of nodes that serve as fuel stations.
- `depot` (Node): Node that is designated as the depot.
- `tank_capacity` (float): Maximum fuel capacity of the electric vehicle.
- `load_capacity` (float): Maximum carrying capacity of the electric vehicle.
- `fuel_consumption_rate` (float): Rate of fuel consumption by the electric vehicle.
- `charging_rate` (float): Rate at which the vehicle's battery can be charged.
- `velocity` (float): Average velocity of the electric vehicle.

### Constructor

```python
def __init__(self, file_path, rho=0.1):
    # Initializes an EvrptwGraph instance.
```

#### Parameters

- `file_path` (str): Path to the file containing the EVRPTW instance data.
- `rho` (float, optional): Evaporation rate for the pheromone trail.

## Methods Overview

The `EvrptwGraph` class includes several methods to solve the Electric Vehicle Routing Problem with Time Windows (EVRPTW). These methods are designed to calculate distances, update pheromone levels, select paths and charging stations, and read problem instances.

### Global Pheromone Update

#### `global_update_pheromone(self, best_path: list, best_path_distance: float)`

- **Purpose:** Updates pheromone levels globally based on the best path found.
- **Returns:** `None`

### Distance Matrix Calculation

#### `calculate_distance_matrix(nodes: list)`

- **Purpose:** Computes the Euclidean distance matrix for the provided nodes.
- **Returns:** `np.ndarray` representing the distance matrix.

### Instance Reading

#### `_read_instance(self, file_path: str)`

- **Purpose:** Reads and parses the EVRPTW instance data from a file.
- **Returns:** `tuple` containing nodes, distance matrix, fuel stations, depot, capacities, rates, and velocity.

### Closest Charging Station Selection

#### `select_closest_station(self, i: int, j: int)`

- **Purpose:** Identifies the nearest charging station to the given nodes.
- **Returns:** `tuple` with the index of the selected station and the distance. Returns (-1, None) if no station is found.

### Path Coordinates Mapping

#### `get_coordinates_from_path(self, path: list)`

- **Purpose:** Retrieves the (x, y) coordinates for each node in the path.
- **Returns:** List of coordinate tuples for the nodes in the path.

### Node Type Mapping

#### `create_node_type_map(self)`

- **Purpose:** Generates a map linking each node's index to its type.
- **Returns:** `dict` mapping node indices (int) to their types (str).

### Nearest Neighbor Heuristic

#### `nearest_neighbor_heuristic(self)`

- **Purpose:** Implements the nearest neighbor heuristic for route planning, considering various constraints.
- **Returns:** `tuple` with the travel path, total travel distance, and the number of vehicles used.

### Next Index Calculation

#### `_cal_nearest_next_index(self, index_to_visit: list, current_index: int, current_battery: float, current_time: float)`

- **Purpose:** Finds the nearest viable next node for the vehicle to visit.
- **Returns:** `int` index of the nearest next node, or None if none are suitable.






