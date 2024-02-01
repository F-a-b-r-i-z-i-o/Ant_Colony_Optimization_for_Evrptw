# Module: ant

## Class: Ant

The `Ant` class is designed to simulate an ant's journey in solving the Electric Vehicle Routing Problem with Time Windows (EVRPTW).

### Constructor

```python
def __init__(self, graph: EvrptwGraph, start_index=0)
    # Initializes an `Ant` object.
```
#### Parameters:

- `graph` (EvrptwGraph): The graph representing the EVRPTW problem.
- `start_index` (int, default=0): The starting index in the graph, typically the depot.

### Static Methods

#### `cal_total_travel_distance(graph: EvrptwGraph, travel_path: list) -> float`

Calculates the total travel distance for a given path.

- **Returns:** 
  - `float`: The total travel distance of the path.

### Methods

#### `update_eta_matrix(self)`

Updates the eta matrix for routing decisions based on travel time and time window constraints.

#### `move_to_next_index(self, next_index) -> bool`

Attempts to move the ant to the specified next node index.

- **Parameters:**
  - `next_index` (int): The index of the next node to move to.
- **Returns:**
  - `bool`: True if the move was successful, False otherwise.

#### `update_travel_state(self, next_index)`

Updates the travel state of the ant when moving to a new node.

- **Parameters:**
  - `next_index` (int): The index of the next node.

#### `handle_depot_visit(self)`

Handles the ant's visit to the depot.

#### `handle_customer_visit(self, customer_index) -> bool`

Handles the ant's visit to a customer node.

- **Parameters:**
  - `customer_index` (int): The index of the customer node.
- **Returns:**
  - `bool`: True if the visit is successful, False otherwise.

#### `handle_station_visit(self)`

Handles the ant's visit to a charging station.

#### `has_enough_energy(self, current_index, next_index) -> bool`

Checks if the ant has enough energy to move to the next node.

- **Parameters:**
  - `current_index` (int): The current node index.
  - `next_index` (int): The next node index.
- **Returns:**
  - `bool`: True if there is enough energy, False otherwise.

#### `check_condition(self, next_index) -> bool`

Checks if moving to the next node meets all constraints.

- **Parameters:**
  - `next_index` (int): The index of the next node.
- **Returns:**
  - `bool`: True if all constraints are met, False otherwise.

#### `calculate_arrival_time(self, next_index) -> float`

Calculates the arrival time at the next node.

- **Parameters:**
  - `next_index` (int): The index of the next node.
- **Returns:**
  - `float`: The estimated arrival time.

#### `check_time_window(self, arrival_time, next_node) -> bool`

Checks if the service at the next node can start within its time window.

- **Parameters:**
  - `arrival_time` (float): The arrival time at the node.
  - `next_node` (Node): The node object.
- **Returns:**
  - `bool`: True if service can start within the time window, False otherwise.

#### `can_return_to_depot_from(self, next_index, current_time) -> bool`

Checks if the ant can return to the depot in time from the next node.

- **Parameters:**
  - `next_index` (int): The index of the next node.
  - `current_time` (float): The current time.
- **Returns:**
  - `bool`: True if the ant can return to the depot in time, False otherwise.

#### `return_to_depot(self)`

Handles the logic for the ant to return to the depot.

#### `check_capacity_constraint(self, next_index) -> bool`

Checks if the ant's load capacity is not exceeded when visiting the next node.

- **Parameters:**
  - `next_index` (int): The index of the next node.
- **Returns:**
  - `bool`: True if capacity constraint is not violated, False otherwise.

#### `cal_nearest_next_index(self, next_index_list) -> int`

Calculates the nearest node index from the current position.

- **Parameters:**
  - `next_index_list` (list): A list of node indices to consider.
- **Returns:**
  - `int`: The index of the nearest node.

#### `calculate_feasible_neighbors(self) -> list`

Calculates feasible neighbors for the ant, excluding stations and the depot.

- **Returns:**
  - `list`: A list of indices of feasible neighbor nodes.

#### `is_feasible(self, path, ant) -> bool`

Verifies if a given path is feasible for the EVRPTW.

- **Parameters:**
  - `path` (list): The path to be checked.
  - `ant` (Ant): The ant object with methods to check constraints.
- **Returns:**
  - `bool`: True if the path is feasible, False otherwise.

#### `local_search_swap(self, graph, best_path) -> tuple`

Performs a local search optimization by swapping nodes in the best path.

- **Parameters:**
  - `graph` (EvrptwGraph): The graph representing the EVRPTW problem.
  - `best_path` (list): The current best path.
- **Returns:**
  - A tuple containing the updated best path and its total travel distance.
