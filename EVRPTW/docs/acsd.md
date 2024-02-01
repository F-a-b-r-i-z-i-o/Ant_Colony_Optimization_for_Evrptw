# Module: Ant Colony System (ACS)

## Class: AntColonySystem

The `AntColonySystem` class implements a Multiple Ant Colony System for solving the Electric Vehicle Routing Problem with Time Windows (EVRPTW).

### Constructor

```python
def __init__(self, graph: EvrptwGraph, ants_num=10, max_iter=400, alpha=1, beta=2, q0=0.9, k1=2, apply_local_search=True)`
  # Initializes the Ant Colony System.
```
- **Parameters:**
  - `graph` (EvrptwGraph): The graph representing the EVRPTW problem.
  - `ants_num` (int, default=10): Number of ants in the system.
  - `max_iter` (int, default=400): Maximum number of iterations.
  - `alpha` (float, default=1): Parameter influencing the importance of pheromone.
  - `beta` (float, default=2): Parameter influencing the importance of heuristic information.
  - `q0` (float, default=0.9): Parameter for pseudo-random proportional decision rule.
  - `k1` (float, default=2): Coefficient for time window penalty.
  - `apply_local_search` (bool, default=True): Flag to apply local search or not.

### Methods

#### `_construct_solution(self, sigma: int)`

Constructs a solution (path) based on the ant colony system and constraints.

- **Parameters:**
  - `sigma` (int): Number of vehicles.
- **Returns:**
  - `tuple`: The constructed travel path and cumulative time window penalty.

#### `evaluate_solution(self, travel_path: list)`

Evaluates the total distance of a given travel path.

- **Parameters:**
  - `travel_path` (list): The travel path to evaluate.
- **Returns:**
  - `float`: The total distance of the travel path.

#### `_ACS_DIST_G(self, apply_local_search=True)`

Performs the Ant Colony System algorithm for EVRPTW.

- **Parameters:**
  - `apply_local_search` (bool): Flag to apply local search or not.
- **Returns:**
  - `tuple`: Final best path distance, the best path, and various histories.

#### `select_next_index(self, ant, nodes)`

Selects the next node index for the ant to visit.

- **Parameters:**
  - `ant` (Ant): The ant making the decision.
  - `nodes` (set): Set of node indices.
- **Returns:**
  - `int`: The chosen node index.

#### `stochastic_accept(index_to_visit, transition_prob)`

Stochastic acceptance rule for selecting the next index.

- **Parameters:**
  - `index_to_visit` (list): List of indices to visit.
  - `transition_prob` (list): Transition probabilities.
- **Returns:**
  - `int`: The chosen index.

#### `roulette_wheel_selection(self, ants: list)`

Roulette wheel selection to pick an ant.

- **Parameters:**
  - `ants` (list): List of ant instances.
- **Returns:**
  - `Ant`: The selected ant.

#### `get_active_vei(self, path: list)`

Calculates the number of active vehicles used in a given path.

- **Parameters:**
  - `path` (list): The path containing node indices, including the depot (0).
- **Returns:**
  - `int`: The number of active vehicles used in the path.

