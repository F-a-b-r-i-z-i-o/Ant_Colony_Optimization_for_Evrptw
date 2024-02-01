# **Module: Target**

This document describes the `Node` class, which represents a node in an Electric Vehicle Routing Problem with Time Windows (EVRPTW).

## Class: Node

### Attributes

- `idx` (int): Unique index or identifier for the node.
- `string_id` (str): String ID to represent the node. Example formats include 'D0' for depot, 'S0' for station, 'C20' for customer.
- `node_type` (str): Type of the node, which can be 'd' for depot, 'f' for station, or 'c' for customer.
- `x` (float): The x-coordinate of the node's location on a map or grid.
- `y` (float): The y-coordinate of the node's location on a map or grid.
- `demand` (float): The demand or requirement at the node, relevant mainly for customers.
- `ready_time` (float): The earliest time at which service can begin at this node, used in time windows.
- `due_date` (float): The latest time by which service should be completed at this node, used in time windows.
- `service_time` (float): The time required to service the node, relevant for scheduling and routing.

### Constructor

```python
def __init__(self, idx, string_id, node_type, x, y, demand, ready_time, due_date, service_time):
    # Initializes a Node instance for EVRPTW
```

#### Parameters

- `idx` (int): Unique index or identifier for the node.
- `string_id` (str): String representation ID for the node.
- `node_type` (str): Character indicating the type of node ('d', 'f', 'c').
- `x` (float): The x-coordinate of the node's location.
- `y` (float): The y-coordinate of the node's location.
- `demand` (float): Demand or requirement at the node (relevant for customers).
- `ready_time` (float): Earliest time service can begin at the node.
- `due_date` (float): Latest time by which service should be completed at the node.
- `service_time` (float): Time required to service the node.

## Method Descriptions

### Customer Check

#### `is_customer(self)`

- **Purpose:** Checks if the node represents a customer.
- **Returns:** `bool` - Returns `True` if the node is a customer, otherwise `False`.

### Depot Check

#### `is_depot(self)`

- **Purpose:** Identifies if the node is the depot.
- **Returns:** `bool` - Returns `True` for a depot node, `False` for others.

### Charging Station Check

#### `is_station(self)`

- **Purpose:** Determines if the node is a charging station.
- **Returns:** `bool` - True for a charging station, False otherwise.