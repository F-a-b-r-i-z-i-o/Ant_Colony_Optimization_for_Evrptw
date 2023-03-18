class Node:
    def __init__(self, idx, string_id, node_type, x, y, demand, ready_time, due_date, service_time):
        # Constructor for the Node class.
        self.idx = idx                   # Unique index or identifier for the node.
        self.string_id = string_id       # String ID to represent the node (e.g., D0 for depot, S0 for station, C20 for customer).
        self.node_type = node_type       # Type of the node: 'd' for depot, 'f' for station, 'c' for customer.
        self.x = x                       # The x-coordinate of the node's location.
        self.y = y                       # The y-coordinate of the node's location.
        self.demand = demand             # The demand or requirement at the node (relevant for customers).
        self.ready_time = ready_time     # The earliest time at which service can begin at this node.
        self.due_date = due_date         # The latest time by which service should be completed at this node.
        self.service_time = service_time # Time required to service the node.

    def is_depot(self):
        # Checks if the node is a depot.
        return self.node_type == 'd'
    
    def is_station(self):
        # Checks if the node is a station.
        return self.node_type == 'f'
    
    def is_customer(self):
        # Checks if the node is a customer.
        return self.node_type == 'c'
