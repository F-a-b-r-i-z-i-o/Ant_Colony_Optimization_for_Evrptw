import pytest
from ACO.target import Node

def test_node_creation():
    # Test the constructor and attribute assignments.
    node = Node(1, "C20", 'c', 10, 20, 30, 40, 50, 60)
    assert node.idx == 1
    assert node.string_id == "C20"
    assert node.node_type == 'c'
    assert node.x == 10
    assert node.y == 20
    assert node.demand == 30
    assert node.ready_time == 40
    assert node.due_date == 50
    assert node.service_time == 60

def test_is_depot():
    # Test the is_depot method.
    depot_node = Node(0, "D0", 'd', 0, 0, 0, 0, 0, 0)
    assert depot_node.is_depot() is True
    assert depot_node.is_station() is False
    assert depot_node.is_customer() is False

def test_is_station():
    # Test the is_station method.
    station_node = Node(2, "S0", 'f', 5, 5, 0, 0, 0, 0)
    assert station_node.is_depot() is False
    assert station_node.is_station() is True
    assert station_node.is_customer() is False

def test_is_customer():
    # Test the is_customer method.
    customer_node = Node(1, "C20", 'c', 10, 20, 30, 40, 50, 60)
    assert customer_node.is_depot() is False
    assert customer_node.is_station() is False
    assert customer_node.is_customer() is True
