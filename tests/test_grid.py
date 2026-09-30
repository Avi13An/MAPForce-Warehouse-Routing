"""
Unit tests for Module 1 environment and layout components.
"""
import pytest
from src.environment.grid import WarehouseGrid
from src.environment.layouts import build_custom_narrow_aisle_warehouse

def test_bounds_and_static_obstacles():
    raw_map = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    env = WarehouseGrid(raw_map, height=3, width=3)
    
    assert env.is_traversable(0, 0) is True
    assert env.is_traversable(1, 1) is False
    assert env.is_traversable(-1, 0) is False
    assert env.is_traversable(3, 3) is False

def test_neighbors_and_wait_action():
    raw_map = [[0, 0], [0, 0]]
    env = WarehouseGrid(raw_map, height=2, width=2)
    neighbors = env.get_neighbors(0, 0)
    
    assert (0, 0) in neighbors
    assert (0, 1) in neighbors
    assert (1, 0) in neighbors
    assert len(neighbors) == 3

def test_blocked_aisle_disruption():
    raw_map = [[0, 0, 0], [0, 0, 0]]
    env = WarehouseGrid(raw_map, height=2, width=3)
    assert env.is_traversable(1, 0) is True
    
    env.block_aisle([(1, 0)])
    assert env.is_traversable(1, 0) is False
    
    env.clear_disruptions()
    assert env.is_traversable(1, 0) is True

def test_custom_narrow_aisle_layout_generation():
    env = build_custom_narrow_aisle_warehouse(rows=35, cols=50)
    assert env.height == 35
    assert env.width == 50
    assert len(env.shelves) > 0
    assert len(env.packing_stations) > 0
