import pytest
from src.environment.grid import WarehouseGrid
from src.environment.map_loader import load_movingai_map
from src.environment.layouts import build_custom_narrow_aisle_warehouse
from src.environment.events import DisruptionManager

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

def test_dynamic_disruption_manager():
    raw_map = [[0, 0, 0], [0, 0, 0]]
    env = WarehouseGrid(raw_map, height=2, width=3)
    dm = DisruptionManager(env)
    
    dm.schedule_event(trigger_timestep=2, duration=3, blocked_cells=[(1, 0)])
    
    dm.update(1)
    assert env.is_traversable(1, 0) is True
    
    dm.update(2)
    assert env.is_traversable(1, 0) is False
    
    dm.update(5)
    assert env.is_traversable(1, 0) is True

def test_movingai_loading_with_indexing():
    env = load_movingai_map("data/maps/warehouse-10-20-10-2-1.map")
    assert env.height == 46
    assert env.width == 65
    assert len(env.shelves) > 0
    assert len(env.packing_stations) > 0