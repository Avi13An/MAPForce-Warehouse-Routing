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
    env = load_movingai_map("data/maps/warehouse-65x46-handbuilt.map")
    assert env.height == 46
    assert env.width == 65
    assert len(env.shelves) > 0
    assert len(env.packing_stations) > 0


def test_real_movingai_map_loads_without_border_pick_spots():
    """The real warehouse-10-20-10-2-1 map (161 x 63) is fenced by a wall.
    Cells along that wall must not become pick spots (28 Oct)."""
    env = load_movingai_map("data/maps/warehouse-10-20-10-2-1.map")
    assert (env.width, env.height) == (161, 63)
    free = sum(env.is_traversable(x, y)
               for y in range(env.height) for x in range(env.width))
    assert free == 5699
    assert len(env.packing_stations) == 40
    assert all(env.is_traversable(x, y) for x, y in env.shelves)
    # every pick spot is beside a shelf block, not just beside the outer wall
    for x, y in env.shelves:
        assert any(
            0 < x + dx < env.width - 1 and 0 < y + dy < env.height - 1
            and env.static_grid[y + dy][x + dx] == 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    # the side columns next to the wall have no shelves beside them, so no
    # pick spots (the old loader put one on every cell there)
    assert not any(x in (1, env.width - 2) for x, _ in env.shelves)
