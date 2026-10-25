"""Goal sampling in the one-shot lambda / w sweep (added 28 Oct)."""

import random

from src.tuning.sweep import narrow_aisle_grid, sample_instance, shelves_and_stations

GRID = narrow_aisle_grid(rows=11, cols=15)
SHELVES, STATIONS = shelves_and_stations(GRID)


def test_uniform_sampling_is_unchanged_by_the_new_argument():
    """The reported sweeps used uniform goals; omitting `goal_weights` must
    give exactly the instances the old code gave."""
    a = sample_instance(GRID, SHELVES, STATIONS, 6, random.Random(3))
    b = sample_instance(GRID, SHELVES, STATIONS, 6, random.Random(3), goal_weights=None)
    assert a == b


def test_demand_sampling_sends_robots_to_the_popular_shelves():
    hot = SHELVES[:4]
    weights = {s: (1.0 if s in hot else 0.0) for s in SHELVES}
    _, goals = sample_instance(GRID, SHELVES, STATIONS, 4, random.Random(0),
                               goal_weights=weights)
    assert set(goals.values()) == set(hot)          # distinct goals, all hot
