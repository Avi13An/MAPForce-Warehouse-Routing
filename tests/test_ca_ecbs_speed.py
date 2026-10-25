"""The 28 Oct CA-ECBS speed-up must not change a single decision.

`make_conflict_counter` replaces a per-node loop over every other path with
table lookups; the occupancy seeding now reuses ecbs()'s distance tables.
Both are pure optimisations: same values, same plans, same expansions.
"""

import random

import pytest

import src.planners.ca_ecbs as ca_module
from src.demand.orders import OrderStream, build_demand_model
from src.experiments.runner import make_planners
from src.lifelong.simulator import run_lifelong
from src.planners.ca_ecbs import ca_ecbs
from src.planners.congestion import CongestionMap
from src.planners.low_level import _conflict_penalty, make_conflict_counter
from src.tuning.sweep import narrow_aisle_grid, sample_instance, shelves_and_stations

GRID = narrow_aisle_grid(rows=11, cols=15)
SHELVES, STATIONS = shelves_and_stations(GRID)


def _random_path(rng, length):
    cells = [(rng.randrange(4), rng.randrange(4))]
    for _ in range(length - 1):
        x, y = cells[-1]
        dx, dy = rng.choice(((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.append((min(3, max(0, x + dx)), min(3, max(0, y + dy))))
    return cells


def test_counter_equals_the_original_everywhere():
    rng = random.Random(0)
    for _ in range(300):
        others = [_random_path(rng, rng.randrange(0, 9)) for _ in range(rng.randrange(0, 7))]
        fast = make_conflict_counter(others)
        for _ in range(40):
            cell = (rng.randrange(4), rng.randrange(4))
            prev = rng.choice([cell, (rng.randrange(4), rng.randrange(4))])
            t = rng.randrange(0, 12)               # includes t past every path end
            assert fast(cell, t, prev) == _conflict_penalty(cell, t, prev, others)


def _slow_counter(others):
    return lambda cell, t, prev: _conflict_penalty(cell, t, prev, others)


@pytest.mark.parametrize("seed", range(6))
def test_ca_ecbs_plans_are_identical_with_the_old_counter(seed, monkeypatch):
    starts, goals = sample_instance(GRID, SHELVES, STATIONS, 8, random.Random(seed))
    demand = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=seed)

    def solve():
        cong = CongestionMap(GRID, SHELVES, STATIONS, demand.shelf_weights)
        return ca_ecbs(GRID, starts, goals, cong, w=1.5, lam=1.0, time_limit=60.0)

    fast = solve()
    monkeypatch.setattr(ca_module, "make_conflict_counter", _slow_counter)
    slow = solve()
    assert (fast.solved, fast.cost, fast.expansions) == (slow.solved, slow.cost, slow.expansions)
    assert fast.solution == slow.solution


def test_lifelong_run_is_identical_with_the_old_counter(monkeypatch):
    demand = build_demand_model(SHELVES, STATIONS, skew="real", seed=1)
    free = GRID.free_cells()
    random.Random(7).shuffle(free)

    def run():
        cong = CongestionMap(GRID, SHELVES, STATIONS, demand.shelf_weights)
        planner = make_planners(cong, time_limit=60.0, window=6)["CA-ECBS"]
        row = run_lifelong(GRID, planner, OrderStream(demand, 0.5, 1), free[:10],
                           horizon=50, delay_prob=0.05, seed=1, congestion=cong).as_row()
        row.pop("mean_plan_time")
        return row

    fast = run()
    monkeypatch.setattr(ca_module, "make_conflict_counter", _slow_counter)
    assert run() == fast
