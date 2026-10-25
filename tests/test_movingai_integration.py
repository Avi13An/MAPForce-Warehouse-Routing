"""Gates for running the planner stack and experiments on Avi's WarehouseGrid.

Regression for 28 Oct: `python -m src.experiments.runner` crashed with
AttributeError ('WarehouseGridView' object has no attribute 'height') as soon
as it reached the MovingAI layout, so no full experiment had ever completed.
"""

import random

from src.demand.orders import OrderStream, build_demand_model
from src.environment.grid import WarehouseGrid
from src.environment.map_loader import load_movingai_map
from src.experiments import runner
from src.experiments.runner import make_planners, run_experiments
from src.lifelong.simulator import run_lifelong
from src.planners.common import WarehouseGridView
from src.planners.congestion import CongestionMap
from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations


def _small_warehouse() -> WarehouseGridView:
    """A 9x5 WarehouseGrid with two shelf blocks, indexed like Avi's loader."""
    rows = [
        ".........",
        ".##...##.",
        ".##...##.",
        ".........",
        ".........",
    ]
    grid = [[1 if ch == "#" else 0 for ch in r] for r in rows]
    env = WarehouseGrid(grid, height=5, width=9)
    for y in range(5):
        for x in range(9):
            if grid[y][x] == 0 and any(
                0 <= x + dx < 9 and 0 <= y + dy < 5 and grid[y + dy][x + dx] == 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                env.register_shelf(x, y)
    for x in (0, 4, 8):
        env.register_packing_station(x, 4)
    return WarehouseGridView(env)


def test_view_exposes_size_and_free_cells():
    env = load_movingai_map("data/maps/warehouse-65x46-handbuilt.map")
    view = WarehouseGridView(env)
    assert (view.width, view.height) == (65, 46)
    free = view.free_cells()
    assert len(free) == len(set(free))
    assert all(env.is_traversable(x, y) for x, y in free)
    expected = sum(env.is_traversable(x, y)
                   for y in range(env.height) for x in range(env.width))
    assert len(free) == expected


def test_movingai_uses_avis_indexed_stations_and_pick_spots():
    env = load_movingai_map("data/maps/warehouse-65x46-handbuilt.map")
    view = WarehouseGridView(env)
    shelves, stations = shelves_and_stations(view)
    assert set(stations) == set(env.packing_stations)   # 16, not 4 corners
    assert not set(shelves) & set(stations)             # never pick at a drop-off
    assert set(shelves) <= set(env.shelves)
    assert all(view.passable(c) for c in shelves + stations)


def test_synthetic_layout_still_uses_the_generic_rule():
    g = narrow_aisle_grid(rows=11, cols=15)
    _, stations = shelves_and_stations(g)
    assert stations == [(0, 0), (14, 0), (0, 10), (14, 10)]


def test_runner_completes_on_a_warehouse_grid(monkeypatch):
    """The exact path that crashed: run_experiments over a WarehouseGridView."""
    monkeypatch.setattr(runner, "load_layouts",
                        lambda *a, **k: {"warehouse": _small_warehouse()})
    rows = run_experiments(robot_counts=(3,), skews=("uniform", "extreme"),
                           delays=(0.0,), seeds=(0,), horizon=15,
                           planner_time_limit=1.0, verbose=False)
    assert len(rows) == 2 * 5                   # 2 skews x 5 planners
    assert {r["planner"] for r in rows} == {
        "independent_astar", "prioritized", "CBS", "ECBS", "CA-ECBS"}
    assert all(r["layout"] == "warehouse" for r in rows)


def test_custom_layout_pick_spots_are_standable():
    """Regression: the custom layout registered its shelf *walls* as pick
    spots, so no pick task on it could ever be reached."""
    from src.environment.layouts import build_custom_narrow_aisle_warehouse
    env = build_custom_narrow_aisle_warehouse()
    assert env.shelves
    assert all(env.is_traversable(x, y) for x, y in env.shelves)
    # every pick spot really is beside a shelf wall
    for x, y in env.shelves:
        assert any(env.in_bounds(x + dx, y + dy) and env.static_grid[y + dy][x + dx] == 1
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    view = WarehouseGridView(env)
    shelves, stations = shelves_and_stations(view)
    assert len(shelves) == len(env.shelves) and len(stations) == 14


def test_runner_uses_avis_layouts_not_the_synthetic_one():
    layouts = runner.load_layouts(custom_size=(17, 23))
    # default = the two layouts the reported results were made on
    assert set(layouts) == {"custom_narrow_aisle", "warehouse_65x46"}
    assert all(isinstance(g, WarehouseGridView) for g in layouts.values())
    real = runner.load_layouts(names=("movingai_warehouse",))["movingai_warehouse"]
    assert (real.width, real.height) == (161, 63)


def test_hand_built_map_is_not_called_movingai():
    """Until 28 Oct the 65 x 46 hand-built map was loaded as
    "movingai_warehouse". That name now means the real benchmark map only."""
    assert runner.MAP_FILES["warehouse_65x46"].endswith("warehouse-65x46-handbuilt.map")
    assert runner.MAP_FILES["movingai_warehouse"].endswith("warehouse-10-20-10-2-1.map")
    env = load_movingai_map(runner.MAP_FILES["movingai_warehouse"])
    assert (env.width, env.height) == (161, 63)


def test_reused_congestion_map_matches_a_fresh_one():
    """The runner now builds the congestion map once and calls clear_planned()
    between runs. That must give exactly what a freshly built map gives."""
    g = narrow_aisle_grid(rows=11, cols=15)
    shelves, stations = shelves_and_stations(g)
    demand = build_demand_model(shelves, stations, skew="extreme", seed=0)
    free = g.free_cells()
    random.Random(3).shuffle(free)
    starts = free[:6]

    def run(cong):
        planner = make_planners(cong, time_limit=5.0, window=6)["CA-ECBS"]
        row = run_lifelong(g, planner, OrderStream(demand, 0.5, 0), starts,
                           horizon=40, seed=0, congestion=cong).as_row()
        row.pop("mean_plan_time")                # wall-clock
        return row

    fresh = run(CongestionMap(g, shelves, stations, demand.shelf_weights))
    reused = CongestionMap(g, shelves, stations, demand.shelf_weights)
    run(reused)                                  # dirty it with a full run
    reused.clear_planned()
    assert run(reused) == fresh
