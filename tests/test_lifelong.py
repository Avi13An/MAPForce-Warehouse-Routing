"""Gates for the demand model and the lifelong execution loop."""

import random

import pytest

from src.demand.orders import (
    OrderStream,
    build_demand_model,
    synthetic_frequencies,
)
from src.experiments.runner import make_planners, wilcoxon_signed_rank
from src.lifelong.simulator import (
    Robot,
    assign_tasks,
    count_execution_collisions,
    dock_cells,
    run_lifelong,
)
from src.planners.congestion import CongestionMap
from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations

GRID = narrow_aisle_grid(rows=11, cols=15)
SHELVES, STATIONS = shelves_and_stations(GRID)


# ------------------------------------------------------------------- demand
def test_demand_weights_are_normalised():
    for skew in ("uniform", "real", "extreme"):
        m = build_demand_model(SHELVES, STATIONS, skew=skew, seed=0)
        assert abs(sum(m.shelf_weights.values()) - 1.0) < 1e-6


def test_skew_levels_actually_differ():
    """The control and the treatment must not be the same distribution."""
    uni = build_demand_model(SHELVES, STATIONS, skew="uniform", seed=0)
    ext = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=0)
    spread = lambda m: max(m.shelf_weights.values()) - min(m.shelf_weights.values())
    assert spread(ext) > spread(uni)


def test_provenance_is_recorded():
    """No figure in the report may be unattributable to its data source."""
    m = build_demand_model(SHELVES, STATIONS, skew="real", seed=0)
    assert m.source in ("instacart", "synthetic")
    assert m.skew in m.provenance()


def test_synthetic_frequencies_are_heavy_tailed():
    f = synthetic_frequencies(n_products=100)
    ranked = sorted(f.values(), reverse=True)
    assert ranked[0] > 10 * ranked[-1]


def test_order_stream_rate_is_respected():
    m = build_demand_model(SHELVES, STATIONS, skew="real", seed=0)
    stream = OrderStream(m, rate=0.5, seed=0)
    released = sum(len(stream.release(t)) for t in range(100))
    assert 45 <= released <= 55


# ----------------------------------------------------------------- lifelong
def test_docks_are_distinct_and_passable():
    docks = dock_cells(GRID, STATIONS)
    assert len(docks) == len(set(docks))
    assert all(GRID.passable(d) for d in docks)
    assert len(docks) > len(STATIONS)       # each station has a queue


def test_assignment_never_duplicates_a_goal():
    """Two robots sharing a goal cell makes the MAPF window unsolvable."""
    m = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=0)
    stream = OrderStream(m, rate=5.0, seed=0)
    free = GRID.free_cells()
    random.Random(1).shuffle(free)
    robots = [Robot(rid=i, pos=free[i]) for i in range(8)]
    pending = []
    for t in range(5):
        pending.extend(stream.release(t))
        assign_tasks(robots, pending, t)
        goals = [r.goal for r in robots]
        assert len(goals) == len(set(goals)), f"duplicate goal at t={t}"


def test_assignment_keeps_a_claim_shared_with_a_standing_robot():
    """Regression: an idle robot standing on another robot's shelf takes a
    new order. Un-claiming the cell it stood on used to free that shelf, and a
    third robot was sent to it -- two robots, one goal, unsolvable window."""
    from src.demand.orders import Task
    heading = Robot(rid=0, pos=(0, 0),
                    task=Task(task_id=0, shelf=(4, 0), station=(0, 0), released_at=0))
    standing = Robot(rid=1, pos=(4, 0))           # idle, on robot 0's shelf
    other = Robot(rid=2, pos=(2, 0))
    pending = [Task(task_id=1, shelf=(3, 0), station=(0, 0), released_at=0),
               Task(task_id=2, shelf=(4, 0), station=(0, 0), released_at=0)]
    assign_tasks([heading, standing, other], pending, 0)
    goals = [r.goal for r in (heading, standing, other)]
    assert len(goals) == len(set(goals)), goals


def test_collision_counter_detects_both_kinds():
    before = {0: (0, 0), 1: (1, 0)}
    assert count_execution_collisions(before, {0: (1, 0), 1: (1, 0)}) == 1  # vertex
    assert count_execution_collisions(before, {0: (1, 0), 1: (0, 0)}) == 1  # swap
    assert count_execution_collisions(before, {0: (0, 1), 1: (1, 1)}) == 0


@pytest.mark.parametrize("delay", [0.0, 0.2])
@pytest.mark.parametrize("planner_name", ["prioritized", "ECBS", "CA-ECBS"])
def test_robots_never_share_a_cell_during_execution(planner_name, delay):
    """The simulator's core invariant.

    Two robots on one square is physically impossible and leaves the next
    planning window with no solution at all. A high delay probability is used
    here deliberately, because drift is what breaks naive resolution.
    `run_lifelong` asserts this internally every timestep; this test is what
    exercises it.
    """
    m = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=0)
    cong = CongestionMap(GRID, SHELVES, STATIONS, m.shelf_weights)
    planners = make_planners(cong, time_limit=0.4, window=6)
    free = GRID.free_cells()
    random.Random(2).shuffle(free)
    stream = OrderStream(m, rate=0.5, seed=0)
    result = run_lifelong(GRID, planners[planner_name], stream, free[:10],
                          horizon=40, replan_every=5, delay_prob=delay,
                          seed=3, congestion=cong)
    assert result.timesteps == 40
    assert result.planning_calls > 0


def test_planner_failure_is_recorded_not_raised():
    """Section 4.4: robots hold position and the failure becomes a datum."""
    from src.planners.common import PlanResult

    def always_fails(grid, starts, goals):
        return PlanResult(None, 0, 0, 0, 0.0, False, "forced failure")

    m = build_demand_model(SHELVES, STATIONS, skew="uniform", seed=0)
    stream = OrderStream(m, rate=0.5, seed=0)
    free = GRID.free_cells()
    res = run_lifelong(GRID, always_fails, stream, free[:5], horizon=20)
    assert res.planning_failures == res.planning_calls
    assert res.success_rate == 0.0
    assert res.completed_orders == 0          # nobody moved, nothing delivered
    assert res.collisions == 0                # holding position is safe


def test_throughput_is_orders_per_thousand_timesteps():
    m = build_demand_model(SHELVES, STATIONS, skew="uniform", seed=0)
    cong = CongestionMap(GRID, SHELVES, STATIONS, m.shelf_weights)
    planners = make_planners(cong, time_limit=0.4, window=6)
    free = GRID.free_cells()
    random.Random(4).shuffle(free)
    stream = OrderStream(m, rate=0.5, seed=0)
    res = run_lifelong(GRID, planners["prioritized"], stream, free[:6],
                       horizon=100, seed=0, congestion=cong)
    assert abs(res.throughput - 1000.0 * res.completed_orders / 100) < 1e-6


def test_collision_rate_is_per_thousand_robot_timesteps():
    """Section 4.5: collisions per 1000 robot-timesteps, not a raw count.

    Independent A* is used because it ignores the other robots and so is
    guaranteed to produce collisions -- a zero count would make the check
    vacuous.
    """
    m = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=0)
    cong = CongestionMap(GRID, SHELVES, STATIONS, m.shelf_weights)
    planners = make_planners(cong, time_limit=0.4, window=6)
    free = GRID.free_cells()
    random.Random(5).shuffle(free)
    n_robots, horizon = 10, 60
    stream = OrderStream(m, rate=0.5, seed=0)
    res = run_lifelong(GRID, planners["independent_astar"], stream,
                       free[:n_robots], horizon=horizon, seed=0, congestion=cong)
    assert res.collisions > 0
    expected = 1000.0 * res.collisions / (n_robots * horizon)
    assert abs(res.collision_rate - expected) < 1e-9
    assert res.as_row()["collision_rate"] == round(expected, 3)


# --------------------------------------------------------------- statistics
def test_wilcoxon_matches_a_known_case():
    """Identical samples give p = 1; a consistent shift gives a small p."""
    a = [1, 2, 3, 4, 5]
    assert wilcoxon_signed_rank(a, a)[1] == 1.0
    big = list(range(1, 21))
    shifted = [v + 5 for v in big]
    _, p = wilcoxon_signed_rank(shifted, big)
    assert p < 0.01


def test_wilcoxon_drops_zero_differences():
    _, p = wilcoxon_signed_rank([1, 1, 1], [1, 1, 1])
    assert p == 1.0


def test_path_cost_ratio_is_one_for_a_lone_robot_and_never_below_one():
    """path_cost_ratio = path cost / sum of shortest distances (28 Oct). A lone
    robot always takes a shortest path, so its ratio is exactly 1; with a
    fleet it can only be >= 1, because the denominator is a lower bound."""
    from src.experiments.runner import make_planners
    from src.planners.congestion import CongestionMap
    from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations
    g = narrow_aisle_grid(rows=11, cols=15)
    shelves, stations = shelves_and_stations(g)
    demand = build_demand_model(shelves, stations, skew="real", seed=0)
    cong = CongestionMap(g, shelves, stations, demand.shelf_weights)
    free = g.free_cells()
    random.Random(4).shuffle(free)
    for n, planner in ((1, "CBS"), (6, "ECBS"), (6, "CA-ECBS"), (6, "prioritized")):
        cong.clear_planned()
        p = make_planners(cong, time_limit=5.0, window=6)[planner]
        res = run_lifelong(g, p, OrderStream(demand, 0.5, 0), free[:n],
                           horizon=40, seed=0, congestion=cong)
        assert res.mean_path_cost > 0
        if n == 1:
            assert res.path_cost_ratio == 1.0
        else:
            assert res.path_cost_ratio >= 1.0
