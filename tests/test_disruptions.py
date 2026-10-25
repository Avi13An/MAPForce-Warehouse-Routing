"""Gates for blocked-aisle disruptions in the lifelong loop (Module 6)."""

import random

import pytest

from src.demand.orders import OrderStream, build_demand_model
from src.environment.events import TimedBlockedAisleEvent
from src.experiments.runner import make_planners
from src.lifelong.disruptions import (
    BlockedGridView,
    Reachability,
    active_blocked_cells,
    random_aisle_blockages,
)
from src.lifelong.simulator import (
    Robot,
    clear_claimed_cells,
    dock_cells,
    invalidate_cut_off_goals,
    run_lifelong,
)
from src.demand.orders import Task
from src.planners.common import SimpleGrid
from src.planners.congestion import CongestionMap
from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations

GRID = narrow_aisle_grid(rows=11, cols=15)
SHELVES, STATIONS = shelves_and_stations(GRID)
DOCKS = dock_cells(GRID, STATIONS)


def _setup(skew="extreme", n=8, seed=2):
    m = build_demand_model(SHELVES, STATIONS, skew=skew, seed=0)
    cong = CongestionMap(GRID, SHELVES, STATIONS, m.shelf_weights)
    planners = make_planners(cong, time_limit=0.4, window=6)
    free = GRID.free_cells()
    random.Random(seed).shuffle(free)
    return m, cong, planners, free[:n]


# ------------------------------------------------------------ the schedule
def test_active_cells_use_a_half_open_window():
    e = TimedBlockedAisleEvent(5, 10, [(2, 2)])
    assert active_blocked_cells([e], 4) == set()
    assert active_blocked_cells([e], 5) == {(2, 2)}
    assert active_blocked_cells([e], 14) == {(2, 2)}
    assert active_blocked_cells([e], 15) == set()


def test_overlapping_events_do_not_clear_each_other():
    """DisruptionManager.update removes a shared cell when the first event
    ends; the schedule-based view keeps it closed until the last one ends."""
    a = TimedBlockedAisleEvent(0, 5, [(1, 1), (2, 1)])
    b = TimedBlockedAisleEvent(3, 10, [(2, 1)])
    assert active_blocked_cells([a, b], 6) == {(2, 1)}
    assert active_blocked_cells([a, b], 13) == set()


def test_skipped_timesteps_do_not_miss_an_event():
    """Querying only every 4th step still sees an event that started between."""
    e = TimedBlockedAisleEvent(5, 3, [(0, 0)])
    seen = [t for t in range(0, 20, 4) if active_blocked_cells([e], t)]
    assert seen == []           # 5,6,7 are all skipped by 0,4,8 ...
    e2 = TimedBlockedAisleEvent(5, 4, [(0, 0)])
    seen2 = [t for t in range(0, 20, 4) if active_blocked_cells([e2], t)]
    assert seen2 == [8]         # ... but anything overlapping a query is found


# ---------------------------------------------------------------- the view
def test_blocked_view_hides_closed_cells_only():
    v = BlockedGridView(GRID, {(2, 0)})
    assert not v.passable((2, 0))
    assert v.passable((1, 0)) and v.passable((3, 0))
    assert (2, 0) not in v.neighbours((1, 0))
    assert v.free_cells() == GRID.free_cells()   # falls through to the base


def test_reachability_sees_a_cut():
    g = SimpleGrid(["....."])
    r = Reachability(BlockedGridView(g, {(2, 0)}))
    assert r.connected((0, 0), (1, 0))
    assert not r.connected((0, 0), (4, 0))
    assert not r.connected((0, 0), (2, 0))       # a closed cell is no goal


def test_random_blockages_are_valid_and_seeded():
    protect = set(DOCKS)
    ev = random_aisle_blockages(GRID, horizon=100, n_events=6, length=3,
                                duration=20, seed=7, protect=protect)
    assert len(ev) == 6
    for e in ev:
        assert len(e.blocked_cells) == 3
        assert all(GRID.passable(c) and c not in protect for c in e.blocked_cells)
        assert 0 <= e.trigger_timestep <= 80
    again = random_aisle_blockages(GRID, horizon=100, n_events=6, length=3,
                                   duration=20, seed=7, protect=protect)
    assert [(e.trigger_timestep, e.blocked_cells) for e in ev] == \
           [(e.trigger_timestep, e.blocked_cells) for e in again]


# --------------------------------------------------------- goal handling
def test_cut_off_shelf_task_is_handed_back():
    g = SimpleGrid(["....."])
    task = Task(task_id=0, shelf=(4, 0), station=(0, 0), released_at=0)
    robot = Robot(rid=0, pos=(0, 0), task=task)
    pending = []
    reach = Reachability(BlockedGridView(g, {(2, 0)}))
    assert invalidate_cut_off_goals([robot], pending, [(0, 0)], reach) == 1
    assert robot.task is None and pending == [task]


def test_carrying_robot_with_no_reachable_dock_waits_and_releases_it():
    g = SimpleGrid(["....."])
    task = Task(task_id=0, shelf=(4, 0), station=(0, 0), released_at=0)
    robot = Robot(rid=0, pos=(4, 0), task=task, carrying=True, dock=(0, 0))
    reach = Reachability(BlockedGridView(g, {(2, 0)}))
    assert invalidate_cut_off_goals([robot], [], [(0, 0), (1, 0)], reach) == 0
    assert robot.waiting and robot.goal == (4, 0)
    assert robot.dock is None      # claim released, so nobody else is misled
    assert robot.task is task      # still carrying it


# ------------------------------------------------------- the whole loop
def test_events_outside_the_run_change_nothing():
    """The feature must be invisible when no event is active."""
    m, cong, planners, starts = _setup()
    base = run_lifelong(GRID, planners["prioritized"], OrderStream(m, 0.5, 0),
                        starts, horizon=40, seed=1, congestion=cong)
    m2, cong2, planners2, _ = _setup()
    late = [TimedBlockedAisleEvent(500, 10, [(2, 0)])]
    with_ev = run_lifelong(GRID, planners2["prioritized"], OrderStream(m2, 0.5, 0),
                           starts, horizon=40, seed=1, congestion=cong2,
                           disruptions=late)
    a, b = base.as_row(), with_ev.as_row()
    a.pop("mean_plan_time"), b.pop("mean_plan_time")   # wall-clock varies
    assert a == b


@pytest.mark.parametrize("delay", [0.0, 0.1])
@pytest.mark.parametrize("planner_name", ["prioritized", "ECBS", "CA-ECBS"])
def test_no_robot_ever_enters_a_closed_cell(planner_name, delay):
    m, cong, planners, starts = _setup()
    events = random_aisle_blockages(GRID, horizon=60, n_events=5, length=3,
                                    duration=15, seed=3, protect=DOCKS)
    trace = []
    run_lifelong(GRID, planners[planner_name], OrderStream(m, 0.5, 0), starts,
                 horizon=60, seed=4, delay_prob=delay, congestion=cong,
                 disruptions=events,
                 on_step=lambda t, pos, closed: trace.append((pos, closed)))
    prev = {i: c for i, c in enumerate(starts)}
    for t, (pos, closed) in enumerate(trace):
        for rid, cell in pos.items():
            if cell != prev[rid]:
                assert cell not in closed, f"robot {rid} entered {cell} at t={t}"
        assert len(set(pos.values())) == len(pos)
        prev = pos


def test_planner_is_shown_the_closed_cells():
    m, cong, planners, starts = _setup(n=4)
    closed = [(2, 0), (3, 0), (4, 0)]
    events = [TimedBlockedAisleEvent(0, 30, closed)]
    seen = []

    def spy(grid, s, g):
        seen.append({c: grid.passable(c) for c in closed if c not in s.values()})
        return planners["prioritized"](grid, s, g)

    run_lifelong(GRID, spy, OrderStream(m, 0.5, 0), starts, horizon=20,
                 seed=0, congestion=cong, disruptions=events)
    assert seen and all(not any(v.values()) for v in seen)


def test_closing_an_aisle_triggers_an_immediate_replan():
    m, cong, planners, starts = _setup()
    quiet = run_lifelong(GRID, planners["prioritized"], OrderStream(m, 0.5, 0),
                         starts, horizon=30, replan_every=5, seed=0,
                         congestion=cong)
    # Starts at t=7 and ends at t=17: neither is a multiple of 5.
    events = [TimedBlockedAisleEvent(7, 10, [(6, 2), (7, 2), (8, 2)])]
    m2, cong2, planners2, _ = _setup()
    busy = run_lifelong(GRID, planners2["prioritized"], OrderStream(m2, 0.5, 0),
                        starts, horizon=30, replan_every=5, seed=0,
                        congestion=cong2, disruptions=events)
    assert busy.disruption_replans >= 2          # closing and reopening
    assert busy.planning_calls >= quiet.planning_calls + 2


@pytest.mark.parametrize("planner_name", ["prioritized", "CBS", "ECBS", "CA-ECBS"])
def test_planner_is_never_handed_an_impossible_instance(planner_name):
    """Heavy disruption: every planning call gets distinct goals, each reachable
    from its robot's start. A planner may still time out or get boxed in --
    that is a measured result -- but never because the simulator asked for the
    impossible."""
    m, cong, planners, starts = _setup()
    events = random_aisle_blockages(GRID, horizon=80, n_events=10, length=3,
                                    duration=25, seed=11, protect=DOCKS)
    bad = []

    def spy(grid, s, g):
        reach = Reachability(grid)
        if len(set(g.values())) != len(g):
            bad.append("duplicate goals")
        bad.extend(f"robot {a} cannot reach {g[a]}" for a in s
                   if not reach.connected(s[a], g[a]))
        return planners[planner_name](grid, s, g)

    res = run_lifelong(GRID, spy, OrderStream(m, 0.5, 0), starts, horizon=80,
                       seed=0, congestion=cong, disruptions=events)
    assert bad == []
    assert res.completed_orders > 0
    row = res.as_row()
    for k in ("disruption_replans", "disruption_stalls", "tasks_deferred"):
        assert k in row


def test_replan_at_adds_exactly_the_requested_replans():
    """The stress test's control run relies on this to match the disrupted
    run's replanning moments."""
    m, cong, planners, starts = _setup(n=4)
    calls = []

    def count(grid, s, g):
        calls.append(1)
        return planners["prioritized"](grid, s, g)

    run_lifelong(GRID, count, OrderStream(m, 0.5, 0), starts, horizon=20,
                 replan_every=5, seed=0, congestion=cong)
    plain = len(calls)                            # t = 0, 5, 10, 15
    calls.clear()
    run_lifelong(GRID, count, OrderStream(m, 0.5, 0), starts, horizon=20,
                 replan_every=5, seed=0, congestion=cong, replan_at=[3, 5, 12])
    assert plain == 4 and len(calls) == 6         # 3 and 12 added; 5 already one


def test_a_waiting_robot_does_not_deliver_where_it_waits():
    """A carrying robot cut off from every drop-off must keep its order until
    it actually reaches one. One robot, one closed cell between it and the
    only station, closed for the whole run: nothing may be delivered."""
    g = SimpleGrid(["......."])
    from src.demand.orders import DemandModel
    demand = DemandModel({(6, 0): 1.0}, [(0, 0)], "test", "random", "uniform")
    events = [TimedBlockedAisleEvent(3, 100, [(3, 0)])]

    def straight(grid, s, goals):
        from src.planners.low_level import independent_astar
        from src.planners.common import PlanResult
        return PlanResult(independent_astar(grid, s, goals), 0, 0, 0, 0.0, True)

    res = run_lifelong(g, straight, OrderStream(demand, rate=0.2, seed=0),
                       [(5, 0)], horizon=40, disruptions=events)
    assert res.completed_orders == 0


def test_idle_robot_steps_off_another_robots_goal():
    g = SimpleGrid(["....."])
    task = Task(task_id=0, shelf=(2, 0), station=(0, 0), released_at=0)
    worker = Robot(rid=0, pos=(0, 0), task=task)
    idle = Robot(rid=1, pos=(2, 0))               # standing on the worker's shelf
    clear_claimed_cells([worker, idle], g)
    assert idle.park is not None and idle.park != (2, 0)
    assert idle.goal != worker.goal
    idle.pos = idle.park                          # it gets there ...
    clear_claimed_cells([worker, idle], g)
    assert idle.park is None and idle.goal == idle.pos   # ... and stays idle
