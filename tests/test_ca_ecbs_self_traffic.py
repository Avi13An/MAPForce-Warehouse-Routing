"""Regression: CA-ECBS must not treat a robot's own route as traffic.

Until 28 Oct the planned-occupancy term counted every robot's route,
including the one being planned. A robot's shortest path therefore always
looked busier than any detour, and CA-ECBS spent its whole w-budget avoiding
itself; on the custom layout it completed about half as many orders as ECBS.
"""

from src.planners.ca_ecbs import ca_ecbs
from src.planners.common import SimpleGrid, path_cost
from src.planners.congestion import CongestionMap

OPEN = SimpleGrid([".....", ".....", "....."])


def _no_demand_map(grid):
    # No shelves -> no demand traffic; only planned occupancy remains.
    return CongestionMap(grid, shelves=[], stations=[])


def test_score_for_subtracts_only_the_agents_own_route():
    cong = _no_demand_map(OPEN)
    a = [(0, 0), (1, 0), (2, 0)]
    b = [(2, 1), (2, 0), (3, 0)]
    cong.update_planned({0: a, 1: b})
    # (1,0) is only on agent 0's route: busy for agent 1, free for agent 0.
    assert cong.score((1, 0)) > 0
    assert cong.score_for(0, (1, 0)) == 0.0
    assert cong.score_for(1, (1, 0)) == cong.score((1, 0))
    # (2,0) is on both: each agent sees just the other one.
    assert cong.score_for(0, (2, 0)) == cong.score((2, 0)) / 2


def test_a_lone_robot_sees_no_planned_traffic_at_all():
    cong = _no_demand_map(OPEN)
    path = [(0, 0), (1, 0), (2, 0), (2, 1)]
    cong.update_planned({7: path})
    assert all(cong.score_for(7, c) == 0.0 for c in path)


def test_reporting_score_is_unchanged():
    """`score` / `solution_score` still measure total traffic (exposure
    metric, heatmap) -- only planning uses the per-agent view."""
    cong = _no_demand_map(OPEN)
    cong.update_planned({0: [(0, 0), (1, 0)], 1: [(1, 1), (1, 0)]})
    assert cong.score((1, 0)) == 1.0


def test_lone_robot_takes_its_shortest_path():
    """One robot, no demand traffic, an open grid. Nothing is congested, so
    CA-ECBS must return the optimal path. The old version took a cost-6
    detour (within w = 1.5 of the optimum 4) to avoid its own route."""
    cong = _no_demand_map(OPEN)
    res = ca_ecbs(OPEN, {0: (0, 0)}, {0: (4, 0)}, cong, w=1.5, lam=1.0)
    assert res.solved
    assert path_cost(res.solution[0]) == 4


def test_two_robots_on_separate_rows_both_go_straight():
    cong = _no_demand_map(OPEN)
    res = ca_ecbs(OPEN, {0: (0, 0), 1: (0, 2)}, {0: (4, 0), 1: (4, 2)}, cong,
                  w=1.5, lam=1.0)
    assert res.solved
    assert res.cost == 8
