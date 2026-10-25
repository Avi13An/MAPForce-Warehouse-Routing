"""Gates for ECBS and CA-ECBS -- the suboptimality bound and the lambda = 0 identity.

These two tests are the ones the report's central claim depends on:

* every plan ECBS or CA-ECBS returns costs at most w times the optimum, and
* the congestion term cannot break that, for any lambda.

Both are checked against CBS's independently verified optimum, not against the
planners' own lower bounds -- a planner grading its own bound proves nothing.
"""

import pytest

from src.planners.ca_ecbs import ca_ecbs
from src.planners.cbs import cbs
from src.planners.common import SimpleGrid, solution_cost, validate_solution
from src.planners.congestion import CongestionMap, clustered_demand, uniform_demand
from src.planners.ecbs import ecbs

OPEN4 = SimpleGrid([
    "....",
    "....",
    "....",
    "....",
])

AISLES = SimpleGrid([
    ".........",
    ".#.#.#.#.",
    ".#.#.#.#.",
    ".........",
    ".#.#.#.#.",
    ".#.#.#.#.",
    ".........",
])

SCENARIOS = [
    ({0: (0, 0), 1: (3, 0)}, {0: (3, 0), 1: (0, 0)}, OPEN4),
    ({0: (0, 0), 1: (3, 3), 2: (0, 3)}, {0: (3, 3), 1: (0, 0), 2: (3, 0)}, OPEN4),
    ({0: (0, 0), 1: (8, 0), 2: (0, 6), 3: (8, 6)},
     {0: (8, 6), 1: (0, 6), 2: (8, 0), 3: (0, 0)}, AISLES),
]


def _congestion_for(grid):
    shelves = [(2, 1), (4, 1), (6, 1), (2, 4), (4, 4), (6, 4)]
    shelves = [s for s in shelves if grid.passable(s)]
    stations = [(0, 0), (grid.width - 1, grid.height - 1)]
    stations = [s for s in stations if grid.passable(s)]
    return CongestionMap(grid, shelves, stations, uniform_demand(shelves))


@pytest.mark.parametrize("starts,goals,grid", SCENARIOS)
@pytest.mark.parametrize("w", [1.0, 1.2, 1.5, 2.0])
def test_ecbs_respects_its_suboptimality_bound(starts, goals, grid, w):
    """The headline guarantee: cost <= w * optimal, measured against CBS."""
    optimal = cbs(grid, starts, goals, time_limit=20.0)
    if not optimal.solved:
        pytest.skip("CBS could not solve this instance in time")

    result = ecbs(grid, starts, goals, w=w, time_limit=20.0)
    assert result.solved, result.failure
    assert validate_solution(result.solution, grid) == []
    assert result.cost <= w * optimal.cost + 1e-9, (
        f"ECBS returned {result.cost}, bound was {w} x {optimal.cost}"
    )


@pytest.mark.parametrize("starts,goals,grid", SCENARIOS)
@pytest.mark.parametrize("w", [1.2, 1.5, 2.0])
@pytest.mark.parametrize("lam", [0.0, 0.5, 2.0, 10.0])
def test_ca_ecbs_respects_the_bound_for_every_lambda(starts, goals, grid, w, lam):
    """Congestion reorders FOCAL; it must never widen it.

    Lambda is pushed to 10 deliberately -- far past any value the sweep would
    choose -- because the claim is that *no* lambda can break the bound. If the
    congestion term had leaked into g or h instead of the focal key, this test
    is what would catch it.
    """
    optimal = cbs(grid, starts, goals, time_limit=20.0)
    if not optimal.solved:
        pytest.skip("CBS could not solve this instance in time")

    cong = _congestion_for(grid)
    result = ca_ecbs(grid, starts, goals, cong, w=w, lam=lam, time_limit=20.0)
    assert result.solved, result.failure
    assert validate_solution(result.solution, grid) == []
    assert result.cost <= w * optimal.cost + 1e-9, (
        f"CA-ECBS(lam={lam}) returned {result.cost}, bound was {w} x {optimal.cost}"
    )


@pytest.mark.parametrize("starts,goals,grid", SCENARIOS)
def test_lambda_zero_reduces_to_ecbs(starts, goals, grid):
    """At lambda = 0 the two planners must agree exactly.

    This is the structural check behind the report's claim that CA-ECBS changes
    only the focal ordering: with the congestion weight switched off, there is
    no difference left to observe.
    """
    cong = _congestion_for(grid)
    plain = ecbs(grid, starts, goals, w=1.5, time_limit=20.0)
    aware = ca_ecbs(grid, starts, goals, cong, w=1.5, lam=0.0, time_limit=20.0)
    assert plain.solved and aware.solved
    assert aware.cost == plain.cost
    assert aware.expansions == plain.expansions


def test_w_equal_one_matches_cbs_cost():
    """w = 1 collapses FOCAL to OPEN, so ECBS must return the optimal cost."""
    starts, goals, grid = SCENARIOS[0]
    optimal = cbs(grid, starts, goals)
    result = ecbs(grid, starts, goals, w=1.0)
    assert optimal.solved and result.solved
    assert result.cost == optimal.cost


def test_reported_cost_matches_returned_plan():
    starts, goals, grid = SCENARIOS[1]
    cong = _congestion_for(grid)
    result = ca_ecbs(grid, starts, goals, cong, w=1.5, lam=1.0)
    assert result.solved
    assert result.cost == solution_cost(result.solution)


def test_congestion_steers_away_from_hot_cells():
    """The hypothesis itself, as a measurement rather than an assertion.

    Under clustered demand, CA-ECBS should route less traffic through the
    busiest cells than plain ECBS does. This is the effect the project exists to
    measure, so the test records it and only fails if the congestion-aware
    planner is strictly *worse* on its own objective -- which would mean the
    focal key is wired up backwards.
    """
    grid = AISLES
    starts, goals = SCENARIOS[2][0], SCENARIOS[2][1]
    shelves = [(2, 1), (4, 1), (6, 1), (2, 4), (4, 4), (6, 4)]
    stations = [(0, 0), (8, 6)]
    demand = clustered_demand(shelves, centre=(2, 1), share=0.9)
    cong = CongestionMap(grid, shelves, stations, demand)

    plain = ecbs(grid, starts, goals, w=1.5, time_limit=20.0)
    aware = ca_ecbs(grid, starts, goals, cong, w=1.5, lam=3.0, time_limit=20.0)
    assert plain.solved and aware.solved

    plain_traffic = cong.solution_score(plain.solution)
    aware_traffic = cong.solution_score(aware.solution)
    print(f"\ncongestion exposure: ECBS={plain_traffic:.4f} CA-ECBS={aware_traffic:.4f}")
    assert aware_traffic <= plain_traffic + 1e-6, (
        "CA-ECBS took a busier route than ECBS -- focal key likely inverted"
    )
