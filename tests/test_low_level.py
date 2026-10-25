"""Gates for the low-level search.

Everything above this layer -- CBS, ECBS, CA-ECBS -- inherits its correctness
from these properties, so they are checked directly rather than assumed.
"""

from src.planners.common import ConstraintSet, SimpleGrid, validate_solution
from src.planners.low_level import (
    TrueDistanceHeuristic,
    prioritized_planning,
    space_time_astar,
)

OPEN = SimpleGrid([
    ".....",
    ".....",
    ".....",
    ".....",
    ".....",
])

MAZE = SimpleGrid([
    ".....",
    ".###.",
    ".#...",
    ".#.#.",
    "...#.",
])


def test_heuristic_is_exact_on_open_grid():
    """On an empty grid the true-distance heuristic is Manhattan distance."""
    h = TrueDistanceHeuristic(OPEN, (0, 0))
    for x in range(5):
        for y in range(5):
            assert h((x, y)) == x + y


def test_heuristic_is_admissible_and_consistent():
    """h never overestimates, and never changes by more than one per move.

    Admissibility is what makes CBS optimal; consistency is what lets A* close
    a node permanently. Both are checked on the maze, where the heuristic is no
    longer trivially Manhattan.
    """
    goal = (4, 4)
    h = TrueDistanceHeuristic(MAZE, goal)
    for cell in MAZE.free_cells():
        if not h.reachable(cell):
            continue
        path = space_time_astar(MAZE, cell, goal)
        assert path is not None
        true_cost = len(path) - 1
        assert h(cell) <= true_cost, f"inadmissible at {cell}"
        for n in MAZE.neighbours(cell):
            if h.reachable(n):
                assert abs(h(cell) - h(n)) <= 1, f"inconsistent between {cell},{n}"


def test_heuristic_marks_unreachable_cells_infinite():
    walled = SimpleGrid([
        "..#..",
        "..#..",
        "..#..",
    ])
    h = TrueDistanceHeuristic(walled, (0, 0))
    assert h((4, 0)) == float("inf")
    assert not h.reachable((4, 0))


def test_respects_vertex_constraints():
    c = ConstraintSet()
    c = c.extend_vertex((1, 0), 1)
    path = space_time_astar(OPEN, (0, 0), (2, 0), constraints=c)
    assert path is not None
    assert path[1] != (1, 0)


def test_respects_edge_constraints():
    """The agent must not traverse the forbidden edge on the forbidden step."""
    c = ConstraintSet().extend_edge((0, 0), (1, 0), 1)
    path = space_time_astar(OPEN, (0, 0), (2, 0), constraints=c)
    assert path is not None
    assert not (path[0] == (0, 0) and path[1] == (1, 0))


def test_waits_before_parking_on_a_constrained_goal():
    """A later constraint on the goal must delay arrival, not be ignored.

    This is the classic CBS bug: an agent 'finishes' at t=2 while a constraint
    forbids its goal at t=5, producing a plan that cannot actually be executed.
    """
    c = ConstraintSet().extend_vertex((2, 0), 5)
    path = space_time_astar(OPEN, (0, 0), (2, 0), constraints=c)
    assert path is not None
    assert len(path) - 1 > 5
    assert path[5] != (2, 0)


def test_returns_none_when_no_path_exists():
    walled = SimpleGrid(["..#..", "..#..", "..#.."])
    assert space_time_astar(walled, (0, 0), (4, 0)) is None


def test_prioritized_planning_is_collision_free_when_it_succeeds():
    starts = {0: (0, 0), 1: (4, 4), 2: (0, 4)}
    goals = {0: (4, 4), 1: (0, 0), 2: (4, 0)}
    sol = prioritized_planning(OPEN, starts, goals)
    assert sol is not None
    assert validate_solution(sol, OPEN) == []


def test_prioritized_planning_can_fail_by_design():
    """A single-lane corridor with a head-on swap defeats prioritized planning.

    Reported as a failure rather than a crash -- Section 4.5 counts this as the
    success-rate metric, so the incompleteness is a measurement, not a defect.
    """
    corridor = SimpleGrid(["###", ".#.", "..."])
    starts = {0: (0, 1), 1: (2, 1)}
    goals = {0: (2, 1), 1: (0, 1)}
    result = prioritized_planning(corridor, starts, goals)
    assert result is None or validate_solution(result, corridor) == []
