"""Gates for CBS -- the optimality claim the whole comparison rests on.

CBS is the yardstick: ECBS and CA-ECBS are only "within w of optimal" if we can
say what optimal is. So CBS is checked against an independent brute-force
optimum rather than against itself.
"""

import heapq
import itertools
from typing import Dict, Tuple

from src.planners.cbs import cbs
from src.planners.common import SimpleGrid, solution_cost, validate_solution

OPEN4 = SimpleGrid([
    "....",
    "....",
    "....",
    "....",
])

CORRIDOR = SimpleGrid([
    "#####",
    ".....",
    "#####",
])

POCKET = SimpleGrid([
    "#.###",
    ".....",
    "#####",
])


def brute_force_optimal(grid, starts: Dict[int, Tuple[int, int]],
                        goals: Dict[int, Tuple[int, int]]) -> int:
    """Optimal sum-of-costs by Dijkstra over the joint state space.

    Exponential in the number of agents, so only usable on the tiny instances
    here -- which is exactly the point: it is a reference implementation that
    shares no code with CBS, so agreement between the two is real evidence.

    A step costs 1 per agent that is not resting on its goal, which is precisely
    the sum-of-costs accounting in `common.path_cost`.
    """
    agents = sorted(starts)
    start = tuple(starts[a] for a in agents)
    goal = tuple(goals[a] for a in agents)

    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, state = heapq.heappop(pq)
        if state == goal:
            return d
        if d > dist.get(state, 1 << 30):
            continue
        options = []
        for i, cell in enumerate(state):
            moves = [cell] + grid.neighbours(cell)
            options.append(moves)
        for nxt in itertools.product(*options):
            if len(set(nxt)) != len(nxt):
                continue  # vertex conflict
            swap = False
            for i in range(len(nxt)):
                for j in range(i + 1, len(nxt)):
                    if nxt[i] == state[j] and nxt[j] == state[i]:
                        swap = True
            if swap:
                continue
            step = sum(
                0 if (state[i] == goal[i] and nxt[i] == goal[i]) else 1
                for i in range(len(nxt))
            )
            nd = d + step
            if nd < dist.get(nxt, 1 << 30):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))
    raise AssertionError("brute force found no solution")


def test_cbs_matches_brute_force_on_open_grid():
    starts = {0: (0, 0), 1: (3, 0)}
    goals = {0: (3, 0), 1: (0, 0)}
    result = cbs(OPEN4, starts, goals)
    assert result.solved
    assert validate_solution(result.solution, OPEN4) == []
    assert result.cost == brute_force_optimal(OPEN4, starts, goals)


def test_cbs_matches_brute_force_in_a_pocket():
    """A corridor with one passing place -- the case prioritized planning fails."""
    starts = {0: (0, 1), 1: (4, 1)}
    goals = {0: (4, 1), 1: (0, 1)}
    result = cbs(POCKET, starts, goals)
    assert result.solved
    assert validate_solution(result.solution, POCKET) == []
    assert result.cost == brute_force_optimal(POCKET, starts, goals)


def test_cbs_matches_brute_force_with_three_agents():
    starts = {0: (0, 0), 1: (3, 3), 2: (0, 3)}
    goals = {0: (3, 3), 1: (0, 0), 2: (3, 0)}
    result = cbs(OPEN4, starts, goals)
    assert result.solved
    assert validate_solution(result.solution, OPEN4) == []
    assert result.cost == brute_force_optimal(OPEN4, starts, goals)


def test_cbs_reports_failure_rather_than_hanging():
    """A head-on swap in a one-lane corridor has no solution at all.

    CBS must exhaust its tree and report that, not spin. Section 4.4 requires
    failures to be recorded as results.
    """
    starts = {0: (0, 1), 1: (4, 1)}
    goals = {0: (4, 1), 1: (0, 1)}
    result = cbs(CORRIDOR, starts, goals, time_limit=5.0, expansion_limit=2000)
    assert not result.solved
    assert result.failure


def test_cbs_solution_cost_matches_reported_cost():
    """The reported number and the returned plan must not disagree.

    Technical Depth & Correctness in the rubric is explicitly about reported
    results matching what the system actually does.
    """
    starts = {0: (0, 0), 1: (3, 0)}
    goals = {0: (3, 0), 1: (0, 0)}
    result = cbs(OPEN4, starts, goals)
    assert result.solved
    assert result.cost == solution_cost(result.solution)
