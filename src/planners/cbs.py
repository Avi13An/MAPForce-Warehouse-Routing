"""Conflict-Based Search (Sharon et al., 2015) -- the optimal baseline.

Two levels, exactly as in the paper and in Section 3 of the proposal:

* High level: best-first search over a *constraint tree*. Each node holds a set
  of constraints and the best solution satisfying them. Expanding a node means
  finding its first conflict and splitting into two children, each forbidding
  one of the two agents from causing it.
* Low level: `space_time_astar`, re-run for the one agent whose constraints
  changed.

CBS is optimal for sum-of-costs because the high level is best-first on a cost
that never decreases down the tree, and the low level is optimal. That
optimality is what makes CBS useful here even though it is too slow at high
density: it is the yardstick ECBS and CA-ECBS are measured against, and the
reference the w-bound test compares to.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .common import (
    Cell,
    Conflict,
    ConstraintSet,
    GridView,
    PlanResult,
    Solution,
    find_conflicts,
    solution_cost,
)
from .low_level import TrueDistanceHeuristic, space_time_astar

DEFAULT_TIME_LIMIT = 30.0
DEFAULT_EXPANSION_LIMIT = 20000


@dataclass(order=True)
class CTNode:
    """One node of the constraint tree."""

    cost: int
    tiebreak: int
    constraints: Dict[int, ConstraintSet] = field(compare=False, default_factory=dict)
    solution: Solution = field(compare=False, default_factory=dict)


def _branch_constraints(conflict: Conflict) -> List[tuple]:
    """The two constraints a conflict splits into.

    Returns [(agent, kind, payload), ...]. Exactly one agent is constrained per
    child, which is what keeps CBS complete: any solution avoiding this conflict
    must satisfy at least one of the two.
    """
    if conflict.kind == "vertex":
        return [
            (conflict.a, "vertex", (conflict.a_cell, conflict.t)),
            (conflict.b, "vertex", (conflict.a_cell, conflict.t)),
        ]
    return [
        (conflict.a, "edge", (conflict.a_cell, conflict.b_cell, conflict.t)),
        (conflict.b, "edge", (conflict.b_cell, conflict.a_cell, conflict.t)),
    ]


def _apply(constraints: ConstraintSet, kind: str, payload) -> ConstraintSet:
    if kind == "vertex":
        cell, t = payload
        return constraints.extend_vertex(cell, t)
    frm, to, t = payload
    return constraints.extend_edge(frm, to, t)


def cbs(
    grid: GridView,
    starts: Dict[int, Cell],
    goals: Dict[int, Cell],
    time_limit: float = DEFAULT_TIME_LIMIT,
    expansion_limit: int = DEFAULT_EXPANSION_LIMIT,
    window: Optional[int] = None,
) -> PlanResult:
    """Optimal MAPF by CBS. Returns a PlanResult; `solved` is False on timeout.

    `window` bounds conflict checking to the first N timesteps -- see
    `common.find_conflicts`. None means full-horizon, which is what makes CBS
    optimal for one-shot MAPF.
    """
    # The time limit is CPU time used by this process, not wall-clock time.
    # Wall-clock limits let anything outside the planner -- Windows standby,
    # other programs, parallel experiment shards -- turn into recorded
    # 'time limit' failures; on 21 Oct a run lost most of its time to
    # standby that way. CPU time makes failures the planner's own and the
    # runs reproducible. (Resolution on Windows is ~16 ms.)
    t0 = time.process_time()
    agents = sorted(starts)
    heuristics = {a: TrueDistanceHeuristic(grid, goals[a]) for a in agents}

    root_constraints = {a: ConstraintSet() for a in agents}
    root_solution: Solution = {}
    for a in agents:
        p = space_time_astar(grid, starts[a], goals[a], heuristic=heuristics[a])
        if p is None:
            return PlanResult(None, 0, 0, 0, time.process_time() - t0, False,
                              f"agent {a} has no path to its goal")
        root_solution[a] = p

    counter = itertools.count()
    root = CTNode(solution_cost(root_solution), next(counter),
                  root_constraints, root_solution)
    open_list: List[CTNode] = [root]
    expansions = 0

    while open_list:
        if time.process_time() - t0 > time_limit:
            return PlanResult(None, 0, root.cost, expansions,
                              time.process_time() - t0, False, "time limit")
        if expansions >= expansion_limit:
            return PlanResult(None, 0, root.cost, expansions,
                              time.process_time() - t0, False, "expansion limit")

        node = heapq.heappop(open_list)
        expansions += 1

        conflicts = find_conflicts(node.solution, first_only=True, window=window)
        if not conflicts:
            # First conflict-free node popped from a best-first OPEN is optimal.
            return PlanResult(node.solution, node.cost, node.cost, expansions,
                              time.process_time() - t0, True)

        for agent, kind, payload in _branch_constraints(conflicts[0]):
            child_constraints = dict(node.constraints)
            child_constraints[agent] = _apply(node.constraints[agent], kind, payload)
            path = space_time_astar(
                grid, starts[agent], goals[agent],
                constraints=child_constraints[agent], heuristic=heuristics[agent],
            )
            if path is None:
                continue  # this branch is infeasible; the sibling may not be
            child_solution = dict(node.solution)
            child_solution[agent] = path
            heapq.heappush(
                open_list,
                CTNode(solution_cost(child_solution), next(counter),
                       child_constraints, child_solution),
            )

    return PlanResult(None, 0, root.cost, expansions,
                      time.process_time() - t0, False, "constraint tree exhausted")
