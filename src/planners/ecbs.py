"""Enhanced CBS (Barer et al., 2014) -- bounded-suboptimal MAPF.

ECBS is CBS with focal search at *both* levels:

* High level: OPEN is ordered by a lower bound LB(N) = sum of per-agent lower
  bounds. FOCAL holds the nodes whose actual cost is within a factor w of the
  smallest LB in OPEN, ordered by a secondary heuristic (number of conflicts).
* Low level: `focal_space_time_astar`, which returns a path within w of that
  agent's optimum plus the lower bound proving it.

The guarantee is the point. FOCAL only ever *reorders* candidates that are
already inside the w-window; it never admits one that is outside it. So the
returned solution costs at most w times the optimal sum of costs, whatever the
secondary heuristic does. That is precisely the property CA-ECBS exploits: it
substitutes a different secondary heuristic and inherits the bound unchanged.

`ecbs()` is written with the focal ordering injected rather than hard-coded, so
`ca_ecbs.py` is a parameterisation of this file, not a copy of it. If the two
ever diverge, the claim in the report that "CA-ECBS changes only the focal
ordering" stops being true -- keeping them one function keeps the claim honest.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .common import (
    Cell,
    ConstraintSet,
    GridView,
    Path,
    PlanResult,
    Solution,
    count_conflicts,
    find_conflicts,
    solution_cost,
)
from .cbs import _apply, _branch_constraints
from .low_level import TrueDistanceHeuristic, focal_space_time_astar

DEFAULT_TIME_LIMIT = 30.0
DEFAULT_EXPANSION_LIMIT = 20000


@dataclass
class ECBSNode:
    cost: int
    lb: int
    constraints: Dict[int, ConstraintSet] = field(default_factory=dict)
    solution: Solution = field(default_factory=dict)
    agent_lb: Dict[int, int] = field(default_factory=dict)
    # Creation serial number. Identifies the node in OPEN, FOCAL and the
    # expanded set, and breaks ties deterministically. (Until 21 Oct this
    # was id(node): CPython reuses the address of a freed node, so a new
    # child could share it with an already-expanded node, be taken as
    # expanded and skipped -- spurious "constraint tree exhausted"
    # failures that depended on whatever ran earlier in the process.)
    uid: int = -1


# A high-level focal key: lower is expanded first. The default is ECBS's own
# "fewest conflicts"; CA-ECBS passes a congestion-aware version.
HighLevelKey = Callable[[ECBSNode], float]

# A factory for the low-level focal key, given the agent being replanned and the
# other agents' current paths. Returning None means "use the default conflict
# count", which is what plain ECBS does.
LowLevelKeyFactory = Callable[[int, Sequence[Path]], Optional[Callable]]


def default_high_level_key(node: ECBSNode) -> float:
    return count_conflicts(node.solution)


def ecbs(
    grid: GridView,
    starts: Dict[int, Cell],
    goals: Dict[int, Cell],
    w: float = 1.5,
    time_limit: float = DEFAULT_TIME_LIMIT,
    expansion_limit: int = DEFAULT_EXPANSION_LIMIT,
    high_level_key: Optional[HighLevelKey] = None,
    low_level_key_factory: Optional[LowLevelKeyFactory] = None,
    window: Optional[int] = None,
    heuristics: Optional[Dict[int, TrueDistanceHeuristic]] = None,
) -> PlanResult:
    """Bounded-suboptimal MAPF. Returned cost <= w * optimal sum of costs.

    `high_level_key` and `low_level_key_factory` are the only extension points,
    and neither can affect the bound -- they order FOCAL, they do not define it.
    """
    if w < 1.0:
        raise ValueError("w must be >= 1")
    # The time limit is CPU time used by this process, not wall-clock time.
    # Wall-clock limits let anything outside the planner -- Windows standby,
    # other programs, parallel experiment shards -- turn into recorded
    # 'time limit' failures; on 21 Oct a run lost most of its time to
    # standby that way. CPU time makes failures the planner's own and the
    # runs reproducible. (Resolution on Windows is ~16 ms.)
    t0 = time.process_time()
    agents = sorted(starts)
    # `heuristics` may be passed in by a caller that already built them (CA-ECBS
    # does, for its occupancy seeding); the tables are a pure function of the
    # grid and the goal, so reusing them changes nothing but the time.
    if heuristics is None:
        heuristics = {a: TrueDistanceHeuristic(grid, goals[a]) for a in agents}
    hl_key = high_level_key or (lambda n: count_conflicts(n.solution, window=window))

    def plan_agent(agent: int, constraints: ConstraintSet, solution: Solution):
        """Low-level focal search for one agent against the others' current paths."""
        others = [p for a, p in solution.items() if a != agent]
        secondary = None
        if low_level_key_factory is not None:
            secondary = low_level_key_factory(agent, others)
        return focal_space_time_astar(
            grid, starts[agent], goals[agent],
            constraints=constraints, heuristic=heuristics[agent],
            w=w, others=others, secondary=secondary,
        )

    # --- root ---------------------------------------------------------------
    root_constraints = {a: ConstraintSet() for a in agents}
    root_solution: Solution = {}
    root_lb: Dict[int, int] = {}
    for a in agents:
        path, lb = plan_agent(a, root_constraints[a], root_solution)
        if path is None:
            return PlanResult(None, 0, 0, 0, time.process_time() - t0, False,
                              f"agent {a} has no path to its goal")
        root_solution[a] = path
        root_lb[a] = lb

    counter = itertools.count()
    root = ECBSNode(solution_cost(root_solution), sum(root_lb.values()),
                    root_constraints, root_solution, root_lb,
                    uid=next(counter))

    # OPEN is ordered by LB; FOCAL by the secondary key. Both index the same
    # nodes, so a node removed from one is skipped by uid in the other.
    open_heap: List[tuple] = [(root.lb, root.uid, root.uid, root)]
    focal_heap: List[tuple] = [(hl_key(root), root.cost, root.uid, root)]
    dead: set = set()
    expansions = 0
    best_lb = root.lb

    while open_heap:
        if time.process_time() - t0 > time_limit:
            return PlanResult(None, 0, best_lb, expansions,
                              time.process_time() - t0, False, "time limit")
        if expansions >= expansion_limit:
            return PlanResult(None, 0, best_lb, expansions,
                              time.process_time() - t0, False, "expansion limit")

        while open_heap and open_heap[0][3].uid in dead:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        new_lb = open_heap[0][0]

        # FOCAL is maintained incrementally. Rebuilding it on every expansion
        # is O(|OPEN|) per expansion and so quadratic overall, which is what
        # made the lifelong runs time out rather than solve. The admission
        # threshold only ever rises, so a full rebuild is needed only when
        # best_lb moves; otherwise newly generated children are admitted as
        # they are created (see the branch loop below).
        if new_lb != best_lb or not focal_heap:
            best_lb = new_lb
            focal_heap = [
                (hl_key(n), n.cost, n.uid, n)
                for (_, _, _, n) in open_heap
                if n.uid not in dead and n.cost <= w * best_lb
            ]
            heapq.heapify(focal_heap)

        while focal_heap and focal_heap[0][3].uid in dead:
            heapq.heappop(focal_heap)

        if not focal_heap:
            # The window is empty; fall back to OPEN's head, which is inside
            # the bound by definition of best_lb.
            node = open_heap[0][3]
        else:
            node = heapq.heappop(focal_heap)[3]

        dead.add(node.uid)
        expansions += 1

        conflicts = find_conflicts(node.solution, first_only=True, window=window)
        if not conflicts:
            return PlanResult(node.solution, node.cost, best_lb, expansions,
                              time.process_time() - t0, True)

        for agent, kind, payload in _branch_constraints(conflicts[0]):
            child_constraints = dict(node.constraints)
            child_constraints[agent] = _apply(node.constraints[agent], kind, payload)
            path, lb = plan_agent(agent, child_constraints[agent], node.solution)
            if path is None:
                continue
            child_solution = dict(node.solution)
            child_solution[agent] = path
            child_lb = dict(node.agent_lb)
            child_lb[agent] = lb
            child = ECBSNode(
                solution_cost(child_solution), sum(child_lb.values()),
                child_constraints, child_solution, child_lb,
                uid=next(counter),
            )
            # A child's LB can never fall below its parent's: constraints only
            # ever remove options. Clamping keeps the bound sound against
            # floating-point and tie-breaking noise in the low level.
            child.lb = max(child.lb, node.lb)
            heapq.heappush(open_heap, (child.lb, child.uid, child.uid, child))
            # Admit it to FOCAL now if it already qualifies, so the list stays
            # current without a rebuild.
            if child.cost <= w * best_lb:
                heapq.heappush(focal_heap, (hl_key(child), child.cost,
                                            child.uid, child))

    return PlanResult(None, 0, best_lb, expansions,
                      time.process_time() - t0, False, "constraint tree exhausted")
