"""CA-ECBS -- Congestion-Aware Enhanced CBS. The proposed method.

CA-ECBS is ECBS with one substitution: the focal ordering at both levels
becomes

    conflicts  +  lambda * congestion

instead of conflicts alone. Nothing else changes -- not the objective, not the
constraint tree, not the bound. This file is deliberately short, and it calls
`ecbs()` rather than reimplementing it, so that "CA-ECBS changes only the focal
ordering" is enforced by the code rather than asserted in the report.

What the two parameters do:

* **w** sets how much path cost you are willing to trade. It defines the FOCAL
  window, and the returned plan is guaranteed within w of the optimal sum of
  costs. w = 1 collapses FOCAL to a single node and CA-ECBS becomes CBS.
* **lambda** sets how strongly congestion reorders candidates *inside* that
  window. It cannot move the plan outside the window, so it cannot break the
  guarantee -- at lambda = 0, CA-ECBS is exactly ECBS, which tests/test_ecbs.py
  (test_lambda_zero_reduces_to_ecbs) checks by running both and comparing.

So w buys the room and lambda decides how to spend it. Those are different
knobs, which is why the sweep (report Section 7.4) varies them on separate axes rather than
together.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .common import Cell, GridView, Path, PlanResult, count_conflicts
from .congestion import CongestionMap
from .ecbs import ECBSNode, ecbs
from .low_level import TrueDistanceHeuristic, make_conflict_counter


def ca_ecbs(
    grid: GridView,
    starts: Dict[int, Cell],
    goals: Dict[int, Cell],
    congestion: CongestionMap,
    w: float = 1.5,
    lam: float = 1.0,
    time_limit: float = 30.0,
    expansion_limit: int = 20000,
    update_planned: bool = True,
    window: Optional[int] = None,
    occupancy_window: Optional[int] = None,
) -> PlanResult:
    """ECBS whose focal lists are ordered by conflicts + lambda * congestion.

    `lam` is the lambda of the proposal (spelled out because `lambda` is a
    Python keyword). With `update_planned`, the congestion map's occupancy term
    is refreshed from the root solution before search, so the planner reacts to
    where the fleet is actually going and not only to the demand distribution.

    What the occupancy term is, exactly: each robot's *unconstrained* shortest
    path, computed once per call (not updated as the constraint tree grows),
    counted per cell with time ignored. `occupancy_window` limits that count
    to the first N timesteps of each path. The proposal describes "robots
    already planned to occupy it in the current time window"; the reported
    runs used the default `None` (whole path). Added 21 Oct as an option, so
    the reported results stay reproducible and the windowed variant can be
    run side by side (`runner --occupancy-window`).
    """
    if lam < 0:
        raise ValueError("lambda must be >= 0")

    # --- the high-level focal key ------------------------------------------
    def high_level_key(node: ECBSNode) -> float:
        base = count_conflicts(node.solution, window=window)
        if lam == 0:
            return base
        # Each path is scored against *other* robots' planned traffic only;
        # see CongestionMap.score_for.
        return base + lam * congestion.solution_score_for_planning(node.solution)

    # --- the low-level focal key -------------------------------------------
    def low_level_key_factory(agent: int, others: Sequence[Path]):
        if lam == 0:
            return None  # fall through to ECBS's own conflict-count heuristic
        # Same value as _conflict_penalty(cell, t, prev, others), by table
        # lookup instead of a loop over every other path (21 Oct speed-up).
        conflicts = make_conflict_counter(others)
        def secondary(cell: Cell, t: int, prev: Cell) -> float:
            return conflicts(cell, t, prev) + lam * congestion.score_for(agent, cell)
        return secondary

    # One true-distance table per agent, shared by the occupancy seeding below
    # and by ecbs(). They used to be built twice per call (21 Oct speed-up).
    #
    # Known asymmetry, disclosed in the report (Section 8.1): `time_limit` is
    # enforced inside ecbs(), so this set-up (the tables and the seeding
    # searches) is not counted against CA-ECBS's limit, while plain ECBS and
    # CBS build their tables inside theirs. Measured at 40 robots: about 0.1 s
    # of CPU per call, 2% of the 5 s limit. Left as it was when the reported
    # runs were made, so the results stay reproducible from this code; the
    # simulator's planning-time column does include it.
    heuristics = {a: TrueDistanceHeuristic(grid, goals[a]) for a in sorted(starts)}

    if update_planned:
        # Seed the occupancy term with each agent's unconstrained shortest path,
        # which is what the root node of the constraint tree will hold.
        from .low_level import space_time_astar
        seed = {}
        for a, s in starts.items():
            p = space_time_astar(grid, s, goals[a], heuristic=heuristics[a])
            if p:
                seed[a] = p
        congestion.update_planned(seed, window=occupancy_window)

    return ecbs(
        grid, starts, goals,
        w=w, time_limit=time_limit, expansion_limit=expansion_limit,
        high_level_key=high_level_key,
        low_level_key_factory=low_level_key_factory,
        window=window,
        heuristics=heuristics,
    )
