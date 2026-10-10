"""Low-level single-agent search: the engine underneath every MAPForce planner.

Three things live here:

1. `TrueDistanceHeuristic` -- exact cost-to-go from a backward BFS over the
   static grid. Admissible and consistent by construction (Section 3 of the
   proposal), which is what lets CBS claim optimality.
2. `space_time_astar` -- A* over (cell, timestep) respecting CBS constraints.
   Used as the low level of CBS and as the prioritized-planning baseline.
3. `focal_space_time_astar` -- the bounded-suboptimal variant ECBS needs. It
   returns both a path and a lower bound, and the gap between them is exactly
   what the w-guarantee is measured against.

Design note: the wait action is handled explicitly rather than read off
`WarehouseGrid.get_neighbors` (which bundles it in as the (0, 0) move), because
waiting has different constraint semantics from moving -- it can never trigger
an edge conflict, and it must stay legal even when the agent is boxed in.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

from .common import (
    Cell,
    ConstraintSet,
    GridView,
    MOVES,
    Path,
    Solution,
    cell_at,
)

# Hard cap on the time dimension so a malformed instance fails fast instead of
# searching forever. The experiment runner reports a failure rather than hanging,
# which is the behaviour Section 4.4 requires ("if a planner fails, robots wait
# in place and the failure is recorded").
DEFAULT_MAX_T = 512


class TrueDistanceHeuristic:
    """Exact distance to one goal over the static grid, by backward BFS.

    Exact on the obstacle-free grid and a strict underestimate once other agents
    are in the way, so it is admissible; it changes by at most one per move, so
    it is consistent. Both properties are asserted in tests/test_low_level.py.

    Unreachable cells map to infinity, which prunes them from the search rather
    than letting A* expand a dead component.
    """

    __slots__ = ("goal", "dist")

    def __init__(self, grid: GridView, goal: Cell) -> None:
        self.goal = goal
        self.dist: Dict[Cell, int] = {}
        if not grid.passable(goal):
            return
        self.dist[goal] = 0
        frontier = [goal]
        while frontier:
            nxt: List[Cell] = []
            for cell in frontier:
                d = self.dist[cell] + 1
                for n in grid.neighbours(cell):
                    if n not in self.dist:
                        self.dist[n] = d
                        nxt.append(n)
            frontier = nxt

    def __call__(self, cell: Cell) -> float:
        return self.dist.get(cell, float("inf"))

    def reachable(self, cell: Cell) -> bool:
        return cell in self.dist


def _successors(
    grid: GridView, cell: Cell, t: int, constraints: ConstraintSet
) -> List[Cell]:
    """Legal cells at t+1 given the constraints on this agent.

    Waiting is offered first so that, on ties, a plan that holds position is
    preferred to one that paces back and forth -- cosmetic, but it makes the
    visualiser output far easier to read.
    """
    out: List[Cell] = []
    if grid.passable(cell) and not constraints.blocks_vertex(cell, t + 1):
        out.append(cell)
    x, y = cell
    for dx, dy in MOVES:
        n = (x + dx, y + dy)
        if not grid.passable(n):
            continue
        if constraints.blocks_vertex(n, t + 1):
            continue
        if constraints.blocks_edge(cell, n, t + 1):
            continue
        out.append(n)
    return out


def _reconstruct(came: Dict[Tuple[Cell, int], Tuple[Cell, int]], end: Tuple[Cell, int]) -> Path:
    path: Path = []
    node: Optional[Tuple[Cell, int]] = end
    while node is not None:
        path.append(node[0])
        node = came.get(node)
    path.reverse()
    return path


def space_time_astar(
    grid: GridView,
    start: Cell,
    goal: Cell,
    constraints: Optional[ConstraintSet] = None,
    heuristic: Optional[TrueDistanceHeuristic] = None,
    max_t: int = DEFAULT_MAX_T,
) -> Optional[Path]:
    """Optimal single-agent path under `constraints`, or None if there is none.

    The goal test is not simply "standing on the goal": the agent must be able to
    stay there. If a vertex constraint on the goal exists at some later timestep,
    arriving before it would produce a plan CBS cannot actually execute, so the
    search keeps going until it can park safely.
    """
    constraints = constraints or ConstraintSet()
    h = heuristic or TrueDistanceHeuristic(grid, goal)
    if not h.reachable(start):
        return None

    safe_from = constraints.latest_vertex_time(goal)

    start_node = (start, 0)
    open_heap: List[Tuple[float, int, Tuple[Cell, int]]] = [(h(start), 0, start_node)]
    came: Dict[Tuple[Cell, int], Tuple[Cell, int]] = {}
    g: Dict[Tuple[Cell, int], int] = {start_node: 0}
    closed = set()

    while open_heap:
        f, gc, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        cell, t = node

        if cell == goal and t > safe_from:
            return _reconstruct(came, node)
        if t >= max_t:
            continue

        for nxt in _successors(grid, cell, t, constraints):
            if not h.reachable(nxt):
                continue
            child = (nxt, t + 1)
            ng = gc + 1
            if ng < g.get(child, 1 << 30):
                g[child] = ng
                came[child] = node
                heapq.heappush(open_heap, (ng + h(nxt), ng, child))
    return None


def make_conflict_counter(others: Sequence[Path]):
    """A table-driven `_conflict_penalty` for a fixed set of other paths.

    Returns `count(cell, t, prev)` giving exactly the value
    `_conflict_penalty(cell, t, prev, others)` would, but by dictionary lookup
    instead of a loop over every other agent. Built once per low-level search;
    CA-ECBS uses it (10 Oct) because its focal ordering makes the low-level
    search visit many more nodes than ECBS's, and the per-node loop over ~40
    other paths dominated its runtime. Equivalence is tested in
    tests/test_ca_ecbs_speed.py.
    """
    vertex: Dict[Tuple[Cell, int], int] = {}    # (cell, t) for t < len(path)
    parked: Dict[Cell, List[int]] = {}         # last cell -> path lengths
    moves: Dict[Tuple[Cell, Cell, int], int] = {}  # (from, to, t) moved at t-1 -> t
    for p in others:
        if not p:
            continue
        for t, c in enumerate(p):
            vertex[(c, t)] = vertex.get((c, t), 0) + 1
            if t > 0:
                key = (p[t - 1], c, t)
                moves[key] = moves.get(key, 0) + 1
        parked.setdefault(p[-1], []).append(len(p))

    def count(cell: Cell, t: int, prev: Cell) -> int:
        # Vertex: another agent is on `cell` at t -- on its path, or parked at
        # its end (cell_at holds the last cell after the path ends).
        n = vertex.get((cell, t), 0)
        lens = parked.get(cell)
        if lens:
            n += sum(1 for L in lens if L <= t)
        # Swap: another agent moves cell -> prev while we move prev -> cell.
        # Impossible when waiting (cell == prev: that is a vertex hit), and
        # impossible after a path ends (a parked agent does not move).
        if t > 0 and cell != prev:
            n += moves.get((cell, prev, t), 0)
        return n

    return count


def _conflict_penalty(
    cell: Cell, t: int, prev: Cell, others: Sequence[Path]
) -> int:
    """How many other agents this single step collides with.

    This is the low-level focal heuristic from Barer et al.: ECBS prefers, among
    the paths it is allowed to return, the one that walks into the fewest other
    agents. It never affects which paths are *allowed* -- only their order --
    which is why the suboptimality bound survives.
    """
    n = 0
    for p in others:
        if not p:
            continue
        if cell_at(p, t) == cell:
            n += 1
        elif t > 0 and cell_at(p, t) == prev and cell_at(p, t - 1) == cell:
            n += 1  # swap
    return n


def focal_space_time_astar(
    grid: GridView,
    start: Cell,
    goal: Cell,
    constraints: Optional[ConstraintSet] = None,
    heuristic: Optional[TrueDistanceHeuristic] = None,
    w: float = 1.5,
    others: Optional[Sequence[Path]] = None,
    secondary=None,
    max_t: int = DEFAULT_MAX_T,
) -> Tuple[Optional[Path], int]:
    """Bounded-suboptimal single-agent search.

    Returns `(path, lower_bound)` where `lower_bound` is the smallest f-value
    still in OPEN -- a provable lower bound on this agent's optimal cost under
    these constraints. The returned path costs at most `w * lower_bound`.

    OPEN is ordered by f, as in A*. FOCAL holds the subset of OPEN with
    f <= w * f_min and is ordered by `secondary`, which defaults to the conflict
    count above. CA-ECBS overrides `secondary` to fold in congestion -- that
    single substitution is the whole of Chit's method at the low level.

    `secondary(cell, t, prev)` returns a number; lower is expanded first.
    """
    constraints = constraints or ConstraintSet()
    others = others or []
    h = heuristic or TrueDistanceHeuristic(grid, goal)
    if not h.reachable(start):
        return None, 0
    if secondary is None:
        def secondary(cell: Cell, t: int, prev: Cell) -> float:
            return _conflict_penalty(cell, t, prev, others)

    safe_from = constraints.latest_vertex_time(goal)
    start_node = (start, 0)

    # OPEN: (f, g, node). FOCAL: (secondary_total, f, g, node).
    open_heap: List[Tuple[float, int, Tuple[Cell, int]]] = [(h(start), 0, start_node)]
    focal: List[Tuple[float, float, int, Tuple[Cell, int]]] = [
        (0.0, h(start), 0, start_node)
    ]
    came: Dict[Tuple[Cell, int], Tuple[Cell, int]] = {}
    g: Dict[Tuple[Cell, int], int] = {start_node: 0}
    sec_total: Dict[Tuple[Cell, int], float] = {start_node: 0.0}
    closed = set()
    f_min = h(start)

    while focal or open_heap:
        # Keep OPEN's head valid, then refresh FOCAL's admission threshold.
        while open_heap and open_heap[0][2] in closed:
            heapq.heappop(open_heap)
        if not open_heap:
            break
        new_f_min = open_heap[0][0]
        if new_f_min > f_min:
            # The bound has risen: admit everything now within w * f_min.
            f_min = new_f_min
            focal = [
                (sec_total.get(n, 0.0), f, gc, n)
                for (f, gc, n) in open_heap
                if n not in closed and f <= w * f_min
            ]
            heapq.heapify(focal)

        while focal and focal[0][3] in closed:
            heapq.heappop(focal)
        if not focal:
            # Nothing admissible cached; rebuild from OPEN.
            focal = [
                (sec_total.get(n, 0.0), f, gc, n)
                for (f, gc, n) in open_heap
                if n not in closed and f <= w * f_min
            ]
            heapq.heapify(focal)
            if not focal:
                break

        _, f, gc, node = heapq.heappop(focal)
        if node in closed:
            continue
        closed.add(node)
        cell, t = node

        if cell == goal and t > safe_from:
            return _reconstruct(came, node), int(f_min)
        if t >= max_t:
            continue

        for nxt in _successors(grid, cell, t, constraints):
            if not h.reachable(nxt):
                continue
            child = (nxt, t + 1)
            ng = gc + 1
            if ng < g.get(child, 1 << 30):
                g[child] = ng
                came[child] = node
                st = sec_total[node] + secondary(nxt, t + 1, cell)
                sec_total[child] = st
                nf = ng + h(nxt)
                heapq.heappush(open_heap, (nf, ng, child))
                if nf <= w * f_min:
                    heapq.heappush(focal, (st, nf, ng, child))
    return None, int(f_min)


def independent_astar(
    grid: GridView, starts: Dict[int, Cell], goals: Dict[int, Cell]
) -> Solution:
    """Lower baseline: every agent plans alone and ignores the others.

    Produces colliding plans on purpose -- it is the control that shows how much
    coordination is actually buying, and it is where the collision-rate metric in
    Section 4.5 is measured.
    """
    out: Solution = {}
    for a, s in starts.items():
        p = space_time_astar(grid, s, goals[a])
        out[a] = p if p else [s]
    return out


def prioritized_planning(
    grid: GridView,
    starts: Dict[int, Cell],
    goals: Dict[int, Cell],
    order: Optional[Sequence[int]] = None,
    max_t: int = DEFAULT_MAX_T,
) -> Optional[Solution]:
    """Classic cooperative A* (Silver 2005): plan in priority order, reserving cells.

    Incomplete by design -- a later agent can be boxed in by earlier ones, and
    then this returns None. That failure rate is a reportable result, not a bug.
    """
    agents = list(order) if order is not None else sorted(starts)
    solution: Solution = {}
    reserved = ConstraintSet()

    for a in agents:
        path = space_time_astar(
            grid, starts[a], goals[a], constraints=reserved, max_t=max_t
        )
        if path is None:
            return None
        solution[a] = path
        # Reserve this path's cells and edges against everyone planned later.
        horizon = max_t
        for t, cell in enumerate(path):
            reserved = reserved.extend_vertex(cell, t)
        # Hold the goal for the rest of the horizon so later agents route around it.
        goal_cell = path[-1]
        for t in range(len(path), min(len(path) + 32, horizon)):
            reserved = reserved.extend_vertex(goal_cell, t)
        for t in range(len(path) - 1):
            reserved = reserved.extend_edge(path[t + 1], path[t], t + 1)
    return solution
