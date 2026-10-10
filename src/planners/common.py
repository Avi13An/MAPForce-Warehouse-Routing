"""Shared types for every MAPForce planner.

Everything below is planner-side only: it depends on the environment through the
narrow `GridView` protocol, never on a concrete `WarehouseGrid`. That keeps the
planners testable on hand-built toy grids (which is how the correctness gates in
tests/ work) and means a change to the environment module cannot silently break
a planner.

Coordinate convention: a cell is an (x, y) integer tuple. Time is an integer
timestep starting at 0. An agent occupies exactly one cell per timestep and may
wait in place at unit cost, exactly as Section 4.1 of the proposal specifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

Cell = Tuple[int, int]
Path = List[Cell]          # path[t] is the cell occupied at timestep t
Solution = Dict[int, Path]  # agent id -> path

# The four cardinal moves. Waiting is handled separately so that the wait action
# is always legal even when every neighbour is blocked.
MOVES: Tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GridView(Protocol):
    """The only thing a planner needs to know about the warehouse floor."""

    def passable(self, cell: Cell) -> bool:
        """True if `cell` is inside the map and not a static or dynamic obstacle."""
        ...

    def neighbours(self, cell: Cell) -> List[Cell]:
        """Passable cells reachable in one move, excluding `cell` itself."""
        ...


class SimpleGrid:
    """A GridView backed by a list of strings. Used by the tests.

    '.' is free, '#' is blocked. Row-major, so grid[y][x]; `passable` takes (x, y).
    """

    def __init__(self, rows: Sequence[str]) -> None:
        self.rows = [r for r in rows]
        self.height = len(self.rows)
        self.width = len(self.rows[0]) if self.rows else 0
        if any(len(r) != self.width for r in self.rows):
            raise ValueError("all rows must be the same width")

    def passable(self, cell: Cell) -> bool:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        return self.rows[y][x] != "#"

    def neighbours(self, cell: Cell) -> List[Cell]:
        x, y = cell
        out = []
        for dx, dy in MOVES:
            n = (x + dx, y + dy)
            if self.passable(n):
                out.append(n)
        return out

    def free_cells(self) -> List[Cell]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.rows[y][x] != "#"
        ]


class WarehouseGridView:
    """Adapter binding Avi's `WarehouseGrid` to `GridView`.

    `WarehouseGrid.is_traversable` is called as `is_traversable(x, y)` in
    examples/demo_environment.py, but we accept the tuple form too so that a
    later signature change in the environment module does not break the
    planners. Neighbours are derived here from `passable` rather than taken from
    `WarehouseGrid.get_neighbors`, because that method includes the cell itself
    (the (0, 0) wait action) and the planners need the wait action handled
    separately -- see `low_level.py`.
    """

    def __init__(self, grid) -> None:
        self.grid = grid

    # `width`, `height` and `free_cells` mirror `SimpleGrid`, so code written
    # against the synthetic layout (the experiment runner, the sweep, the
    # stress test) also runs on the MovingAI maps. Without them the runner
    # crashed with AttributeError the moment it reached the MovingAI layout.
    @property
    def width(self) -> int:
        return self.grid.width

    @property
    def height(self) -> int:
        return self.grid.height

    def free_cells(self) -> List[Cell]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.passable((x, y))
        ]

    def passable(self, cell: Cell) -> bool:
        x, y = cell
        try:
            return bool(self.grid.is_traversable(x, y))
        except TypeError:
            return bool(self.grid.is_traversable((x, y)))

    def neighbours(self, cell: Cell) -> List[Cell]:
        x, y = cell
        return [
            n for n in ((x + dx, y + dy) for dx, dy in MOVES) if self.passable(n)
        ]


@dataclass(frozen=True)
class VertexConstraint:
    """Agent `agent` may not occupy `cell` at timestep `t`."""

    agent: int
    cell: Cell
    t: int


@dataclass(frozen=True)
class EdgeConstraint:
    """Agent `agent` may not traverse `frm` -> `to` arriving at timestep `t`."""

    agent: int
    frm: Cell
    to: Cell
    t: int


Constraint = object  # VertexConstraint | EdgeConstraint


@dataclass
class ConstraintSet:
    """The constraints applying to one agent, indexed for O(1) lookup.

    CBS adds one constraint per child node, so this is copied often; `extend`
    returns a new set rather than mutating, which keeps constraint-tree nodes
    independent.
    """

    vertex: Set[Tuple[Cell, int]] = field(default_factory=set)
    edge: Set[Tuple[Cell, Cell, int]] = field(default_factory=set)

    def blocks_vertex(self, cell: Cell, t: int) -> bool:
        return (cell, t) in self.vertex

    def blocks_edge(self, frm: Cell, to: Cell, t: int) -> bool:
        return (frm, to, t) in self.edge

    def latest_vertex_time(self, cell: Cell) -> int:
        """Last timestep at which `cell` is constrained, or -1 if never.

        The low-level search uses this to decide when an agent may stop at its
        goal: parking on the goal is only safe once no constraint can still
        evict it, otherwise CBS can return a plan that is infeasible later.
        """
        times = [t for (c, t) in self.vertex if c == cell]
        return max(times) if times else -1

    def extend_vertex(self, cell: Cell, t: int) -> "ConstraintSet":
        return ConstraintSet(self.vertex | {(cell, t)}, set(self.edge))

    def extend_edge(self, frm: Cell, to: Cell, t: int) -> "ConstraintSet":
        return ConstraintSet(set(self.vertex), self.edge | {(frm, to, t)})


@dataclass(frozen=True)
class Conflict:
    """A collision between two agents in a candidate solution.

    kind is "vertex" (both in `a_cell` at time `t`) or "edge" (they swap:
    agent `a` moves a_cell -> b_cell while `b` moves b_cell -> a_cell).
    """

    kind: str
    a: int
    b: int
    a_cell: Cell
    b_cell: Cell
    t: int


@dataclass
class PlanResult:
    """What every planner returns.

    cost      sum of individual path costs (sum-of-costs objective)
    lower_bound  a provable lower bound on the optimal cost for this instance.
                 CBS sets it equal to cost (it is optimal); ECBS and CA-ECBS set
                 it to the focal-search lower bound, which is what makes the
                 w-suboptimality claim checkable rather than asserted.
    expansions   high-level nodes expanded, the runtime metric in Section 4.5
    """

    solution: Optional[Solution]
    cost: int
    lower_bound: int
    expansions: int
    runtime: float
    solved: bool
    failure: str = ""

    @property
    def suboptimality(self) -> float:
        if not self.solved or self.lower_bound <= 0:
            return 1.0
        return self.cost / self.lower_bound


def path_cost(path: Path) -> int:
    """Cost of one path under sum-of-costs.

    A path is stored padded to the makespan, so trailing waits at the goal are
    not charged: cost is the number of timesteps until the agent last moves.
    Without this, an agent that arrives early is penalised for the time another
    agent spends finishing, which is not the sum-of-costs objective.
    """
    if not path:
        return 0
    last = len(path) - 1
    while last > 0 and path[last] == path[last - 1]:
        last -= 1
    return last


def solution_cost(solution: Solution) -> int:
    return sum(path_cost(p) for p in solution.values())


def cell_at(path: Path, t: int) -> Cell:
    """Where an agent is at time `t`, holding its goal after the path ends."""
    if t < len(path):
        return path[t]
    return path[-1]


def find_conflicts(solution: Solution, first_only: bool = True,
                   window: Optional[int] = None) -> List[Conflict]:
    """Every vertex and edge conflict in `solution`.

    CBS only needs the first conflict (it branches on one at a time), but the
    focal heuristic in ECBS needs the total count, so both modes share this code
    to guarantee the two never disagree about what a conflict is.

    `window` restricts the check to the first `window` timesteps. That is what
    makes lifelong planning tractable (Li et al., 2021): over an unbounded
    horizon every agent must park on its goal forever, so a single idle robot
    standing in a one-lane aisle makes the whole instance unsolvable. Bounding
    the horizon to the replanning window asks only that the plan be safe for as
    long as it will actually be executed. Leave it None for one-shot MAPF, where
    the stronger guarantee is the right one.
    """
    conflicts: List[Conflict] = []
    agents = sorted(solution)
    if not agents:
        return conflicts
    horizon = max(len(p) for p in solution.values())
    if window is not None:
        horizon = min(horizon, max(1, window))

    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            pa, pb = solution[a], solution[b]
            for t in range(horizon):
                ca, cb = cell_at(pa, t), cell_at(pb, t)
                if ca == cb:
                    conflicts.append(Conflict("vertex", a, b, ca, cb, t))
                    if first_only:
                        return conflicts
                    continue
                if t + 1 < horizon:
                    na, nb = cell_at(pa, t + 1), cell_at(pb, t + 1)
                    if na == cb and nb == ca:
                        conflicts.append(Conflict("edge", a, b, ca, cb, t + 1))
                        if first_only:
                            return conflicts
    return conflicts


def count_conflicts(solution: Solution, window: Optional[int] = None) -> int:
    """The focal heuristic used by ECBS: how many collisions this plan has."""
    return len(find_conflicts(solution, first_only=False, window=window))


def validate_solution(solution: Solution, grid: GridView) -> List[str]:
    """Structural check used by the tests and by the experiment runner.

    Returns a list of human-readable violations; empty means the plan is
    executable. This is what lets the report claim "collision-free by
    construction" as a measured fact rather than a design intention.
    """
    problems: List[str] = []
    for agent, path in solution.items():
        if not path:
            problems.append(f"agent {agent}: empty path")
            continue
        for t, cell in enumerate(path):
            if not grid.passable(cell):
                problems.append(f"agent {agent}: cell {cell} at t={t} not passable")
        for t in range(len(path) - 1):
            x0, y0 = path[t]
            x1, y1 = path[t + 1]
            if abs(x1 - x0) + abs(y1 - y0) > 1:
                problems.append(
                    f"agent {agent}: teleport {path[t]} -> {path[t + 1]} at t={t}"
                )
    for c in find_conflicts(solution, first_only=False):
        problems.append(
            f"{c.kind} conflict: agents {c.a},{c.b} at {c.a_cell} t={c.t}"
        )
    return problems
