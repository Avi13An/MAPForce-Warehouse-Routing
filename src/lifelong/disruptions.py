"""Blocked-aisle disruptions for the lifelong loop.

Owner: Jwalant (lifelong execution).

A disruption is a set of cells that becomes impassable at `trigger_timestep`
and reopens `duration` timesteps later -- a spilled pallet, a person working in
an aisle. The event objects are Avi's `TimedBlockedAisleEvent`
(src/environment/events.py), so the environment module and the simulator
describe disruptions the same way.

Two deliberate differences from `DisruptionManager.update`:

* **Active cells are computed from the schedule, not toggled.** A cell is
  blocked at t iff some event has trigger <= t < trigger + duration. Nothing
  depends on `update` being called on exactly the trigger step, and two
  overlapping events on one cell cannot clear each other early.
* **The planner sees disruptions through a view, not by mutating the grid.**
  `BlockedGridView` wraps any `GridView`, so the synthetic narrow-aisle layout
  and the MovingAI map are handled identically, and nothing has to remember to
  undo a mutation.

Occupied cells are never blocked. A robot already standing in an aisle when
it closes is allowed to drive out; the cell closes behind it. The physical
reading is that the aisle is closed to new entries. The planning reason is
that a robot whose start cell is impassable has no MAPF solution at all, and
neither has anyone planned together with it.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..environment.events import TimedBlockedAisleEvent
from ..planners.common import MOVES, Cell, GridView


def active_blocked_cells(events: Iterable, t: int) -> Set[Cell]:
    """Cells closed by at least one event at timestep `t`.

    Accepts any objects with `trigger_timestep`, `duration` and
    `blocked_cells` -- Avi's `TimedBlockedAisleEvent` in practice. The window
    is half-open: blocked from `trigger_timestep` up to, not including,
    `trigger_timestep + duration`.
    """
    out: Set[Cell] = set()
    for e in events:
        if e.trigger_timestep <= t < e.trigger_timestep + e.duration:
            out.update(tuple(c) for c in e.blocked_cells)
    return out


class BlockedGridView:
    """A `GridView` with some extra cells closed. Used for one planning call."""

    def __init__(self, base: GridView, blocked: Set[Cell]) -> None:
        self.base = base
        self.blocked = frozenset(blocked)

    def passable(self, cell: Cell) -> bool:
        return cell not in self.blocked and self.base.passable(cell)

    def neighbours(self, cell: Cell) -> List[Cell]:
        x, y = cell
        return [n for n in ((x + dx, y + dy) for dx, dy in MOVES)
                if self.passable(n)]

    def __getattr__(self, name):
        # Anything else (free_cells, width, ...) comes from the base grid.
        return getattr(self.base, name)


class Reachability:
    """Which cells can reach which, on a (possibly disrupted) grid.

    A blockage can cut a shelf or a dock off from a robot. Handing that robot
    the unreachable goal would make the whole planning call fail -- every
    other robot would then be frozen too -- so the simulator checks
    reachability first and defers the task instead.

    Components are labelled lazily, by one flood fill per component touched,
    so a query costs nothing once its component is known.
    """

    def __init__(self, grid: GridView) -> None:
        self.grid = grid
        self._label: Dict[Cell, int] = {}
        self._next = 0

    def _label_of(self, cell: Cell) -> Optional[int]:
        if not self.grid.passable(cell):
            return None
        if cell not in self._label:
            lab = self._next
            self._next += 1
            self._label[cell] = lab
            frontier = [cell]
            while frontier:
                nxt = []
                for c in frontier:
                    for n in self.grid.neighbours(c):
                        if n not in self._label:
                            self._label[n] = lab
                            nxt.append(n)
                frontier = nxt
        return self._label[cell]

    def connected(self, a: Cell, b: Cell) -> bool:
        la = self._label_of(a)
        return la is not None and la == self._label_of(b)


def random_aisle_blockages(
    grid: GridView,
    horizon: int,
    n_events: int,
    length: int = 3,
    duration: int = 20,
    seed: int = 0,
    protect: Sequence[Cell] = (),
    cells: Optional[Sequence[Cell]] = None,
) -> List[TimedBlockedAisleEvent]:
    """Random straight aisle segments closed at random times.

    Each event closes `length` consecutive free cells in a row or column for
    `duration` timesteps, starting uniformly in [0, horizon - duration].
    Cells in `protect` (packing stations and their drop-off cells, typically)
    are never closed: closing a station would test the order stream, not the
    planner. Seeded, so every planner sees the same disruptions -- the same
    fairness rule as the order stream.
    """
    rng = random.Random(seed)
    if cells is None:
        cells = grid.free_cells()
    cells = sorted(cells)
    protected = set(protect)
    events: List[TimedBlockedAisleEvent] = []
    latest_start = max(0, horizon - duration)

    for _ in range(n_events):
        for _attempt in range(200):
            x, y = rng.choice(cells)
            dx, dy = rng.choice(((1, 0), (0, 1)))
            seg = [(x + i * dx, y + i * dy) for i in range(length)]
            if all(grid.passable(c) and c not in protected for c in seg):
                start = rng.randint(0, latest_start)
                events.append(TimedBlockedAisleEvent(start, duration, seg))
                break
    return events
