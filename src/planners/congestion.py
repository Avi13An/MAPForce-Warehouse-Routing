"""The congestion map -- the demand-derived signal CA-ECBS steers by.

This is the core of the MAPForce contribution, so it is worth being precise
about what it is and what it is deliberately not.

A cell's congestion score has two parts:

1. **Demand traffic (static).** Given which shelves orders actually hit and how
   often, some corridors are structurally busy: they lie on the shortest routes
   between popular shelves and the packing stations. For a shelf s, a station p
   and a demand weight w(s), every cell c lying on *some* shortest s->p route
   picks up w(s). The membership test is the standard distance-field identity

       c is on a shortest s->p path  <=>  d(s, c) + d(c, p) == d(s, p)

   which needs only one backward BFS per shelf and per station, not an
   enumeration of paths. This term depends on the order distribution, not on
   the robots, so it is computed once per demand setting.

2. **Planned occupancy (dynamic).** Cells that the robots already planned in
   this window will sit on. This changes every replan.

Both are normalised to [0, 1] before being combined, so that a single lambda
means the same thing on a 20x20 toy grid and on a MovingAI warehouse map.
Without that, the lambda sweep in `src/tuning/` would be measuring map size.

**Why congestion is a focal term and never a cost term.** Adding congestion to
g or h would change what "optimal" means and void the ECBS bound -- the planner
would be optimising a different objective and could return a path arbitrarily
worse in true path cost. Instead congestion only orders the FOCAL list, among
candidates already proven to be within w of optimal. That is the whole design,
and it is why Section 4.3 of the proposal can claim the guarantee survives.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .common import Cell, GridView, Path, Solution, cell_at
from .low_level import TrueDistanceHeuristic

try:  # optional: only speeds up CongestionMap._build_static on large maps
    import numpy as _np
except ImportError:  # pragma: no cover - the pure-Python path is used instead
    _np = None


def uniform_demand(shelves: Sequence[Cell]) -> Dict[Cell, float]:
    """Control condition: every shelf equally likely (standard MAPF benchmarks)."""
    if not shelves:
        return {}
    w = 1.0 / len(shelves)
    return {s: w for s in shelves}


def zipf_demand(shelves: Sequence[Cell], exponent: float = 1.0) -> Dict[Cell, float]:
    """Stand-in for the Instacart frequency distribution.

    Real order data is heavy-tailed: a small set of products takes most of the
    volume. A Zipf law over shelf rank reproduces that shape well enough to
    develop and test against while the real pipeline is being built.

    This is scaffolding, not the deliverable. Replace it with the measured
    Instacart frequencies before any result goes in the report, and say in the
    report which one produced each number.
    """
    if not shelves:
        return {}
    raw = {s: 1.0 / ((i + 1) ** exponent) for i, s in enumerate(shelves)}
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def clustered_demand(
    shelves: Sequence[Cell], centre: Cell, share: float = 0.8
) -> Dict[Cell, float]:
    """The 'extreme' skew level: most demand concentrated near one zone."""
    if not shelves:
        return {}
    ranked = sorted(shelves, key=lambda c: abs(c[0] - centre[0]) + abs(c[1] - centre[1]))
    hot = ranked[: max(1, len(ranked) // 5)]
    cold = ranked[len(hot):]
    out: Dict[Cell, float] = {}
    for s in hot:
        out[s] = share / len(hot)
    for s in cold:
        out[s] = (1.0 - share) / len(cold) if cold else 0.0
    return out


class CongestionMap:
    """Per-cell expected traffic, from demand plus the robots' current plans."""

    def __init__(
        self,
        grid: GridView,
        shelves: Sequence[Cell],
        stations: Sequence[Cell],
        demand: Optional[Dict[Cell, float]] = None,
    ) -> None:
        self.grid = grid
        self.shelves = list(shelves)
        self.stations = list(stations)
        self.demand = demand if demand is not None else uniform_demand(self.shelves)
        self.static: Dict[Cell, float] = {}
        self.planned: Dict[Cell, float] = {}
        self._peak: float = 0.0                       # normaliser of `planned`
        self._own: Dict[int, Dict[Cell, float]] = {}  # each agent's own counts
        self._build_static()

    # -- part 1: demand traffic ------------------------------------------------
    def _build_static(self) -> None:
        """Accumulate demand-weighted shortest-route traffic over every cell.

        Uses numpy when it is installed and the pure-Python loop otherwise.
        The two give bit-identical maps (tests/test_congestion.py checks it):
        the numpy path adds each shelf's share to a cell the same number of
        times and in the same order as the loop does, so not even the
        floating-point rounding differs. The loop is fine on the small layouts
        but scales with shelves x stations x cells, about 25 times the work of
        the 65 x 46 map on the real 161 x 63 MovingAI map (2,514 pick spots,
        40 stations), which is why the fast path was added (21 Oct). With
        numpy that map builds in about 20 s.
        """
        if not self.shelves or not self.stations:
            return
        raw = self._raw_traffic_numpy() if _np is not None else None
        if raw is None:
            raw = self._raw_traffic_python()
        peak = max(raw.values()) if raw else 0.0
        if peak > 0:
            self.static = {c: v / peak for c, v in raw.items()}

    def _raw_traffic_numpy(self) -> Optional[Dict[Cell, float]]:
        np = _np
        station_fields = [TrueDistanceHeuristic(self.grid, p) for p in self.stations]
        cells = sorted(set().union(*(f.dist for f in station_fields)))
        if not cells:
            return {}
        index = {c: i for i, c in enumerate(cells)}
        big = 1 << 28                              # "unreachable"; never a real distance
        n, n_st = len(cells), len(self.stations)
        dp = np.full((n_st, n), big, dtype=np.int64)
        for j, f in enumerate(station_fields):
            for c, d in f.dist.items():
                dp[j, index[c]] = d

        raw = np.zeros(n, dtype=np.float64)
        touched = np.zeros(n, dtype=bool)
        for s in self.shelves:
            w = self.demand.get(s, 0.0)
            if w <= 0.0:
                continue
            field = TrueDistanceHeuristic(self.grid, s)   # one at a time: memory
            ds = np.full(n, big, dtype=np.int64)
            for c, d in field.dist.items():
                i = index.get(c)
                if i is not None:
                    ds[i] = d
            span = np.array([field.dist.get(p, big) for p in self.stations],
                            dtype=np.int64)
            ok = span < big                         # stations this shelf can reach
            if not ok.any():
                continue
            on_route = (ds[None, :] + dp[ok]) == span[ok][:, None]
            on_route &= ds[None, :] < big
            count = on_route.sum(axis=0)
            share = w / len(self.stations)
            # The loop adds `share` once per (station, cell) hit; repeat the
            # same additions so the rounding matches it exactly.
            for k in range(int(count.max()) if count.size else 0):
                hit = count > k
                raw[hit] += share
            touched |= count > 0
        return {cells[i]: float(raw[i]) for i in np.flatnonzero(touched)}

    def _raw_traffic_python(self) -> Dict[Cell, float]:
        station_fields = {p: TrueDistanceHeuristic(self.grid, p) for p in self.stations}

        raw: Dict[Cell, float] = {}
        for s in self.shelves:
            w = self.demand.get(s, 0.0)
            if w <= 0.0:
                continue
            ds = TrueDistanceHeuristic(self.grid, s)
            for p in self.stations:
                dp = station_fields[p]
                span = ds(p)
                if span == float("inf"):
                    continue  # station unreachable from this shelf
                # Each shelf splits its demand evenly across the stations it can
                # reach; a real slotting policy would weight this, and that is a
                # drop-in change here.
                share = w / len(self.stations)
                for c, d_from_shelf in ds.dist.items():
                    d_to_station = dp(c)
                    if d_to_station == float("inf"):
                        continue
                    if d_from_shelf + d_to_station == span:
                        raw[c] = raw.get(c, 0.0) + share
        return raw

    # -- part 2: planned occupancy --------------------------------------------
    def update_planned(self, solution: Solution, window: Optional[int] = None) -> None:
        """Recount how many robots are planned through each cell.

        Called once per replan. `window` limits the lookahead to the rolling
        horizon; beyond it the plan is about to be discarded anyway.
        """
        counts: Dict[Cell, float] = {}
        own: Dict[int, Dict[Cell, float]] = {}
        for agent, path in solution.items():
            mine: Dict[Cell, float] = {}
            horizon = len(path) if window is None else min(len(path), window)
            for t in range(horizon):
                c = cell_at(path, t)
                counts[c] = counts.get(c, 0.0) + 1.0
                mine[c] = mine.get(c, 0.0) + 1.0
            own[agent] = mine
        peak = max(counts.values()) if counts else 0.0
        self.planned = {c: v / peak for c, v in counts.items()} if peak > 0 else {}
        self._peak = peak
        self._own = own if peak > 0 else {}

    def clear_planned(self) -> None:
        self.planned = {}
        self._peak = 0.0
        self._own = {}

    # -- combined score --------------------------------------------------------
    def score(self, cell: Cell) -> float:
        """Congestion of one cell in [0, 2]: demand traffic plus planned traffic."""
        return self.static.get(cell, 0.0) + self.planned.get(cell, 0.0)

    def path_score(self, path: Path) -> float:
        """Mean congestion along a path.

        Mean rather than sum so that a longer path is not penalised twice -- its
        length is already paid for in the cost term, and double-counting it
        would make lambda quietly behave like a cost multiplier.
        """
        if not path:
            return 0.0
        return sum(self.score(c) for c in path) / len(path)

    def solution_score(self, solution: Solution) -> float:
        if not solution:
            return 0.0
        return sum(self.path_score(p) for p in solution.values()) / len(solution)

    # -- congestion as one agent sees it (what CA-ECBS plans with) -----------
    #
    # `score` counts *every* planned route, which is right for measuring how
    # busy the floor is (the exposure metric, the heatmap). It is wrong for
    # planning: a robot's own route would count as traffic in its own way, so
    # its shortest path always looks busier than any detour, and CA-ECBS spent
    # its whole w-budget on detours away from itself. Measured on 21 Oct on
    # the custom layout: completed orders roughly halved at 10-20 robots.
    # The `*_for` variants subtract the agent's own contribution, so a robot
    # steers around *other* robots' planned traffic only.

    def score_for(self, agent: int, cell: Cell) -> float:
        """Demand traffic plus planned occupancy by robots other than `agent`."""
        planned = self.planned.get(cell, 0.0)
        if planned and self._peak:
            planned -= self._own.get(agent, {}).get(cell, 0.0) / self._peak
        return self.static.get(cell, 0.0) + planned

    def path_score_for(self, agent: int, path: Path) -> float:
        if not path:
            return 0.0
        return sum(self.score_for(agent, c) for c in path) / len(path)

    def solution_score_for_planning(self, solution: Solution) -> float:
        """Mean over agents of each path's congestion from *other* robots."""
        if not solution:
            return 0.0
        return sum(self.path_score_for(a, p) for a, p in solution.items()) / len(solution)

    def hottest(self, n: int = 10) -> List[tuple]:
        """The n busiest cells -- used by the visualiser's heatmap overlay."""
        cells = {c: self.score(c) for c in set(self.static) | set(self.planned)}
        return sorted(cells.items(), key=lambda kv: -kv[1])[:n]
