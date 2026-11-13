"""The lambda / w sweep -- Chit's tuning deliverable (Section 4.5 of the proposal).

Two parameters, two different questions, so they are swept on separate axes
rather than jointly tuned:

* **w** buys room. It widens the FOCAL window and so sets the maximum path cost
  the plan may pay. Its effect is bounded and known in advance.
* **lambda** decides how that room is spent. It reorders candidates inside the
  window toward low-congestion routes. Its effect is empirical and is the thing
  this sweep exists to measure.

For every cell of the grid the sweep records cost, suboptimality against the
CBS optimum (not against the planner's own lower bound -- a planner grading its
own bound proves nothing), high-level expansions, runtime, congestion exposure
and whether the w-bound actually held. The last column should be True
everywhere; a single False is a bug report, not a result.

Run it directly:

    python -m src.tuning.sweep --agents 6 --out results/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..planners.ca_ecbs import ca_ecbs
from ..planners.cbs import cbs
from ..planners.common import Cell, SimpleGrid, validate_solution
from ..planners.congestion import (
    CongestionMap,
    clustered_demand,
    uniform_demand,
    zipf_demand,
)
from ..planners.ecbs import ecbs

DEMANDS = {
    "uniform": lambda shelves: uniform_demand(shelves),
    "real": lambda shelves: zipf_demand(shelves, exponent=1.0),
    "extreme": lambda shelves: clustered_demand(shelves, centre=shelves[0], share=0.9),
}


@dataclass
class SweepRow:
    planner: str
    demand: str
    agents: int
    w: float
    lam: float
    seed: int
    solved: bool
    cost: int
    optimal_cost: int
    suboptimality: float
    bound_held: bool
    expansions: int
    runtime: float
    congestion: float
    collisions: int
    layout: str = ""
    demand_source: str = ""        # "instacart" or "synthetic"


def narrow_aisle_grid(rows: int = 11, cols: int = 13, aisle_every: int = 2) -> SimpleGrid:
    """A synthetic narrow-aisle warehouse: shelf blocks separated by one-lane aisles.

    Stands in for `layouts.build_custom_narrow_aisle_warehouse` so the sweep can
    run before the environment module is wired in. Swap it for the real layout
    before producing report figures, and say which one produced each number.
    """
    grid: List[List[str]] = []
    for y in range(rows):
        if y % aisle_every == 1 and 0 < y < rows - 1:
            row = ["." if (x == 0 or x == cols - 1) else "#" for x in range(cols)]
            # Punch cross-aisles so the map stays connected.
            for x in range(0, cols, 4):
                row[x] = "."
        else:
            row = ["." for _ in range(cols)]
        grid.append(row)
    return SimpleGrid(["".join(r) for r in grid])


def shelves_and_stations(grid: SimpleGrid) -> Tuple[List[Cell], List[Cell]]:
    """Shelf faces are the free cells beside a block; stations sit on the perimeter.

    If the grid wraps Avi's `WarehouseGrid` (a MovingAI map loaded by
    `load_movingai_map`), the pick spots and packing stations it already
    indexed are used instead -- 16 stations along the bottom edge rather than
    the 4 corners the generic rule would pick. A cell indexed as both is kept
    as a station only, so no order is ever picked from a drop-off cell.
    """
    env = getattr(grid, "grid", None)
    if env is not None and getattr(env, "shelves", None) \
            and getattr(env, "packing_stations", None):
        stations = sorted(s for s in env.packing_stations if grid.passable(s))
        station_set = set(stations)
        shelves = sorted(s for s in env.shelves
                         if s not in station_set and grid.passable(s))
        return shelves, stations

    shelves = []
    for y in range(grid.height):
        for x in range(grid.width):
            if not grid.passable((x, y)):
                continue
            if any(not grid.passable((x + dx, y + dy))
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                shelves.append((x, y))
    stations = [(0, 0), (grid.width - 1, 0), (0, grid.height - 1),
                (grid.width - 1, grid.height - 1)]
    stations = [s for s in stations if grid.passable(s)]
    return shelves or grid.free_cells()[:8], stations


def sample_instance(
    grid: SimpleGrid, shelves: Sequence[Cell], stations: Sequence[Cell],
    n_agents: int, rng: random.Random,
    goal_weights: Optional[Dict[Cell, float]] = None,
) -> Tuple[Dict[int, Cell], Dict[int, Cell]]:
    """One batch of trips: start on a random free cell, goal at a shelf.

    By default goals are drawn uniformly from all shelves, which is what every
    reported sweep used: demand skew then changes only the congestion map,
    not where the robots go. Pass `goal_weights` (a demand model's shelf
    weights) to draw goals from the demand instead (`--goal-sampling demand`,
    added 10 Nov), so popular shelves are visited more often, as in the
    lifelong runs. (The docstring used to say robots start near a station;
    they never did.)
    """
    free = [c for c in grid.free_cells()]
    rng.shuffle(free)
    starts = {i: free[i] for i in range(n_agents)}
    goals = {}
    used = set()
    shelf_list = list(shelves)
    weights = None
    if goal_weights is not None:
        weights = [goal_weights.get(s, 0.0) for s in shelf_list]
        if not any(w > 0 for w in weights):
            weights = None
    for i in range(n_agents):
        for _ in range(50):
            g = (rng.choices(shelf_list, weights=weights, k=1)[0] if weights
                 else rng.choice(shelf_list))
            if g not in used and g != starts[i]:
                goals[i] = g
                used.add(g)
                break
        else:
            goals[i] = rng.choice(free)
    return starts, goals


def run_sweep(
    grid: SimpleGrid,
    n_agents: int = 6,
    ws: Sequence[float] = (1.0, 1.2, 1.5, 2.0),
    lams: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 5.0),
    demand_levels: Sequence[str] = ("uniform", "real", "extreme"),
    seeds: Sequence[int] = (0, 1, 2),
    time_limit: float = 20.0,
    include_baselines: bool = True,
    verbose: bool = True,
    layout: str = "",
    demand_model: str = "orders",
    goal_sampling: str = "uniform",
) -> List[SweepRow]:
    """`demand_model="orders"` (default since 10 Nov) takes shelf weights from
    `demand.orders.build_demand_model`, the same model the lifelong
    experiments use -- Instacart when data/instacart/ has it, the synthetic
    Zipf stand-in otherwise. `"legacy"` keeps the sweep's original built-in
    distributions (DEMANDS above), for reproducing the old results/sweep.csv.

    `goal_sampling="uniform"` (the reported runs) draws robot goals uniformly
    from the shelves; `"demand"` draws them from the demand model, so skew
    also changes where robots go. See `sample_instance`.
    """
    if goal_sampling not in ("uniform", "demand"):
        raise ValueError("goal_sampling must be 'uniform' or 'demand'")
    from ..demand.orders import build_demand_model

    shelves, stations = shelves_and_stations(grid)
    rows: List[SweepRow] = []

    for demand_name, seed in itertools.product(demand_levels, seeds):
        rng = random.Random(seed)
        if demand_model == "legacy":
            demand = DEMANDS[demand_name](shelves)
            source = "synthetic"
        else:
            model = build_demand_model(shelves, stations, skew=demand_name, seed=seed)
            demand, source = model.shelf_weights, model.source
        starts, goals = sample_instance(
            grid, shelves, stations, n_agents, rng,
            goal_weights=demand if goal_sampling == "demand" else None)

        # The optimal reference. If CBS times out, suboptimality is unknown for
        # this instance and the bound column is reported as None rather than
        # quietly assumed to hold.
        reference = cbs(grid, starts, goals, time_limit=time_limit)
        opt = reference.cost if reference.solved else 0

        def record(planner: str, result, w: float, lam: float, cong: CongestionMap):
            exposure = cong.solution_score(result.solution) if result.solved else 0.0
            collisions = (
                len(validate_solution(result.solution, grid)) if result.solved else -1
            )
            subopt = (result.cost / opt) if (result.solved and opt) else 0.0
            held = (result.cost <= w * opt + 1e-9) if (result.solved and opt) else True
            rows.append(SweepRow(
                planner=planner, demand=demand_name, agents=n_agents, w=w, lam=lam,
                seed=seed, solved=result.solved, cost=result.cost, optimal_cost=opt,
                suboptimality=round(subopt, 4), bound_held=held,
                expansions=result.expansions, runtime=round(result.runtime, 4),
                congestion=round(exposure, 4), collisions=collisions,
                layout=layout, demand_source=source,
            ))
            if verbose:
                flag = "" if held else "   *** BOUND VIOLATED ***"
                print(f"  {planner:<10} demand={demand_name:<8} w={w:<4} lam={lam:<4} "
                      f"cost={result.cost:<4} subopt={subopt:.3f} "
                      f"cong={exposure:.3f} exp={result.expansions}{flag}")

        # Built once per instance (its demand term takes ~19 s on the MovingAI
        # map) and reset before each CA-ECBS call; clear_planned() leaves it
        # exactly as a fresh map (tests/test_movingai_integration.py).
        cong = CongestionMap(grid, shelves, stations, demand)

        if include_baselines and reference.solved:
            record("CBS", reference, 1.0, 0.0, cong)

        for w in ws:
            plain = ecbs(grid, starts, goals, w=w, time_limit=time_limit)
            cong.clear_planned()
            record("ECBS", plain, w, 0.0, cong)
            for lam in lams:
                cong.clear_planned()
                result = ca_ecbs(grid, starts, goals, cong, w=w, lam=lam,
                                 time_limit=time_limit)
                cong.clear_planned()
                record("CA-ECBS", result, w, lam, cong)

    return rows


def write_csv(rows: Sequence[SweepRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def summarise(rows: Sequence[SweepRow]) -> None:
    """The two numbers that decide whether the method works, printed plainly."""
    violations = [r for r in rows if not r.bound_held]
    print("\n" + "=" * 72)
    print(f"rows: {len(rows)}   bound violations: {len(violations)}")
    if violations:
        print("  *** the suboptimality guarantee failed -- this is a bug, not a result")
        for r in violations[:5]:
            print(f"    {r.planner} w={r.w} lam={r.lam} cost={r.cost} opt={r.optimal_cost}")

    solved = [r for r in rows if r.solved and r.planner == "CA-ECBS"]
    if not solved:
        print("no CA-ECBS runs solved; nothing to summarise")
        return

    print("\nmean congestion exposure and suboptimality by lambda (CA-ECBS):")
    print(f"  {'lambda':>8}  {'congestion':>11}  {'subopt':>8}  {'runtime':>8}  n")
    by_lam: Dict[float, List[SweepRow]] = {}
    for r in solved:
        by_lam.setdefault(r.lam, []).append(r)
    for lam in sorted(by_lam):
        g = by_lam[lam]
        print(f"  {lam:>8}  {sum(x.congestion for x in g) / len(g):>11.4f}  "
              f"{sum(x.suboptimality for x in g) / len(g):>8.4f}  "
              f"{sum(x.runtime for x in g) / len(g):>8.4f}  {len(g)}")
    print("\nlambda = 0 is plain ECBS. A useful lambda lowers congestion without")
    print("raising suboptimality much; if congestion is flat across lambda, the")
    print("congestion term is not reaching the focal ordering.")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="CA-ECBS lambda/w sweep")
    p.add_argument("--agents", type=int, default=6)
    p.add_argument("--rows", type=int, default=11)
    p.add_argument("--cols", type=int, default=13)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--out", type=str, default="results/sweep.csv")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--layout", default="synthetic",
                   choices=("synthetic", "custom_narrow_aisle", "warehouse_65x46",
                            "movingai_warehouse"),
                   help="synthetic = the original --rows x --cols stand-in; "
                        "the others are Avi's layouts, as in the experiments "
                        "(movingai_warehouse = the real 161x63 benchmark map)")
    p.add_argument("--demand-model", default="orders", choices=("orders", "legacy"))
    p.add_argument("--goal-sampling", default="uniform", choices=("uniform", "demand"),
                   help="uniform = goals drawn evenly from all shelves (the reported "
                        "sweeps); demand = drawn from the demand model")
    args = p.parse_args(argv)

    if args.layout == "synthetic":
        grid = narrow_aisle_grid(rows=args.rows, cols=args.cols)
    else:
        from ..experiments.runner import load_layouts   # avoids a circular import
        grid = load_layouts(names=(args.layout,))[args.layout]
    print(f"{args.layout} {grid.width}x{grid.height}, {args.agents} agents, "
          f"{args.seeds} seeds, CPU time limit {args.time_limit} s\n")
    rows = run_sweep(
        grid, n_agents=args.agents, seeds=tuple(range(args.seeds)),
        time_limit=args.time_limit, verbose=not args.quiet,
        layout=args.layout, demand_model=args.demand_model,
        goal_sampling=args.goal_sampling,
    )
    if rows:
        write_csv(rows, args.out)
        summarise(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
