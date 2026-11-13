"""The experiment runner: the full factor grid from Section 4.5, plus statistics.

Owner per the proposal: Madduri (experiments & analysis).

Factors swept (Section 4.5's table):
    robots       10, 20, 30, 40  (fixed 10 Nov; see below)
    layouts      custom narrow-aisle, hand-built 65x46 warehouse, and the
                 real MovingAI warehouse-10-20-10-2-1 map (added 10 Nov)
    demand skew  uniform, real, extreme
    delays       none, p = 0.05
    seeds        3 per configuration

Robot counts: 10-40, one list everywhere (the proposal gave 5-20 in one
place and 10-80 in another, which the feedback deducted for). Measured on
10 Nov at 150 timesteps with a 5 s planner limit: CBS already fails 67-80%
of its planning calls at 40 robots on both layouts (106-127 s per run), so
60-80 robots would add only time-outs and hours of runtime. 30 is there to
locate where CBS breaks down. Full grid: about 3.5 h.

Every planner sees *identical* instances -- same layout, same robot start
positions, same order stream, same seed -- so differences are attributable to
the planner and the Wilcoxon signed-rank test below is paired, as Section 4.5
specifies.

The Wilcoxon test is implemented here rather than imported from scipy, which is
not in requirements.txt. It is the standard signed-rank statistic with a normal
approximation and tie correction; for the 3-seed cells it will be
underpowered, which is a real limitation to state in the report rather than
paper over.

    python -m src.experiments.runner --quick
    python -m src.experiments.runner --out results/experiments.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from ..demand.orders import OrderStream, build_demand_model
from ..lifelong.simulator import run_lifelong
from ..planners.ca_ecbs import ca_ecbs
from ..planners.cbs import cbs
from ..planners.common import GridView, PlanResult
from ..planners.congestion import CongestionMap
from ..planners.ecbs import ecbs
from ..planners.low_level import independent_astar, prioritized_planning
from ..tuning.sweep import shelves_and_stations

DEFAULT_W = 1.5
DEFAULT_LAMBDA = 1.0


# ------------------------------------------------------------------ planners
def make_planners(congestion: CongestionMap, w: float = DEFAULT_W,
                  lam: float = DEFAULT_LAMBDA, time_limit: float = 5.0,
                  window: Optional[int] = None,
                  occupancy_window: Optional[int] = None) -> Dict:
    """The five planners of Section 4.3, wrapped to one signature.

    `window` bounds conflict resolution to the replanning horizon, which is
    what makes the CBS family usable in the lifelong setting: over an unbounded
    horizon an idle robot standing in a one-lane aisle has to park there
    forever, and the instance becomes unsolvable. Pass the number of timesteps
    that will actually be executed before the next replan.
    """

    def _independent(grid, starts, goals) -> PlanResult:
        sol = independent_astar(grid, starts, goals)
        return PlanResult(sol, 0, 0, 0, 0.0, True)

    def _prioritized(grid, starts, goals) -> PlanResult:
        sol = prioritized_planning(grid, starts, goals)
        if sol is None:
            return PlanResult(None, 0, 0, 0, 0.0, False, "boxed in")
        return PlanResult(sol, 0, 0, 0, 0.0, True)

    return {
        "independent_astar": _independent,
        "prioritized": _prioritized,
        "CBS": lambda g, s, go: cbs(g, s, go, time_limit=time_limit,
                                    window=window),
        "ECBS": lambda g, s, go: ecbs(g, s, go, w=w, time_limit=time_limit,
                                      window=window),
        "CA-ECBS": lambda g, s, go: ca_ecbs(g, s, go, congestion, w=w, lam=lam,
                                            time_limit=time_limit, window=window,
                                            occupancy_window=occupancy_window),
    }


# ------------------------------------------------------------------- layouts
DEFAULT_CUSTOM_SIZE = (35, 50)     # rows, cols: Avi's default layout

# Layout key -> map file. Until 10 Nov "movingai_warehouse" pointed at a
# 65 x 46 map written by hand in the MovingAI file format; it is now
# "warehouse_65x46", and "movingai_warehouse" is the real benchmark map
# (161 x 63). See data/maps/README.md.
MAP_FILES = {
    "warehouse_65x46": os.path.join("data", "maps", "warehouse-65x46-handbuilt.map"),
    "movingai_warehouse": os.path.join("data", "maps", "warehouse-10-20-10-2-1.map"),
}
ALL_LAYOUTS = ("custom_narrow_aisle", "warehouse_65x46", "movingai_warehouse")
# The layouts the main grid runs when none are named: the two the reported
# results were made on, so the documented commands still reproduce them. The
# real map is run with `--layouts movingai_warehouse` (about 4 h on 2 cores
# for the full grid; CA-ECBS is the slow one there).
DEFAULT_LAYOUTS = ("custom_narrow_aisle", "warehouse_65x46")


def load_layouts(use_movingai: bool = True,
                 custom_size: Tuple[int, int] = DEFAULT_CUSTOM_SIZE,
                 names: Optional[Sequence[str]] = None,
                 ) -> Dict[str, GridView]:
    """The warehouse layouts, all built by Avi's environment module.

    * `custom_narrow_aisle`: `build_custom_narrow_aisle_warehouse`, single-lane
      aisles between 5x2 shelf blocks. `custom_size` = (rows, cols); the
      default 35x50 has 1,000 free cells, 580 pick spots and 14 stations.
    * `warehouse_65x46`: a 65x46 map written by hand in the MovingAI file
      format (1,391 free cells, 1,052 pick spots, 16 stations). Reported as
      "MovingAI warehouse" until 10 Nov, which it is not.
    * `movingai_warehouse`: the real warehouse-10-20-10-2-1 benchmark map,
      161x63, 5,699 free cells, 2,514 pick spots, 40 stations.

    `names` picks layouts (default: the two reported ones, see
    DEFAULT_LAYOUTS). `use_movingai=False` keeps only the custom layout.
    """
    from ..environment.layouts import build_custom_narrow_aisle_warehouse
    from ..environment.map_loader import load_movingai_map
    from ..planners.common import WarehouseGridView

    if names is None:
        names = DEFAULT_LAYOUTS if use_movingai else ("custom_narrow_aisle",)
    unknown = set(names) - set(ALL_LAYOUTS)
    if unknown:
        raise ValueError(f"unknown layout(s): {sorted(unknown)}")
    rows, cols = custom_size
    layouts: Dict[str, GridView] = {}
    for name in names:
        if name == "custom_narrow_aisle":
            layouts[name] = WarehouseGridView(
                build_custom_narrow_aisle_warehouse(rows=rows, cols=cols))
            continue
        path = MAP_FILES[name]
        if os.path.exists(path):
            layouts[name] = WarehouseGridView(load_movingai_map(path))
        else:
            print(f"  WARNING: {path} not found -- layout {name} skipped; "
                  "the report must say so")
    return layouts


# ---------------------------------------------------------------- statistics
def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, two-sided p).

    Normal approximation with tie correction. Zero differences are dropped, as
    in the standard treatment. With very few pairs the approximation is poor --
    report n alongside p and do not lean on a p-value from 3 seeds.
    """
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return w, 1.0
    z = (w - mean) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return w, min(1.0, max(0.0, p))


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and half-width of a 95% CI, as Section 4.5 asks results be reported."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(var / n)


# ----------------------------------------------------------------- the sweep
def run_experiments(
    robot_counts: Sequence[int] = (10, 20, 30, 40),
    skews: Sequence[str] = ("uniform", "real", "extreme"),
    delays: Sequence[float] = (0.0, 0.05),
    seeds: Sequence[int] = (0, 1, 2),
    horizon: int = 150,
    replan_every: int = 5,
    order_rate: float = 0.5,
    planner_time_limit: float = 5.0,
    use_movingai: bool = True,
    verbose: bool = True,
    custom_size: Tuple[int, int] = DEFAULT_CUSTOM_SIZE,
    planner_names: Sequence[str] = ("independent_astar", "prioritized",
                                    "CBS", "ECBS", "CA-ECBS"),
    layout_names: Optional[Sequence[str]] = None,
    w: float = DEFAULT_W,
    lam: float = DEFAULT_LAMBDA,
    occupancy_window: Optional[int] = None,
) -> List[Dict]:
    layouts = load_layouts(use_movingai, custom_size=custom_size,
                           names=layout_names)
    rows: List[Dict] = []

    for layout_name, grid in layouts.items():
        shelves, stations = shelves_and_stations(grid)
        free = grid.free_cells() if hasattr(grid, "free_cells") else []
        # The congestion map's demand term depends only on layout, skew and
        # seed, and takes ~18 s to build on the MovingAI map. Build it once per
        # (skew, seed) and wipe the planned-occupancy term before each run,
        # which leaves it in exactly the state a freshly built map would have.
        cache: Dict[Tuple[str, int], Tuple] = {}
        for n_robots in robot_counts:
            if free and n_robots > len(free) // 3:
                if verbose:
                    print(f"  skipping {n_robots} robots on {layout_name}: too dense")
                continue
            for skew in skews:
                for delay in delays:
                    for seed in seeds:
                        rng = random.Random(seed * 7919 + n_robots)
                        pool = list(free) if free else list(shelves)
                        rng.shuffle(pool)
                        starts = pool[:n_robots]
                        if len(starts) < n_robots:
                            continue

                        if (skew, seed) not in cache:
                            demand = build_demand_model(shelves, stations,
                                                        skew=skew, seed=seed)
                            cache[(skew, seed)] = (demand, CongestionMap(
                                grid, shelves, stations, demand.shelf_weights))
                        demand, cong = cache[(skew, seed)]

                        for planner_name in planner_names:
                            cong.clear_planned()
                            planners = make_planners(
                                cong, w=w, lam=lam,
                                time_limit=planner_time_limit,
                                window=replan_every + 1,
                                occupancy_window=occupancy_window)
                            stream = OrderStream(demand, rate=order_rate, seed=seed)
                            res = run_lifelong(
                                grid, planners[planner_name], stream, starts,
                                horizon=horizon, replan_every=replan_every,
                                delay_prob=delay, seed=seed, congestion=cong,
                            )
                            row = {
                                "layout": layout_name,
                                "planner": planner_name,
                                "robots": n_robots,
                                "skew": skew,
                                "delay": delay,
                                "seed": seed,
                                "demand_source": demand.source,
                                "replan_every": replan_every,
                                # The bound and weight each planner ran with
                                # (blank where they do not apply). Files made
                                # before 10 Nov lack these columns; their
                                # ECBS / CA-ECBS rows used w = 1.5, lambda = 1.
                                "w": w if planner_name in ("ECBS", "CA-ECBS") else "",
                                "lam": lam if planner_name == "CA-ECBS" else "",
                                "occupancy_window": (occupancy_window or "")
                                if planner_name == "CA-ECBS" else "",
                                **res.as_row(),
                            }
                            rows.append(row)
                            if verbose:
                                print(f"  {layout_name:<18}{planner_name:<18}"
                                      f"n={n_robots:<3} {skew:<8} d={delay:<5} "
                                      f"s={seed}  thr={res.throughput:7.1f} "
                                      f"succ={res.success_rate:.2f} "
                                      f"coll={res.collisions} ({res.collision_rate:.1f}/1k)")
    return rows


def paired_comparison(rows: Sequence[Dict], metric: str = "throughput",
                      treatment: str = "CA-ECBS") -> None:
    """CA-ECBS against each baseline on identical instances."""
    key = lambda r: (r["layout"], r["robots"], r["skew"], r["delay"], r["seed"])
    by_planner: Dict[str, Dict] = {}
    for r in rows:
        by_planner.setdefault(r["planner"], {})[key(r)] = r

    if treatment not in by_planner:
        return
    print(f"\npaired comparison on {metric} (treatment = {treatment})")
    print(f"  {'baseline':<20}{'n':>4}{'mean diff':>12}{'95% CI':>12}{'p':>10}")
    for baseline, table in by_planner.items():
        if baseline == treatment:
            continue
        shared = sorted(set(table) & set(by_planner[treatment]))
        if not shared:
            continue
        t_vals = [by_planner[treatment][k][metric] for k in shared]
        b_vals = [table[k][metric] for k in shared]
        diffs = [t - b for t, b in zip(t_vals, b_vals)]
        mean, ci = mean_ci(diffs)
        _, p = wilcoxon_signed_rank(t_vals, b_vals)
        print(f"  {baseline:<20}{len(shared):>4}{mean:>12.2f}{ci:>12.2f}{p:>10.4f}")
    print("  (positive mean diff = CA-ECBS higher; with 3 seeds per cell, p is")
    print("   underpowered -- report n and treat p as indicative only)")


def summarise(rows: Sequence[Dict]) -> None:
    if not rows:
        print("no rows")
        return
    print("\nmean throughput by planner and robot count:")
    counts = sorted({r["robots"] for r in rows})
    header = f"  {'planner':<20}" + "".join(f"{c:>10}" for c in counts)
    print(header)
    for planner in ("independent_astar", "prioritized", "CBS", "ECBS", "CA-ECBS"):
        line = f"  {planner:<20}"
        for c in counts:
            vals = [r["throughput"] for r in rows
                    if r["planner"] == planner and r["robots"] == c]
            line += f"{(sum(vals) / len(vals) if vals else 0):>10.1f}"
        print(line)

    print("\nmean planning success rate by planner and robot count:")
    print(header)
    for planner in ("prioritized", "CBS", "ECBS", "CA-ECBS"):
        line = f"  {planner:<20}"
        for c in counts:
            vals = [r["success_rate"] for r in rows
                    if r["planner"] == planner and r["robots"] == c]
            line += f"{(sum(vals) / len(vals) if vals else 0):>10.2f}"
        print(line)

    paired_comparison(rows, "throughput")
    paired_comparison(rows, "congestion_exposure")


def write_csv(rows: Sequence[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        # Union of columns in first-seen order: shards written by different
        # code versions may differ by a column (e.g. replan_every).
        fields: List[str] = []
        for r in rows:
            fields.extend(k for k in r if k not in fields)
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _number(text: str):
    """CSV cell -> int or float when it is one, else the original string."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text in ("True", "False"):
        return text == "True"
    return text


def read_csv(path: str) -> List[Dict]:
    """Rows written by `write_csv`, with numbers turned back into numbers."""
    with open(path, newline="") as f:
        return [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MAPForce experiment runner")
    p.add_argument("--quick", action="store_true",
                   help="small grid for a smoke test")
    p.add_argument("--robots", type=int, nargs="+", default=None,
                   help="robot counts to run, e.g. --robots 10 20 40")
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--replan-every", type=int, default=5,
                   help="replanning interval h (conflict window is h + 1)")
    p.add_argument("--skews", nargs="+", default=None,
                   choices=("uniform", "real", "extreme"))
    p.add_argument("--delays", type=float, nargs="+", default=None)
    p.add_argument("--planners", nargs="+", default=None,
                   choices=("independent_astar", "prioritized", "CBS", "ECBS", "CA-ECBS"),
                   help="run only these planners (e.g. to rerun one after a fix)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-list", type=int, nargs="+", default=None,
                   help="run only these seeds (one shard of a parallel run), "
                        "e.g. --seed-list 1")
    p.add_argument("--merge", type=str, nargs="+", default=None,
                   help="combine shard CSVs into --out and print the summary; "
                        "runs nothing")
    p.add_argument("--time-limit", type=float, default=5.0,
                   help="CPU seconds per planning call")
    p.add_argument("--no-movingai", action="store_true",
                   help="custom layout only (same as --layouts custom_narrow_aisle)")
    p.add_argument("--layouts", nargs="+", default=None, choices=ALL_LAYOUTS,
                   help="layouts to run (default: custom_narrow_aisle "
                        "warehouse_65x46, the reported ones)")
    p.add_argument("--w", type=float, default=DEFAULT_W,
                   help="suboptimality bound for ECBS and CA-ECBS")
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                   help="congestion weight lambda for CA-ECBS")
    p.add_argument("--occupancy-window", type=int, default=None,
                   help="CA-ECBS: count planned occupancy over the first N "
                        "timesteps only (default: whole path, as reported)")
    p.add_argument("--out", type=str, default="results/experiments.csv")
    args = p.parse_args(argv)

    if args.merge:
        rows = [r for path in args.merge for r in read_csv(path)]
        key = lambda r: (r["layout"], r["planner"], r["robots"], r["skew"],
                         r["delay"], r["seed"])
        if len({key(r) for r in rows}) != len(rows):
            print("ERROR: the shards overlap (same configuration twice)")
            return 1
        write_csv(rows, args.out)
        summarise(rows)
        print()
        print(f"merged {len(args.merge)} files, {len(rows)} rows -> {args.out}")
        return 0

    if args.quick:
        rows = run_experiments(robot_counts=tuple(args.robots or (6, 10)),
                               skews=("uniform", "extreme"),
                               delays=(0.0, 0.05), seeds=(0, 1), horizon=60,
                               planner_time_limit=args.time_limit,
                               use_movingai=not args.no_movingai,
                               custom_size=(17, 23),
                               layout_names=args.layouts)
    else:
        extra = {"robot_counts": tuple(args.robots)} if args.robots else {}
        if args.skews:
            extra["skews"] = tuple(args.skews)
        if args.planners:
            extra["planner_names"] = tuple(args.planners)
        if args.delays is not None:
            extra["delays"] = tuple(args.delays)
        if args.layouts:
            extra["layout_names"] = tuple(args.layouts)
        seeds = tuple(args.seed_list) if args.seed_list else tuple(range(args.seeds))
        rows = run_experiments(seeds=seeds,
                               horizon=args.horizon,
                               planner_time_limit=args.time_limit,
                               use_movingai=not args.no_movingai,
                               replan_every=args.replan_every,
                               w=args.w, lam=args.lam,
                               occupancy_window=args.occupancy_window, **extra)
    write_csv(rows, args.out)
    summarise(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
