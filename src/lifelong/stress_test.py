"""Blocked-aisle stress test: every planner, with and without disruptions.

Owner: Jwalant (lifelong execution).

Runs the lifelong loop on identical instances -- same layout, robot starts,
order stream, delays and disruption schedule for every planner -- once with
random aisle closures and once without, and reports how much each planner
loses. The run without closures replans at exactly the timesteps the disrupted
run did: every closure forces an extra replan, and replanning more often
raises throughput on its own, which would otherwise hide the cost.

    python -m src.lifelong.stress_test
    python -m src.lifelong.stress_test --robots 15 --events 12 --seeds 5 --out results/stress.csv

`--layout` picks Avi's custom narrow-aisle layout (default), the MovingAI
map, or the old synthetic 15x21 stand-in. Demand comes from whatever source
`build_demand_model` finds (Instacart if present under data/instacart/, the
synthetic Zipf stand-in otherwise). The provenance line printed at the top says
which -- quote it with any figure taken from here.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Sequence

from ..demand.orders import OrderStream, build_demand_model
from ..experiments.runner import load_layouts, make_planners
from ..planners.congestion import CongestionMap
from ..tuning.sweep import narrow_aisle_grid, shelves_and_stations
from .disruptions import random_aisle_blockages
from .simulator import dock_cells, run_lifelong

PLANNERS = ("independent_astar", "prioritized", "CBS", "ECBS", "CA-ECBS")


def run_stress(
    robots: int = 10,
    seeds: Sequence[int] = (0, 1, 2),
    horizon: int = 150,
    replan_every: int = 5,
    order_rate: float = 0.5,
    delay: float = 0.0,
    events: int = 8,
    length: int = 3,
    duration: int = 20,
    skew: str = "real",
    time_limit: float = 2.0,
    planners: Sequence[str] = PLANNERS,
    verbose: bool = True,
    layout: str = "custom_narrow_aisle",
) -> List[Dict]:
    """`layout` is one of the runner's layouts ("custom_narrow_aisle",
    "warehouse_65x46", "movingai_warehouse"), or "synthetic" for the old 15x21
    stand-in. The reported stress runs used a 2 s planner limit (the default
    here), not the 5 s of the main grid."""
    if layout == "synthetic":
        grid = narrow_aisle_grid(rows=15, cols=21)
    else:
        grid = load_layouts(names=(layout,))[layout]
    shelves, stations = shelves_and_stations(grid)
    docks = dock_cells(grid, stations)
    rows: List[Dict] = []

    for seed in seeds:
        free = grid.free_cells()
        random.Random(seed * 7919 + robots).shuffle(free)
        starts = free[:robots]
        demand = build_demand_model(shelves, stations, skew=skew, seed=seed)
        if verbose and seed == seeds[0]:
            print(f"inputs: {layout} {grid.width}x{grid.height}, {robots} robots, "
                  f"{demand.provenance()}, delay p={delay}")
        schedule = random_aisle_blockages(
            grid, horizon, events, length=length, duration=duration,
            seed=seed, protect=docks)
        # Built once per seed (slow on big maps); its planned-occupancy term
        # is wiped before every run so each starts from the same state.
        cong = CongestionMap(grid, shelves, stations, demand.shelf_weights)

        for name in planners:
            # Disrupted run first, recording when it replanned. Replanning
            # more often raises throughput by itself, so the control run
            # replans at exactly the same timesteps; the difference between
            # the two is then the cost of the closed cells alone.
            replanned_at: List[int] = []
            for disrupted in (True, False):
                cong.clear_planned()
                planner = make_planners(cong, time_limit=time_limit,
                                        window=replan_every + 1)[name]
                clock = [0]

                def recording(g, s, goals, _p=planner, _rec=disrupted):
                    if _rec:
                        replanned_at.append(clock[0])
                    return _p(g, s, goals)

                def tick(t, _pos, _closed):
                    clock[0] = t + 1

                res = run_lifelong(
                    grid, recording, OrderStream(demand, rate=order_rate, seed=seed),
                    starts, horizon=horizon, replan_every=replan_every,
                    delay_prob=delay, seed=seed, congestion=cong,
                    disruptions=schedule if disrupted else (),
                    replan_at=() if disrupted else replanned_at,
                    on_step=tick,
                )
                rows.append({"planner": name, "seed": seed,
                             "disrupted": disrupted, "layout": layout,
                             "time_limit": time_limit, **res.as_row()})
                if verbose:
                    print(f"  seed={seed} {name:<17} disrupted={disrupted!s:<5} "
                          f"orders={res.completed_orders:<3} "
                          f"succ={res.success_rate:.2f} "
                          f"coll={res.collision_rate:.1f}/1k "
                          f"replans+={res.disruption_replans} "
                          f"deferred={res.tasks_deferred}")
    return rows


def summarise(rows: List[Dict]) -> None:
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    print("\nmean over seeds (normal -> disrupted):")
    print(f"{'planner':<18}{'throughput':>22}{'success rate':>20}"
          f"{'collisions/1k':>18}{'extra replans':>15}{'deferred':>10}")
    for name in dict.fromkeys(r["planner"] for r in rows):
        n = [r for r in rows if r["planner"] == name and not r["disrupted"]]
        d = [r for r in rows if r["planner"] == name and r["disrupted"]]
        thr_n, thr_d = mean([r["throughput"] for r in n]), mean([r["throughput"] for r in d])
        drop = 100 * (thr_n - thr_d) / thr_n if thr_n else 0.0
        print(f"{name:<18}"
              f"{thr_n:>8.1f} -> {thr_d:>6.1f} ({drop:>3.0f}%)"
              f"{mean([r['success_rate'] for r in n]):>10.2f} -> "
              f"{mean([r['success_rate'] for r in d]):.2f}"
              f"{mean([r['collision_rate'] for r in n]):>9.1f} -> "
              f"{mean([r['collision_rate'] for r in d]):<5.1f}"
              f"{mean([r['disruption_replans'] for r in d]):>12.1f}"
              f"{mean([r['tasks_deferred'] for r in d]):>10.1f}")


def write_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MAPForce blocked-aisle stress test")
    p.add_argument("--robots", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--events", type=int, default=8)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--duration", type=int, default=20)
    p.add_argument("--skew", choices=("uniform", "real", "extreme"), default="real")
    p.add_argument("--layout", default="custom_narrow_aisle",
                   choices=("custom_narrow_aisle", "warehouse_65x46",
                            "movingai_warehouse", "synthetic"))
    p.add_argument("--time-limit", type=float, default=2.0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=PLANNERS)
    args = p.parse_args(argv)

    rows = run_stress(robots=args.robots, seeds=tuple(range(args.seeds)),
                      horizon=args.horizon, delay=args.delay, events=args.events,
                      length=args.length, duration=args.duration, skew=args.skew,
                      time_limit=args.time_limit, layout=args.layout,
                      planners=tuple(args.planners))
    summarise(rows)
    if args.out:
        write_csv(rows, args.out)
        print(f"\nwrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
