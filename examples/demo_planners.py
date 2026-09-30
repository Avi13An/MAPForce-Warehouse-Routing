"""Side-by-side run of all five MAPForce planners on one instance.

    python examples/demo_planners.py

Prints the comparison table that Section 4.5 of the proposal describes: each
planner on the same warehouse, the same robots and the same demand, judged on
cost, suboptimality against the CBS optimum, high-level expansions, runtime and
congestion exposure.

This runs entirely on the synthetic narrow-aisle layout in src/tuning/sweep.py,
so it works before the environment, demand and lifelong modules are wired
together. To run it on the real warehouse maps, replace `narrow_aisle_grid()`
with:

    from src.environment.map_loader import load_movingai_map
    from src.planners.common import WarehouseGridView
    grid = WarehouseGridView(load_movingai_map("data/maps/warehouse-10-20-10-2-1.map"))

`WarehouseGridView` adapts Avi's `WarehouseGrid` to the planner interface; no
planner code changes.
"""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.planners.ca_ecbs import ca_ecbs
from src.planners.cbs import cbs
from src.planners.common import validate_solution
from src.planners.congestion import CongestionMap, zipf_demand
from src.planners.ecbs import ecbs
from src.planners.low_level import independent_astar, prioritized_planning
from src.tuning.sweep import narrow_aisle_grid, sample_instance, shelves_and_stations

W = 1.5
LAMBDA = 1.0
N_AGENTS = 8
SEED = 7


def show_grid(grid, solution=None, congestion=None):
    """ASCII view: '#' wall, digits are robot start cells, ':' a hot cell."""
    out = []
    starts = {}
    if solution:
        for a, p in solution.items():
            starts[p[0]] = str(a % 10)
    for y in range(grid.height):
        row = ""
        for x in range(grid.width):
            cell = (x, y)
            if not grid.passable(cell):
                row += "#"
            elif cell in starts:
                row += starts[cell]
            elif congestion and congestion.score(cell) > 0.75:
                row += ":"
            else:
                row += "."
        out.append(row)
    return "\n".join(out)


def main() -> int:
    grid = narrow_aisle_grid(rows=13, cols=17)
    shelves, stations = shelves_and_stations(grid)
    demand = zipf_demand(shelves, exponent=1.0)
    rng = random.Random(SEED)
    starts, goals = sample_instance(grid, shelves, stations, N_AGENTS, rng)

    cong = CongestionMap(grid, shelves, stations, demand)
    print(f"warehouse {grid.width}x{grid.height}, {N_AGENTS} robots, "
          f"skewed (Zipf) demand, w={W}, lambda={LAMBDA}\n")
    print(show_grid(grid, congestion=cong))
    print("\n('#' shelf block, ':' cell in the top quartile of demand traffic)\n")

    results = {}

    r = cbs(grid, starts, goals, time_limit=30.0)
    results["CBS (optimal)"] = r
    optimal = r.cost if r.solved else 0

    results["ECBS"] = ecbs(grid, starts, goals, w=W, time_limit=30.0)

    fresh = CongestionMap(grid, shelves, stations, demand)
    results["CA-ECBS"] = ca_ecbs(grid, starts, goals, fresh, w=W, lam=LAMBDA,
                                 time_limit=30.0)

    header = (f"{'planner':<16}{'solved':>7}{'cost':>7}{'subopt':>9}"
              f"{'expand':>8}{'time s':>9}{'congest':>9}{'valid':>7}")
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        if not res.solved:
            print(f"{name:<16}{'no':>7}{'-':>7}{'-':>9}{res.expansions:>8}"
                  f"{res.runtime:>9.3f}{'-':>9}{'-':>7}   ({res.failure})")
            continue
        subopt = res.cost / optimal if optimal else float("nan")
        exposure = fresh.solution_score(res.solution)
        problems = validate_solution(res.solution, grid)
        print(f"{name:<16}{'yes':>7}{res.cost:>7}{subopt:>9.3f}"
              f"{res.expansions:>8}{res.runtime:>9.3f}{exposure:>9.3f}"
              f"{('ok' if not problems else str(len(problems))):>7}")

    # The two uncoordinated baselines are reported separately: they do not
    # produce collision-free plans, so a cost comparison would be misleading.
    print()
    indep = independent_astar(grid, starts, goals)
    indep_problems = validate_solution(indep, grid)
    print(f"independent A*   : {len(indep_problems)} collisions "
          f"(lower baseline -- shows what coordination is buying)")

    prio = prioritized_planning(grid, starts, goals)
    if prio is None:
        print("prioritized      : failed (incomplete by design; a later robot was boxed in)")
    else:
        print(f"prioritized      : {len(validate_solution(prio, grid))} collisions, "
              f"cost {sum(len(p) - 1 for p in prio.values())}")

    if optimal:
        bound = W * optimal
        worst = max((r.cost for r in results.values() if r.solved), default=0)
        print(f"\nsuboptimality bound: every plan must cost <= {W} x {optimal} "
              f"= {bound:.1f};  worst observed = {worst}  "
              f"{'OK' if worst <= bound + 1e-9 else '*** VIOLATED ***'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
