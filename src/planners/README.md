# MAPForce planners

The CBS-family planner stack: low-level search, CBS, ECBS, the congestion map
and CA-ECBS, plus the lambda/w tuning sweep.

## Who owns what

| File | Module | Owner per the proposal |
|---|---|---|
| `common.py` | Shared types, grid adapter, conflict detection | interface contract — change by agreement |
| `low_level.py` | Space-time A*, true-distance heuristic, prioritized planning | **Ritvik** (low-level search) |
| `cbs.py` | Conflict detection, constraint tree, optimal baseline | **Rohit** (CBS) |
| `ecbs.py` | Focal search at both levels, w-bound | **Chit** (CA-ECBS) |
| `congestion.py` | Demand-weighted traffic map | **Chit** |
| `ca_ecbs.py` | The proposed method | **Chit** |
| `../tuning/sweep.py` | lambda/w sweep | **Chit** |

`low_level.py` and `cbs.py` sit underneath CA-ECBS on the critical path, so they
were written together with it on 10 Oct (Chit's AI-usage log) — ECBS cannot be
tested without them. The commit history shows who changed each file since.

## Architecture

```
            CA-ECBS  (ca_ecbs.py)        <- focal key = conflicts + lambda * congestion
                |  parameterises
            ECBS     (ecbs.py)           <- dual OPEN/FOCAL, w-bound
                |  calls
         focal_space_time_astar          <- low_level.py
                |  over
            GridView                     <- common.py, adapts WarehouseGrid
```

`ca_ecbs.py` *calls* `ecbs()` rather than copying it. That is deliberate: it
makes "CA-ECBS changes only the focal ordering" true by construction, which is
the claim the whole bound argument rests on.

## The one idea worth being able to explain

Congestion is a **focal** term, never a cost term.

Adding congestion to `g` or `h` would change what "optimal" means and void the
ECBS guarantee — the planner would be optimising a different objective and
could return a path arbitrarily worse in real path cost. Instead congestion only
*orders* the FOCAL list, among candidates already proven to be within `w` of
optimal. FOCAL membership is decided by cost alone; congestion decides only
which admitted candidate goes first.

That is why the bound survives for any lambda, and `tests/test_ecbs.py` pushes
lambda to 10 to prove it.

- **w** buys room (how much path cost you will pay).
- **lambda** decides how to spend it (how hard to avoid traffic).

Different knobs, swept on different axes.

## Correctness gates

`pytest tests/ -q` — 178 tests across the project. The ones that matter here:

- `test_cbs.py` — CBS cost equals an independent brute-force joint-state
  Dijkstra optimum. Shares no code with CBS, so agreement is real evidence.
- `test_ecbs.py::test_ecbs_respects_its_suboptimality_bound` — cost <= w x the
  CBS optimum, across scenarios and w in {1.0, 1.2, 1.5, 2.0}.
- `test_ecbs.py::test_ca_ecbs_respects_the_bound_for_every_lambda` — same, with
  lambda up to 10.
- `test_ecbs.py::test_lambda_zero_reduces_to_ecbs` — identical cost *and*
  expansion count at lambda = 0.
- `test_low_level.py` — heuristic admissible and consistent; goal parking under
  a later constraint (the classic CBS bug).
- `test_ecbs_node_identity.py`, `test_ca_ecbs_self_traffic.py` — regression
  tests for the two defects found on 10 Oct; both fail on the old code.
- `test_ca_ecbs_speed.py` — the faster CA-ECBS returns the same plans as the
  first implementation.

A bound violation is a bug report, not a result. The sweep prints
`bound violations: N` for exactly this reason.

## Running it

```bash
pytest tests/ -q                      # correctness gates
python examples/demo_planners.py      # five planners side by side
python -m src.tuning.sweep --layout custom_narrow_aisle --agents 10 --quiet --out results/scratch/sweep.csv
```

## Preliminary numbers (synthetic narrow-aisle layout, not the real maps)

> **Superseded (10 Oct).** The table below was produced before two fixes in
> this package: ECBS identified search nodes by memory address, so nodes were
> sometimes skipped (`ecbs.py`, now `ECBSNode.uid`), and CA-ECBS counted each
> robot's own route as congestion (`congestion.score_for`). Both changed the
> numbers. Current results come from `python -m src.experiments.runner` and
> `python -m src.tuning.sweep --layout custom_narrow_aisle` on the real
> layouts; see Section 7.4 of the final report and
> `results/figures/tables.md`. The CSVs behind this old table are not kept
> in this repository.

8 robots, Zipf demand, w = 1.5, lambda = 1.0 (old code, kept for the record):

| planner | cost | subopt | expansions | congestion |
|---|---|---|---|---|
| CBS | 86 | 1.000 | 59 | 0.939 |
| ECBS | 86 | 1.000 | 2 | 0.932 |
| CA-ECBS | 89 | 1.035 | 2 | 0.791 |

## Scaffolding status

| Was | Now | Status |
|---|---|---|
| `zipf_demand`, `clustered_demand` in the sweep | `demand.orders.build_demand_model` (Instacart when `data/instacart/` has it, Zipf stand-in otherwise); `--demand-model legacy` keeps the old ones | done 10 Oct. All reported runs use the Instacart frequencies; the Zipf stand-in is the fallback and the robustness check |
| `narrow_aisle_grid` in the sweep and the runner | Avi's `build_custom_narrow_aisle_warehouse` and the hand-built 65x46 map (`--layout`; reported as "MovingAI" until 10 Oct, which it is not) | done 10 Oct. The real MovingAI map is `--layout(s) movingai_warehouse` since 10 Oct |
| `sample_instance` one-shot batches | the rolling-horizon loop in `lifelong/simulator.py` for all lifelong results; the sweep stays one-shot on purpose, because path cost relative to CBS is only defined there | done |

Planner time limits are **CPU seconds** (`time.process_time`) since 10 Oct, so
standby or parallel runs cannot cause recorded time-outs.

## Review fixes, 10 Oct

- **`congestion.py`**: the demand term is built with numpy when available,
  bit-identical to the loop (`test_numpy_and_python_builds_are_bit_identical`),
  so the 161x63 MovingAI map builds in ~20 s. Shelf distance fields are no
  longer all held in memory at once.
- **`ca_ecbs.py`**: new `occupancy_window` option. What the occupancy term is
  in the reported runs: each robot's *unconstrained* shortest path, set once
  per call, counted per cell with time ignored, over the whole path. The
  proposal says "in the current time window"; `--occupancy-window N` does
  that. Default unchanged, so reported results reproduce.
- **What the sweep shows about lambda**: for any lambda from 0.5 to 5 the
  results are almost the same (exposure and cost within a few percent). The
  likely reason: low-level conflict counts nearly always tie, so lambda only
  rescales an ordering that congestion alone already decides. `w` is what sets how much path cost CA-ECBS pays. A
  fairer throughput comparison runs CA-ECBS at w = 1.1-1.2 (`runner --w`),
  where it spends far less on detours; not yet run on the full grid.
- **Lifelong path cost**: the simulator now records `mean_path_cost` and
  `path_cost_ratio` per run (cost over sum of shortest distances, an upper
  bound on cost relative to CBS).
