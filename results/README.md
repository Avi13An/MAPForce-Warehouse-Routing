# Results: which file is which

Everything the report quotes is rebuilt from the CSV files here by one command:

```bash
python -m src.experiments.plots        # writes results/figures/*.png and results/figures/tables.md
```

`results/figures/tables.md` holds every number in Section 7 of the report.
The command also draws four extra charts the report does not use
(`success_dense`, `sweep_warehouse_65x46`, `sweep_custom_20`, `stress_custom`).

## Layout names: corrected 15 Nov 2026

The map that every result here called **"MovingAI warehouse"**
(`movingai_warehouse`, 65 x 46) is not a MovingAI benchmark map; it is a map
written by hand in the MovingAI file format. On 15 Nov its rows were
relabelled `warehouse_65x46` in every CSV here (only the `layout` value
changed; no number changed), `sweep_movingai.csv` and `stress_movingai.csv`
were renamed to `*_warehouse_65x46.csv`, and the figures and tables were
regenerated with the new names. `movingai_warehouse` now means the real
161 x 63 `warehouse-10-20-10-2-1` benchmark map, which none of the files here
has been run on yet. See `data/maps/README.md`.

Path-cost columns (`mean_path_cost`, `path_cost_ratio`) were added on 15 Nov.
They were filled into `experiments_full.csv` by rerunning every non-CBS run
with the same code and seeds: 574 of 576 reruns reproduced the original row
exactly (orders, success, collisions, exposure, expansions) and got the new
columns. Two CA-ECBS rows at 40 robots matched on every outcome but not on
high-level node count (a CPU-time-limit effect) and are left blank. CBS rows
were not rerun and are blank. `experiments_dense.csv` and the side files
have no path-cost columns.

## Files

All were made on 15 Nov with **Instacart demand**
(`data/instacart/product_frequencies.csv`), except the Zipf comparison file.
The `demand_source` column says `instacart`, `synthetic`, or `uniform` for the
uniform control.

| File | Rows | Feeds |
|---|---|---|
| `experiments_full.csv` | 720 | Main grid: 2 layouts (custom, hand-built 65x46) x robots 10/20/30/40 x 3 skews x 2 delay levels x 3 seeds x 5 planners. Sections 7.1, 7.2, 7.3, 7.5 (delays), 7.6. Figures `throughput`, `success_rate`, `congestion`, `collisions`, `throughput_by_skew` |
| `experiments_dense.csv` | 288 | The same design at 60 and 80 robots, every planner except CBS. Section 7.8. Figure `throughput_dense` |
| `sweep_custom.csv` | 225 | lambda / w sweep, custom layout, 10 agents. Section 7.4, figure `sweep_custom` |
| `sweep_warehouse_65x46.csv`, `sweep_custom_20.csv` | 225 each | The same sweep on the hand-built 65x46 map (was `sweep_movingai.csv`), and on the custom layout with 20 agents. Section 7.4 (tables) |
| `stress_warehouse_65x46.csv`, `stress_custom.csv` | 50 each | Blocked-aisle stress test, Section 7.5 (was `stress_movingai.csv`). Figure `stress_warehouse_65x46`. Planner limit 2 s, not the main grid's 5 s |
| `replan_h2.csv`, `replan_h10.csv` | 60 each | Replanning interval, Section 7.5 (the h = 5 column comes from the main grid) |
| `zipf_baseline/experiments_full.csv` | 720 | The main grid run earlier with synthetic Zipf demand. Section 7.7 only |
| `demo.html` | | The demo: prioritized, CBS, ECBS and CA-ECBS side by side, 30 robots, extreme skew, 5 blocked aisles. Made with `python -m src.viz.render --planners prioritized CBS ECBS CA-ECBS --robots 30 --skew extreme --disruptions 5` |

## Provenance

`experiments_full.provenance.txt` and `experiments_dense.provenance.txt`
record the start and finish time, commit, Python version, machine, SHA-256 of
the demand file, and the exact command of every job. Each grid was run in
pieces (one per seed and robot count) and merged with `runner --merge`; the
pieces and the run logs are not kept in this repository. All runs were made
with the same code on one machine (Linux, Python 3.14, 2 cores). Planner
limits are CPU seconds, so running jobs side by side does not change any
result.

## Cross-machine checks

Runs are deterministic unless a planning call reaches its CPU-time limit.

- 15 Nov: 96 configurations of the Zipf set (made on Windows) were rerun
  under Linux. All 96 rows were identical.
- 15 Nov: the 240 uniform-demand runs do not depend on the demand data. They
  were made on a Windows laptop in the Zipf set and on a Linux machine in the
  final set. Identical runs: independent A* 48 of 48, prioritized 48 of 48,
  ECBS 48 of 48, CA-ECBS 47 of 48, CBS 25 of 48. CBS often reaches its limit,
  and the Linux machine is faster, so CBS solves more calls there. CBS numbers
  from the two sets are therefore not compared with each other.

`scratch/` is git-ignored: put throw-away runs there.
