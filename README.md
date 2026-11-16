# MAPForce — Congestion-Aware Multi-Robot Order Picking under Realistic Demand

**CS F407/U407 Artificial Intelligence — Project (Team 16)**  
**Track:** Application | **Topic:** Search & Problem Solving  
**Institution:** Birla Institute of Technology & Science (BITS) Pilani, Hyderabad Campus

---

## Start here

| You want | Open |
|---|---|
| The demo | `results/demo.html` in any browser (no server, no internet) |
| Every number in the report | `results/figures/tables.md` |
| Which result file is which | `results/README.md` |
| Which map is which | `data/maps/README.md` |

The final report and the members' AI-usage logs are submitted separately and are not kept in this repository.

**Result in one paragraph.** CBS collapses as robots are added (planning success 0.99 at 10 robots, 0.35 at 40). CA-ECBS keeps the ECBS cost guarantee and routes through cells with a 25% lower congestion score than ECBS, but **ECBS delivers the most orders at every density from 10 to 80 robots** (about 49 more per 1,000 timesteps than CA-ECBS). **Our throughput hypothesis is not supported.** Demand comes from the Instacart order data; a second full set of runs on synthetic Zipf demand gives the same conclusions.

---

## Overview

**MAPForce** is a lifelong Multi-Agent Path Finding (MAPF) simulation and benchmarking framework for automated warehouse order picking under realistic, skewed product demand. In real warehouses, customer orders follow heavy-tailed popularity distributions where a small subset of items drives the majority of traffic—creating aisle congestion that standard uniform-random MAPF benchmarks do not capture.

This repository implements a rolling-horizon warehouse simulator alongside **five path planners**:
1. **Independent $A^*$** — Uncoordinated single-agent shortest-path baseline.
2. **Prioritized Planning** — Cooperative Space-Time $A^*$ with a vertex/edge reservation table.
3. **Conflict-Based Search (CBS)** — Two-level optimal MAPF solver minimizing sum of path costs.
4. **Enhanced CBS (ECBS)** — Bounded-suboptimal focal search returning plans within factor $w$ of optimal.
5. **Congestion-Aware ECBS (CA-ECBS)** — Proposed ECBS variant that orders both high-level and low-level `FOCAL` lists by $\text{conflicts} + \lambda \times \text{congestion}$, combining demand-weighted route traffic and fleet occupancy, while preserving the $w$-suboptimality guarantee.

---

## Key Features & System Architecture

* **4-Connected Grid Warehouse Environments** (details and provenance in `data/maps/README.md`):
  * **Custom Narrow-Aisle Layout** (`custom_narrow_aisle`): $50 \times 35$ grid with single-lane aisles between $5 \times 2$ shelf blocks; $1{,}000$ free cells, $580$ pick spots, $14$ packing stations.
  * **Hand-built warehouse** (`warehouse_65x46`): a $65 \times 46$ map written by hand in the MovingAI file format; $1{,}391$ free cells, $1{,}052$ pick spots, $16$ packing stations. **It is not a MovingAI benchmark map.** Until 16 Nov 2026 it was mislabelled as `warehouse-10-20-10-2-1`.
  * **MovingAI benchmark** (`movingai_warehouse`): the real `warehouse-10-20-10-2-1` map, $161 \times 63$, $5{,}699$ free cells, $2{,}514$ pick spots, $40$ packing stations. Added 16 Nov 2026; **all reported results so far use the two layouts above**, not this one (see *Known limitations*).
* **Realistic Demand & Slotting Models:**
  * Product frequencies from the **Instacart Online Grocery Shopping Dataset** (`order_products__prior.csv`, $32.4\text{M}$ order lines across $49{,}677$ products), plus a synthetic **Zipf** ($1/\text{rank}$) distribution for a robustness check. Orders are *sampled* from these product frequencies at a fixed rate; individual Instacart baskets are not replayed.
  * Three demand-skew regimes: **Uniform**, **Real** (Instacart frequencies with random shelf slotting), and **Extreme** (Instacart frequencies with popularity-zoned hot-spot slotting).
* **Rolling-Horizon Lifelong Simulator (RHCR):**
  * Continuous order arrival ($0.5$ orders/timestep), greedy nearest-idle-robot task assignment, windowed conflict resolution ($h = 5$, conflict window $h + 1 = 6$, horizon $T = 150$), stochastic execution delays ($p \in \{0, 0.05\}$), and timed blocked-aisle disruptions.
* **Interactive Side-by-Side Visualizer** (`src/viz/render.py`):
  * Generates one self-contained HTML page that replays several planners side by side on the same layout and order stream, with a congestion heatmap (static demand layer, plus a live layer showing where each fleet is planned to be over the next window, refreshed at every replan), blocked-aisle overlays, and running throughput/collision counters. Planner, layout, robot count and skew are chosen on the command line when the page is generated.

---

## Repository Structure

    MAPForce-Warehouse-Routing/
    ├── data/
    │   ├── instacart/              # Processed Instacart product frequencies (no download needed)
    │   └── maps/                   # Real MovingAI warehouse-10-20-10-2-1 map + hand-built 65x46 map (see README)
    ├── examples/                   # Runnable environment and planner demonstration scripts
    ├── results/
    │   ├── figures/                # Generated plots and summary tables (tables.md)
    │   ├── zipf_baseline/             # Robustness runs under synthetic Zipf demand
    │   ├── experiments_full.csv    # Main 720-run grid (10–40 robots)
    │   ├── experiments_dense.csv   # High-density 288-run grid (60–80 robots)
    │   └── README.md               # Which CSV feeds which figure/table, provenance
    ├── src/
    │   ├── demand/                 # Instacart/Zipf demand loaders, slotting & order stream (orders.py)
    │   ├── environment/            # Grid representation, map loaders, layouts & disruption events
    │   ├── experiments/            # Shard/merge runner, statistical tests & figure generator (plots.py)
    │   ├── lifelong/               # Rolling-horizon execution loop, collision checker, stress test
    │   ├── planners/               # Independent A*, Prioritized Space-Time A*, CBS, ECBS, congestion map, CA-ECBS
    │   ├── tuning/                 # λ / w sweep (one-shot, path cost vs CBS)
    │   └── viz/                    # Side-by-side planner replay visualizer & heatmaps
    ├── tests/                      # 178 pytest unit, invariant and regression tests
    ├── conftest.py                 # Pytest path configuration
    ├── requirements.txt            # Python dependencies
    └── README.md

---

## Installation & Quickstart

### Prerequisites
* **Python 3.9+** (tested on 3.9, 3.10, 3.12, 3.13 and 3.14). The planners and simulator use only the standard library; `matplotlib` is needed for figures, `pytest` for tests, and `numpy` (optional) speeds up the congestion map on the large MovingAI map.

### Setup

    git clone https://github.com/Avi13An/MAPForce-Warehouse-Routing.git
    cd MAPForce-Warehouse-Routing
    pip install -r requirements.txt

### Tests, demos and figures

    python -m pytest tests/ -q                                       # 178 tests, about 20 s
    python -m examples.demo_environment                              # environment module
    python examples/demo_planners.py                                 # five planners on one small instance
    python -m src.viz.render --planners ECBS CA-ECBS --robots 20     # side-by-side demo -> results/demo.html
    python -m src.experiments.plots                                  # every figure and table -> results/figures/

### Reproducing the reported results (about 2 hours on 2 cores)

    python -m src.experiments.runner --quick --out results/scratch/quick.csv     # smoke test, 1-2 minutes
    python -m src.experiments.runner --seed-list 0 --out results/full_seed0.csv  # one shard; repeat for seeds 1 and 2
    python -m src.experiments.runner --merge results/full_seed0.csv results/full_seed1.csv results/full_seed2.csv --out results/experiments_full.csv
    python -m src.tuning.sweep --layout custom_narrow_aisle --agents 10 --quiet --out results/sweep_custom.csv
    python -m src.lifelong.stress_test --layout warehouse_65x46 --events 20 --length 4 --duration 30 --delay 0.05 --seeds 5 --out results/stress_warehouse_65x46.csv
    python -m src.experiments.runner --replan-every 2 --robots 20 40 --skews real --delays 0 --out results/replan_h2.csv

The runner's default layouts are the two the results were made on. To run on the real MovingAI map add `--layouts movingai_warehouse` (the full grid there takes about 4 hours on 2 cores). Other options added on 16 Nov: `--w`, `--lam`, `--occupancy-window` (runner) and `--goal-sampling demand` (sweep). `results/experiments_full.provenance.txt` lists the exact command of every job behind the reported results. Planner time limits are CPU seconds, so jobs can run in parallel, one per core. Use `results/scratch/` for throw-away runs; it is git-ignored.

---

## Summary of Experimental Findings

All planners were evaluated across **720 main runs** ($10\text{–}40$ robots, the custom and hand-built $65 \times 46$ layouts, 3 demand skews, with/without $p=0.05$ delays, 3 seeds), **288 high-density runs** ($60\text{–}80$ robots), **675 one-shot $\lambda/w$ sweep runs** (540 of them CA-ECBS), blocked-aisle stress tests, replanning-interval runs ($h \in \{2, 5, 10\}$), and a full synthetic Zipf replication:

| Metric / Finding | Result |
| :--- | :--- |
| **Throughput hypothesis** | **Not supported.** ECBS delivers the most orders at every density from 10 to 80 robots, about 49 more per 1,000 steps than CA-ECBS (paired over 144 instances, Wilcoxon $p < 10^{-4}$). |
| **Bounded Suboptimality** | **0 bound violations** across all 540 CA-ECBS sweep runs ($w \in \{1.0, 1.2, 1.5, 2.0\}$, $\lambda$ up to 5). |
| **Congestion Exposure** | CA-ECBS reduces mean path congestion exposure by **~25%** vs. ECBS ($0.57$ vs. $0.77$, paired across 144 instances, Wilcoxon $p < 10^{-4}$). |
| **Why throughput drops** | At $w = 1.5$ CA-ECBS spends most of its allowance on congestion-avoiding detours: its one-shot path cost is $18\text{–}32\%$ above the CBS optimum, against $1\text{–}3\%$ for ECBS. The lifelong runs agree: CA-ECBS paths are $16\text{–}23\%$ above the shortest-path lower bound, ECBS $1\text{–}4\%$ (`path_cost_ratio`, `tables.md`). Raising $\lambda$ from 0.5 to 5 barely changes this; $w$ is what sets it. |
| **High density** | CBS planning success drops from $0.99$ (10 robots) to $0.35$ (40 robots). At 80 robots CA-ECBS still solves $99\text{–}100\%$ of calls and beats prioritized planning by $+80.6$ orders / 1,000 steps ($p = 0.0002$), but ECBS still delivers more than CA-ECBS ($251.5$ vs $207.8$ on the custom layout, $205.6$ vs $151.1$ on the $65 \times 46$ map). |
| **Shelf Slotting Impact** | Real product skew with random slotting matches uniform-demand throughput, whereas clustering popular items into a hot zone cuts ECBS throughput by $16\%$ while CA-ECBS keeps exposure flat ($0.55\text{–}0.59$). |

---

## Known limitations and differences from the proposal

* **Real MovingAI map not yet in the results.** The reported grids ran on the custom layout and the hand-built $65 \times 46$ map. The real `warehouse-10-20-10-2-1` map is now in `data/maps/` and loads correctly, but the grid has not been run on it yet. In first timing runs there, CA-ECBS took about 2 s (10 robots) to 10 s (40 robots) of CPU per planning call, against 0.1–0.5 s for ECBS, so expect many CA-ECBS time-outs at the 5 s limit.
* **Robot counts.** The proposal's evaluation table lists 5/10/15/20 robots; the runs use 10/20/30/40, plus 60/80 without CBS, to reach the densities where planners separate.
* **Orders** are drawn from Instacart *product frequencies*, not replayed basket by basket.
* **Path cost relative to CBS.** The one-shot $\lambda/w$ sweep reports cost relative to the CBS optimum. In the lifelong runs, columns `mean_path_cost` and `path_cost_ratio` were added on 16 Nov (`path_cost_ratio` is cost over the sum of shortest distances, an upper bound on cost relative to CBS). They were filled into `experiments_full.csv` by rerunning every non-CBS run: 574 of 576 reruns reproduced the original row exactly and got the new columns; the other two (CA-ECBS, 40 robots) matched on every outcome but not on node count, a CPU-limit effect, and are left blank, as are all CBS rows (CBS was not rerun).
* **CA-ECBS occupancy term.** The planned-occupancy part of the congestion map is each robot's unconstrained shortest path, set once per planning call and counted per cell with time ignored, over the whole path. The proposal describes occupancy "in the current time window"; that variant exists as `--occupancy-window N` but has not been run for the reported results.
* **Sweep goals.** In the one-shot sweep, robot goals are drawn uniformly from all shelves; demand skew changes only the congestion map there. `--goal-sampling demand` draws goals from demand instead (not used for the reported sweep files).
* **Stress test** runs used a 2 s planner limit; the main grid used 5 s.
* **Confidence intervals** in `tables.md` pool skews, delays and seeds within each cell unless a table says otherwise.

---

## Team 16 (MAPForce)

| Member | Student ID | Primary Module & Contributions |
| :--- | :--- | :--- |
| **Avi Tandon** | `2024AAPS0315H` | Environment module, MovingAI & custom layouts, blocked-aisle disruptions, repository integration |
| **Chit Nirav Bhatt** | `2024A3PS0434H` | CA-ECBS congestion map & focal ordering, $\lambda/w$ sweeps, table-driven speed-up, system audit |
| **Ritvik Umesh Modak** | `2024A3PS0404H` | Low-level $A^*$, backward BFS admissible heuristics, Independent $A^*$, Space-Time $A^*$ Prioritized Planning |
| **Rohit Rahul Deshmukh** | `2024A3PS0446H` | High-level CBS constraint tree, ECBS focal search, node-identity fix, Dijkstra & $w$-bound verification |
| **Eeshan Bhandarkar** | `2024A4PS1183H` | Instacart frequency ingestion, random & popularity-zoned slotting, Zipf generator, order stream |
| **Jwalant H. Andharia** | `2024AAPS0325H` | Rolling-horizon simulator (RHCR), collision checker/resolver, delay & stress-test mechanics |
| **Madduri Vishnu Siddhartha Reddy** | `2024AAPS0051H` | Shard/merge experiment runner, 720- & 288-run grids, Wilcoxon tests, automated figure/table scripts |
| **Bysani Chetan Sai Sathvik** | `2024A8PS0506H` | Interactive side-by-side planner replay visualizer, congestion heatmap overlay, report deliverables |
