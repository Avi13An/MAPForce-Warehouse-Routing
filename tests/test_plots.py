"""Gates for the figure/table generator.

The report's figures and every number in its results section come from
`python -m src.experiments.plots`. These tests pin the two properties that
matter for that: each input CSV gets its own, named output (so a figure cannot
silently be redrawn from another layout's file), and the tables are computed
correctly from the rows.
"""

import os

import pytest

pytest.importorskip("matplotlib")

from src.experiments import plots  # noqa: E402
from src.experiments.runner import write_csv  # noqa: E402


def _experiment_rows():
    rows = []
    for layout in ("custom_narrow_aisle", "warehouse_65x46"):
        for robots in (10, 20):
            for skew in ("uniform", "real"):
                for delay in (0.0, 0.05):
                    for seed in (0, 1):
                        for planner, thr, exp in (("ECBS", 200.0, 0.8),
                                                  ("CA-ECBS", 150.0, 0.6),
                                                  ("CBS", 100.0, 0.8)):
                            rows.append({
                                "layout": layout, "planner": planner, "robots": robots,
                                "skew": skew, "delay": delay, "seed": seed,
                                "throughput": thr + seed, "success_rate": 1.0,
                                "congestion_exposure": exp, "collision_rate": 2.0 * delay,
                            })
    return rows


def _stress_rows():
    rows = []
    for seed in (0, 1):
        for planner in ("ECBS", "CA-ECBS"):
            rows.append({"planner": planner, "seed": seed, "disrupted": True,
                         "throughput": 90.0, "success_rate": 1.0})
            rows.append({"planner": planner, "seed": seed, "disrupted": False,
                         "throughput": 100.0, "success_rate": 1.0})
    return rows


def _sweep_rows(layout):
    rows = []
    for w in (1.0, 1.5):
        rows.append({"planner": "ECBS", "w": w, "lam": 0.0, "solved": True,
                     "optimal_cost": 100, "suboptimality": 1.0, "congestion": 0.4,
                     "bound_held": True, "layout": layout, "agents": 10})
        for lam in (0.0, 1.0):
            rows.append({"planner": "CA-ECBS", "w": w, "lam": lam, "solved": True,
                         "optimal_cost": 100, "suboptimality": 1.0 + 0.2 * lam * (w - 1),
                         "congestion": 0.4 - 0.2 * lam * (w - 1), "bound_held": True,
                         "layout": layout, "agents": 10})
    return rows


@pytest.fixture()
def outputs(tmp_path):
    exp = tmp_path / "experiments.csv"
    write_csv(_experiment_rows(), str(exp))
    stress = {tag: tmp_path / f"stress_{tag}.csv" for tag in ("custom", "warehouse_65x46")}
    for path in stress.values():
        write_csv(_stress_rows(), str(path))
    sweep = {tag: tmp_path / f"sweep_{tag}.csv" for tag in ("custom", "warehouse_65x46")}
    write_csv(_sweep_rows("custom_narrow_aisle"), str(sweep["custom"]))
    write_csv(_sweep_rows("warehouse_65x46"), str(sweep["warehouse_65x46"]))
    out_dir = tmp_path / "figures"
    code = plots.main(["--experiments", str(exp),
                       "--stress", *map(str, stress.values()),
                       "--sweep", *map(str, sweep.values()),
                       "--replan", "--out-dir", str(out_dir)])
    assert code == 0
    return out_dir


def test_every_input_gets_its_own_named_figure(outputs):
    made = set(os.listdir(outputs))
    assert {"throughput.png", "success_rate.png", "congestion.png", "collisions.png",
            "throughput_by_skew.png", "tables.md",
            "stress_custom.png", "stress_warehouse_65x46.png",
            "sweep_custom.png", "sweep_warehouse_65x46.png"} <= made
    # The old generic names could be overwritten by whichever CSV was passed.
    assert "stress.png" not in made and "sweep.png" not in made


def test_tables_report_the_paired_difference_and_stress_change(outputs):
    text = (outputs / "tables.md").read_text(encoding="utf-8")
    # CA-ECBS is 50 orders below ECBS on every instance, exposure 0.2 lower.
    ecbs_line = next(l for l in text.splitlines() if l.startswith("| ECBS | 32 |"))
    assert "-50.0 (±0.0)" in ecbs_line and "-0.20 (±0.00)" in ecbs_line
    # 100 -> 90 orders under disruption is a 10% loss.
    assert "100.0 → 90.0 | -10.0%" in text
    assert "## λ / w sweep — Hand-built warehouse, 10 agents" in text
    assert "MovingAI warehouse (65×46)" not in text     # the old, wrong label
    assert "bound violations (all planners): 0" in text


def test_missing_inputs_are_skipped_not_fatal(tmp_path):
    out_dir = tmp_path / "figures"
    code = plots.main(["--experiments", str(tmp_path / "none.csv"), "--sweep",
                       "--stress", "--replan", "--out-dir", str(out_dir)])
    assert code == 0
    assert os.listdir(out_dir) == []


def test_demand_model_comparison_table(tmp_path):
    """Two main-grid files made with different demand models are set side by side."""
    def rows(source, ca_throughput):
        out = []
        for r in _experiment_rows():
            r = dict(r, demand_source="uniform" if r["skew"] == "uniform" else source,
                     collisions=0)
            if r["planner"] == "CA-ECBS" and r["skew"] != "uniform":
                r["throughput"] = ca_throughput + r["seed"]
            out.append(r)
        return out

    write_csv(rows("instacart", 180.0), str(tmp_path / "main.csv"))
    write_csv(rows("synthetic", 150.0), str(tmp_path / "other.csv"))
    out_dir = tmp_path / "figures"
    assert plots.main(["--experiments", str(tmp_path / "main.csv"), "--sweep", "--stress",
                       "--replan", "--compare", str(tmp_path / "other.csv"),
                       "--out-dir", str(out_dir)]) == 0
    text = (out_dir / "tables.md").read_text(encoding="utf-8")
    assert "## Demand model comparison" in text
    real = {l.split("|")[1].strip(): l for l in text.splitlines() if "| real |" in l}
    assert "-20.0 (±0.0)" in real["Instacart"]          # 180 vs 200
    assert "-50.0 (±0.0)" in real["Synthetic Zipf"]     # 150 vs 200
    # Uniform rows do not depend on the demand model: all identical here.
    assert "ECBS 16 of 16" in text and "CA-ECBS (proposed) 16 of 16" in text


def test_runtime_table_is_per_planning_call(tmp_path):
    """CPU time and high-level nodes are averaged over calls, not over runs."""
    rows = []
    for r in _experiment_rows():
        calls = 10 if r["seed"] == 0 else 30
        rows.append(dict(r, planning_calls=calls, total_expansions=0,
                         mean_plan_time=1.0 if r["seed"] == 0 else 0.2))
        if r["planner"] in ("ECBS", "CA-ECBS", "CBS"):
            rows[-1]["total_expansions"] = 2 * calls
        if r["planner"] == "CBS":       # a planner with no high-level search
            rows.append(dict(rows[-1], planner="prioritized", total_expansions=0))
    write_csv(rows, str(tmp_path / "main.csv"))
    out_dir = tmp_path / "figures"
    assert plots.main(["--experiments", str(tmp_path / "main.csv"), "--sweep", "--stress",
                       "--replan", "--compare", "", "--out-dir", str(out_dir)]) == 0
    text = (out_dir / "tables.md").read_text(encoding="utf-8")
    block = text[text.index("## Runtime"):].split("\n## ")[0]
    seeds = sorted({r["seed"] for r in rows})
    w = {s: (10 if s == 0 else 30) for s in seeds}
    t = {s: (1.0 if s == 0 else 0.2) for s in seeds}
    expected = sum(w[s] * t[s] for s in seeds) / sum(w.values())
    ecbs = next(l for l in block.splitlines() if l.startswith("| ECBS |"))
    assert f"{expected:.3f} s / 2.0" in ecbs
    prio = next(l for l in block.splitlines() if l.startswith("| Prioritized |"))
    assert f"{expected:.3f} s |" in prio and "/" not in prio


def test_dense_runs_extend_the_main_grid(tmp_path):
    """60 / 80-robot rows get their own figures and a table next to 40 robots."""
    main = [dict(r, robots=40 if r["robots"] == 20 else r["robots"]) for r in _experiment_rows()]
    dense = [dict(r, robots=80, throughput=r["throughput"] - 10)
             for r in main if r["robots"] == 40 and r["planner"] != "CBS"]
    write_csv(main, str(tmp_path / "main.csv"))
    write_csv(dense, str(tmp_path / "dense.csv"))
    out_dir = tmp_path / "figures"
    assert plots.main(["--experiments", str(tmp_path / "main.csv"), "--sweep", "--stress",
                       "--replan", "--compare", "", "--dense", str(tmp_path / "dense.csv"),
                       "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "throughput_dense.png").exists()
    assert (out_dir / "success_dense.png").exists()
    text = (out_dir / "tables.md").read_text(encoding="utf-8")
    block = text[text.index("## Higher density"):]
    assert "40, 80 robots" in block.splitlines()[0]
    # CA-ECBS is 50 behind ECBS in the fixture, at 40 and at 80 robots alike.
    rows80 = [l for l in block.splitlines() if l.startswith("| 80 | CA-ECBS")]
    assert rows80 and "-50.0 (±0.0)" in rows80[0]
    assert "| CBS (optimal) |" not in block      # CBS has no dense runs

