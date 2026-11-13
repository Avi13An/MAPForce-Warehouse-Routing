"""Figures and tables for the report, from the experiment / sweep / stress CSVs.

    python -m src.experiments.plots            # everything the report uses

One command regenerates every figure and every number in Section 7 of the
report from the CSVs in results/:

    results/figures/throughput.png, success_rate.png, congestion.png,
        collisions.png, throughput_by_skew.png      <- experiments_full.csv
    results/figures/sweep_<name>.png                 <- sweep_<name>.csv
    results/figures/stress_<name>.png                <- stress_<name>.csv
    results/figures/throughput_dense.png,
        success_dense.png                            <- experiments_full.csv
                                                        + experiments_dense.csv
    results/figures/tables.md                        <- all of the above, as text

Each sweep and stress CSV gets its own, named figure (sweep_custom.png,
stress_warehouse_65x46.png, ...), so a figure can never silently be redrawn from a
different layout's file than the one its caption names.

Every point is a mean over the runs in that cell; error bars are the 95%
confidence interval of that mean (normal approximation, as in runner.mean_ci).
Colours follow the planner, never its rank, and every planner also has its own
marker and line style for greyscale printing and colour-blind readers.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .runner import mean_ci, read_csv, wilcoxon_signed_rank  # noqa: E402

# Fixed planner -> style. Categorical slots 1-5 of the reference palette, in
# its documented order; the proposed method takes slot 1.
PLANNERS = ("CA-ECBS", "ECBS", "CBS", "prioritized", "independent_astar")
LABEL = {"CA-ECBS": "CA-ECBS (proposed)", "ECBS": "ECBS", "CBS": "CBS (optimal)",
         "prioritized": "Prioritized", "independent_astar": "Independent A*"}
COLOR = {"CA-ECBS": "#2a78d6", "ECBS": "#eb6834", "CBS": "#1baf7a",
         "prioritized": "#eda100", "independent_astar": "#e87ba4"}
MARKER = {"CA-ECBS": "o", "ECBS": "s", "CBS": "^", "prioritized": "D",
          "independent_astar": "v"}
DASH = {"CA-ECBS": "-", "ECBS": "--", "CBS": "-.", "prioritized": ":",
        "independent_astar": (0, (5, 1, 1, 1))}
# "warehouse_65x46" is a hand-built map in the MovingAI file format, not a
# MovingAI benchmark map; until 10 Nov it was mislabelled "MovingAI warehouse".
# "movingai_warehouse" is the real warehouse-10-20-10-2-1 map (161×63).
LAYOUT_TITLE = {"custom_narrow_aisle": "Custom narrow-aisle (50×35)",
                "warehouse_65x46": "Hand-built warehouse (65×46)",
                "movingai_warehouse": "MovingAI warehouse-10-20-10-2-1 (161×63)"}

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _cell_stats(rows: Sequence[Dict], metric: str, keys: Sequence[str]
                ) -> Dict[Tuple, Tuple[float, float, int]]:
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(float(r[metric]))
    return {k: (*mean_ci(v), len(v)) for k, v in groups.items()}


def _plot_lines(ax, rows, metric, planners, x_key="robots") -> None:
    stats = _cell_stats(rows, metric, ("planner", x_key))
    for p in planners:
        pts = sorted((k[1], v) for k, v in stats.items() if k[0] == p)
        if not pts:
            continue
        xs = [x for x, _ in pts]
        ys = [v[0] for _, v in pts]
        es = [v[1] for _, v in pts]
        ax.errorbar(xs, ys, yerr=es, color=COLOR[p], marker=MARKER[p],
                    markersize=6, linestyle=DASH[p], linewidth=2, capsize=3,
                    elinewidth=1, label=LABEL[p])


def metric_by_robots(rows, metric, ylabel, title, path, planners=PLANNERS,
                     row_filter: Optional[Callable[[Dict], bool]] = None,
                     ylim: Optional[Tuple[float, float]] = None) -> None:
    """Small multiples: one panel per layout, one line per planner, shared y."""
    rows = [r for r in rows if row_filter is None or row_filter(r)]
    layouts = [l for l in LAYOUT_TITLE if any(r["layout"] == l for r in rows)]
    if not layouts:
        return
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.2 * len(layouts), 3.8),
                             sharey=True, squeeze=False)
    for ax, layout in zip(axes[0], layouts):
        _plot_lines(ax, [r for r in rows if r["layout"] == layout], metric, planners)
        ax.set_title(LAYOUT_TITLE[layout])
        ax.set_xlabel("Robots")
        ax.set_xticks(sorted({r["robots"] for r in rows}))
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def throughput_by_skew(rows, path) -> None:
    """Layouts × demand-skew small multiples of throughput."""
    skews = [s for s in ("uniform", "real", "extreme") if any(r["skew"] == s for r in rows)]
    layouts = [l for l in LAYOUT_TITLE if any(r["layout"] == l for r in rows)]
    if not skews or not layouts:
        return
    fig, axes = plt.subplots(len(layouts), len(skews),
                             figsize=(4.2 * len(skews), 3.2 * len(layouts)),
                             sharex=True, sharey=True, squeeze=False)
    for i, layout in enumerate(layouts):
        for j, skew in enumerate(skews):
            ax = axes[i][j]
            sub = [r for r in rows if r["layout"] == layout and r["skew"] == skew]
            _plot_lines(ax, sub, "throughput", PLANNERS)
            ax.set_title(f"{LAYOUT_TITLE[layout].split(' (')[0]} · {skew} demand",
                         fontsize=10)
            if i == len(layouts) - 1:
                ax.set_xlabel("Robots")
            if j == 0:
                ax.set_ylabel("Orders / 1,000 steps")
            ax.set_xticks(sorted({r["robots"] for r in rows}))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Throughput by layout and demand skew", x=0.01, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def sweep_figures(rows, path) -> None:
    """λ on x; two separate charts (congestion, suboptimality), one line per w."""
    ca = [r for r in rows if r["planner"] == "CA-ECBS" and str(r["solved"]) == "True"]
    if not ca:
        return
    ws = sorted({r["w"] for r in ca})
    w_color = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for metric, ax, ylabel in (("congestion", axes[0], "Congestion exposure"),
                               ("suboptimality", axes[1], "Cost / CBS optimum")):
        stats = _cell_stats(ca, metric, ("w", "lam"))
        for i, w in enumerate(ws):
            pts = sorted((k[1], v) for k, v in stats.items() if k[0] == w)
            ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                        yerr=[p[1][1] for p in pts], color=w_color[i % 5],
                        marker="osD^v"[i % 5], markersize=6, linewidth=2,
                        capsize=3, elinewidth=1, label=f"w = {w}")
        ax.set_xlabel("λ (congestion weight)")
        ax.set_ylabel(ylabel)
    axes[1].axhline(1.0, color=INK_2, linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.08))
    layout = ca[0].get("layout") or "synthetic"
    where = LAYOUT_TITLE.get(layout, layout).split(" (")[0]
    fig.suptitle(f"λ / w sweep, CA-ECBS — {where}, {ca[0].get('agents', '?')} agents, "
                 "one-shot instances", x=0.01, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def stress_figure(rows, path, where: str = "") -> None:
    """Throughput change under blocked aisles, one bar per planner."""
    planners = [p for p in PLANNERS if any(r["planner"] == p for r in rows)]
    if not planners:
        return
    drops = []
    for p in planners:
        n = [float(r["throughput"]) for r in rows
             if r["planner"] == p and str(r["disrupted"]) == "False"]
        d = [float(r["throughput"]) for r in rows
             if r["planner"] == p and str(r["disrupted"]) == "True"]
        base = sum(n) / len(n) if n else 0.0
        drops.append(100 * (sum(d) / len(d) - base) / base if base else 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar([LABEL[p] for p in planners], drops,
                  color=[COLOR[p] for p in planners], edgecolor="white", linewidth=2)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.margins(y=0.2)                      # room for the value labels
    for b, v in zip(bars, drops):
        ax.annotate(f"{v:+.0f}%", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, -12 if v < 0 else 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Throughput change (%)")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_title("Throughput change under blocked aisles\n"
                 + (f"{where} · " if where else "") + "vs matched control",
                 loc="left", fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _is_true(value) -> bool:
    return str(value) == "True"


STRESS_NAME = {"custom": "Custom narrow-aisle",
               "warehouse_65x46": "Hand-built warehouse (65×46)",
               "movingai_warehouse": "MovingAI warehouse-10-20-10-2-1"}


def _grid_tables(exp_rows) -> List[str]:
    """Per-layout tables behind the line charts."""
    out: List[str] = []
    for metric, title, fmt in (("throughput", "Throughput (orders / 1,000 steps)", "{:.1f}"),
                               ("success_rate", "Planning success rate", "{:.2f}"),
                               ("congestion_exposure", "Congestion exposure", "{:.3f}"),
                               ("collision_rate", "Collision rate (per 1,000 robot-steps), delay p = 0.05 only", "{:.1f}")):
        rows = exp_rows
        if metric == "collision_rate":
            rows = [r for r in exp_rows if float(r["delay"]) > 0]
        robots = sorted({r["robots"] for r in rows})
        for layout in [l for l in LAYOUT_TITLE if any(r["layout"] == l for r in rows)]:
            stats = _cell_stats([r for r in rows if r["layout"] == layout],
                                metric, ("planner", "robots"))
            out += [f"## {title} — {LAYOUT_TITLE[layout]}", "",
                    "| Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                    "|---|" + "---|" * len(robots)]
            for p in PLANNERS:
                cells = []
                for n in robots:
                    v = stats.get((p, n))
                    cells.append(f"{fmt.format(v[0])} ± {fmt.format(v[1])}" if v else "–")
                out.append(f"| {LABEL[p]} | " + " | ".join(cells) + " |")
            out.append("")
    return out


def _pooled_tables(exp_rows) -> List[str]:
    """Numbers the report quotes pooled over all layouts in the file (Sections 7.1-7.6)."""
    out: List[str] = []
    robots = sorted({r["robots"] for r in exp_rows})
    planners = [p for p in PLANNERS if any(r["planner"] == p for r in exp_rows)]

    def col(planner, metric, **where):
        return [float(r[metric]) for r in exp_rows if r["planner"] == planner
                and all(r[k] == v for k, v in where.items())]

    for metric, title, fmt in (("throughput", "Throughput (orders / 1,000 steps)", "{:.1f}"),
                               ("success_rate", "Planning success rate", "{:.2f}")):
        out += [f"## {title} — all layouts pooled", "",
                "| Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                "|---|" + "---|" * len(robots)]
        for p in planners:
            out.append(f"| {LABEL[p]} | " + " | ".join(
                fmt.format(_mean(col(p, metric, robots=n))) for n in robots) + " |")
        out.append("")

    # Runtime: CPU seconds per planning call, and high-level nodes per call.
    # A failed call is counted at whatever it used before it hit its limit.
    needed = ("mean_plan_time", "total_expansions", "planning_calls")
    if exp_rows and all(k in exp_rows[0] for k in needed):
        def per_call(planner, n):
            sel = [r for r in exp_rows if r["planner"] == planner and r["robots"] == n]
            calls = sum(float(r["planning_calls"]) for r in sel)
            if not calls:
                return "n/a"
            cpu = sum(float(r["mean_plan_time"]) * float(r["planning_calls"]) for r in sel)
            nodes = sum(float(r["total_expansions"]) for r in sel)
            return f"{cpu / calls:.3f} s" + (f" / {nodes / calls:.1f}" if nodes else "")
        out += ["## Runtime — CPU time per planning call / high-level nodes expanded per call, "
                "all layouts pooled", "",
                "All planning calls of all runs in the cell, failed calls included "
                "(they stop at the time limit). Independent A* and prioritized planning "
                "have no high-level search.", "",
                "| Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                "|---|" + "---|" * len(robots)]
        for p in planners:
            out.append(f"| {LABEL[p]} | " + " | ".join(per_call(p, n) for n in robots) + " |")
        out.append("")

    skews = [s for s in ("uniform", "real", "extreme") if any(r["skew"] == s for r in exp_rows)]
    top = robots[-1]
    out += ["## Demand skew — throughput / congestion exposure, all layouts pooled", "",
            "| Planner | " + " | ".join(skews) + " |", "|---|" + "---|" * len(skews)]
    for p in planners:
        out.append(f"| {LABEL[p]} | " + " | ".join(
            f"{_mean(col(p, 'throughput', skew=s)):.0f} / "
            f"{_mean(col(p, 'congestion_exposure', skew=s)):.2f}" for s in skews) + " |")
    if "CA-ECBS" in planners and "ECBS" in planners:
        out += ["", f"CA-ECBS minus ECBS throughput at {top} robots: " + ", ".join(
            f"{_mean(col('CA-ECBS', 'throughput', skew=s, robots=top)) - _mean(col('ECBS', 'throughput', skew=s, robots=top)):+.0f} ({s})"
            for s in skews) + "."]
    out.append("")

    delayed = [r for r in exp_rows if float(r["delay"]) > 0]
    if delayed:
        out += ["## Collision rate (per 1,000 robot-steps), delay p > 0, everything pooled", "",
                "| Planner | Mean ± 95% CI |", "|---|---|"]
        for p in planners:
            m, ci = mean_ci([float(r["collision_rate"]) for r in delayed if r["planner"] == p])
            out.append(f"| {LABEL[p]} | {m:.1f} ± {ci:.1f} |")
        out.append("")

    if "CA-ECBS" in planners:
        key = lambda r: (r["layout"], r["robots"], r["skew"], r["delay"], r["seed"])
        by: Dict[str, Dict] = defaultdict(dict)
        for r in exp_rows:
            by[r["planner"]][key(r)] = r
        out += ["## Paired comparison, CA-ECBS minus baseline (identical instances)", "",
                "| Baseline | n | Throughput difference (95% CI) | "
                "Congestion-exposure difference (95% CI) | Wilcoxon p (throughput / exposure) |",
                "|---|---|---|---|---|"]
        for b in [p for p in planners if p != "CA-ECBS"]:
            shared = sorted(set(by[b]) & set(by["CA-ECBS"]))
            if not shared:
                continue
            cells, ps = [], []
            for metric, fmt in (("throughput", "{:+.1f} (±{:.1f})"),
                                ("congestion_exposure", "{:+.2f} (±{:.2f})")):
                t = [float(by["CA-ECBS"][k][metric]) for k in shared]
                o = [float(by[b][k][metric]) for k in shared]
                m, ci = mean_ci([x - y for x, y in zip(t, o)])
                cells.append(fmt.format(m, ci))
                ps.append(wilcoxon_signed_rank(t, o)[1])
            out.append(f"| {LABEL[b]} | {len(shared)} | {cells[0]} | {cells[1]} | "
                       + " / ".join("< 0.0001" if p < 1e-4 else f"{p:.4f}" for p in ps) + " |")
        e = _mean(col("ECBS", "congestion_exposure"))
        c = _mean(col("CA-ECBS", "congestion_exposure"))
        if e == e and e > 0:
            out += ["", f"Mean congestion exposure: ECBS {e:.3f}, CA-ECBS {c:.3f} "
                        f"({100 * (e - c) / e:.0f}% lower)."]
        out.append("")
    return out


def _path_cost_table(exp_rows) -> List[str]:
    """Path cost per planning call, relative to the shortest-path lower bound.

    Only rows that carry the column (runs made from 10 Nov on) are used. The
    ratio is total path cost over the sum of each robot's own shortest
    distance; that sum is a lower bound on the CBS optimum, so the ratio is an
    upper bound on cost relative to CBS (Section 4.5 of the proposal).
    """
    rows = [r for r in exp_rows if str(r.get("path_cost_ratio", "")) not in ("", "0", "0.0")]
    if not rows:
        return []
    robots = sorted({r["robots"] for r in rows})
    out = []
    for layout in [l for l in LAYOUT_TITLE if any(r["layout"] == l for r in rows)]:
        stats = _cell_stats([r for r in rows if r["layout"] == layout],
                            "path_cost_ratio", ("planner", "robots"))
        out += [f"## Path cost relative to the shortest-path lower bound — {LAYOUT_TITLE[layout]}", "",
                "Total path cost of each solved planning call divided by the sum of every "
                "robot's own shortest distance to its goal. That sum is a lower bound on the "
                "CBS optimum, so 1.10 means at most 10% above optimal. Planners without a "
                "value in a column were not rerun with this metric.", "",
                "| Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                "|---|" + "---|" * len(robots)]
        for p in PLANNERS:
            if not any((p, n) in stats for n in robots):
                continue
            cells = [f"{stats[(p, n)][0]:.3f} ± {stats[(p, n)][1]:.3f}" if (p, n) in stats
                     else "–" for n in robots]
            out.append(f"| {LABEL[p]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def _replan_table(exp_rows, replan_rows) -> List[str]:
    """Throughput by replanning interval h, on the configurations the side runs used."""
    if not replan_rows:
        return []
    combos = {(r["skew"], float(r["delay"]), r["robots"]) for r in replan_rows}
    rows = list(replan_rows) + [
        {**r, "replan_every": r.get("replan_every") or 5} for r in exp_rows
        if (r["skew"], float(r["delay"]), r["robots"]) in combos]
    hs = sorted({int(r["replan_every"]) for r in rows})
    robots = sorted({r["robots"] for r in rows})
    out = ["## Replanning interval h — throughput at " + " / ".join(str(n) for n in robots)
           + " robots, all layouts pooled", "",
           "Configurations: " + ", ".join(sorted({f"{s} skew, delay {d:g}" for s, d, _ in combos}))
           + ". h = 5 rows are from the main grid.", "",
           "| Planner | " + " | ".join(f"h = {h}" for h in hs) + " |",
           "|---|" + "---|" * len(hs)]
    for p in [p for p in PLANNERS if any(r["planner"] == p for r in rows)]:
        cells = []
        for h in hs:
            cells.append(" / ".join(
                f"{_mean([float(r['throughput']) for r in rows if r['planner'] == p and int(r['replan_every']) == h and r['robots'] == n]):.0f}"
                for n in robots))
        out.append(f"| {LABEL[p]} | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _stress_table(tag: str, rows) -> List[str]:
    out = [f"## Blocked-aisle stress test — {STRESS_NAME.get(tag, tag)}", "",
           f"{len({r['seed'] for r in rows})} seeds. Control runs replan at the same "
           "timesteps as the disrupted runs.", "",
           "| Planner | Throughput: control → disrupted | Change | Success: control → disrupted |",
           "|---|---|---|---|"]
    for p in [p for p in PLANNERS if any(r["planner"] == p for r in rows)]:
        n = [r for r in rows if r["planner"] == p and not _is_true(r["disrupted"])]
        d = [r for r in rows if r["planner"] == p and _is_true(r["disrupted"])]
        tn = _mean([float(r["throughput"]) for r in n])
        td = _mean([float(r["throughput"]) for r in d])
        out.append(f"| {LABEL[p]} | {tn:.1f} → {td:.1f} | {100 * (td - tn) / tn:+.1f}% | "
                   f"{_mean([float(r['success_rate']) for r in n]):.2f} → "
                   f"{_mean([float(r['success_rate']) for r in d]):.2f} |")
    out.append("")
    return out


def _sweep_table(tag: str, rows) -> List[str]:
    ca = [r for r in rows if r["planner"] == "CA-ECBS"]
    solved = [r for r in rows if _is_true(r["solved"]) and float(r["optimal_cost"]) > 0]
    layout = rows[0].get("layout") or "synthetic"
    where = LAYOUT_TITLE.get(layout, layout).split(" (")[0]
    lams = sorted({float(r["lam"]) for r in ca})
    out = [f"## λ / w sweep — {where}, {rows[0].get('agents', '?')} agents (`sweep_{tag}.csv`)", "",
           f"One-shot instances. CA-ECBS runs: {len(ca)}, solved: "
           f"{sum(1 for r in ca if _is_true(r['solved']))}; bound violations (all planners): "
           f"{sum(1 for r in rows if not _is_true(r['bound_held']))}. Each cell: congestion "
           "exposure (demand term only) / path cost relative to the CBS optimum.", "",
           "| w | ECBS | " + " | ".join(f"CA-ECBS λ = {l:g}" for l in lams) + " |",
           "|---|---|" + "---|" * len(lams)]

    def cell(sel):
        return (f"{_mean([float(r['congestion']) for r in sel]):.3f} / "
                f"{_mean([float(r['suboptimality']) for r in sel]):.3f}") if sel else "–"

    for w in sorted({float(r["w"]) for r in ca}):
        ecbs_rows = [r for r in solved if r["planner"] == "ECBS" and float(r["w"]) == w]
        out.append(f"| {w:g} | {cell(ecbs_rows)} | " + " | ".join(
            cell([r for r in solved if r["planner"] == "CA-ECBS"
                  and float(r["w"]) == w and float(r["lam"]) == l]) for l in lams) + " |")
    out.append("")
    return out


def _compare_table(datasets: Dict[str, List[Dict]]) -> List[str]:
    """The same grid under different demand models, side by side (Section 7.7).

    Uniform rows do not depend on the demand model at all, so where the two
    sets were run on different machines that line doubles as a machine check.
    """
    out = ["## Demand model comparison — same grid, different product frequencies", "",
           "All layouts, all robot counts, delays and seeds pooled. "
           "Differences are paired over identical instances.", "",
           "| Demand model | Skew | n | ECBS throughput | CA-ECBS throughput | "
           "CA-ECBS − ECBS (95% CI) | ECBS exposure | CA-ECBS exposure | Exposure lower by | "
           "CBS success |", "|---|---|---|---|---|---|---|---|---|---|"]
    key = lambda r: (r["layout"], r["robots"], r["skew"], r["delay"], r["seed"])
    for label, rows in datasets.items():
        skews = [s for s in ("uniform", "real", "extreme") if any(r["skew"] == s for r in rows)]
        for skew in skews + ["all"]:
            sel = [r for r in rows if skew == "all" or r["skew"] == skew]
            by: Dict[str, Dict] = defaultdict(dict)
            for r in sel:
                by[r["planner"]][key(r)] = r
            shared = sorted(set(by["ECBS"]) & set(by["CA-ECBS"]))
            if not shared:
                continue
            e_t = [float(by["ECBS"][k]["throughput"]) for k in shared]
            c_t = [float(by["CA-ECBS"][k]["throughput"]) for k in shared]
            e_x = _mean([float(by["ECBS"][k]["congestion_exposure"]) for k in shared])
            c_x = _mean([float(by["CA-ECBS"][k]["congestion_exposure"]) for k in shared])
            d, ci = mean_ci([c - e for c, e in zip(c_t, e_t)])
            cbs = [float(r["success_rate"]) for r in by["CBS"].values()]
            out.append(
                f"| {label} | {skew} | {len(shared)} | {_mean(e_t):.1f} | {_mean(c_t):.1f} | "
                f"{d:+.1f} (±{ci:.1f}) | {e_x:.3f} | {c_x:.3f} | "
                f"{100 * (e_x - c_x) / e_x:.0f}% | "
                + (f"{_mean(cbs):.2f}" if cbs else "n/a") + " |")
    out.append("")
    labels = list(datasets)
    if len(labels) == 2:
        a, b = (datasets[l] for l in labels)
        full = lambda r: key(r) + (r["planner"],)
        cols = ("throughput", "success_rate", "collisions", "congestion_exposure")
        bb = {full(r): r for r in b if r["skew"] == "uniform"}
        lines = []
        for p in [p for p in PLANNERS if any(r["planner"] == p for r in a)]:
            mine = [r for r in a if r["skew"] == "uniform" and r["planner"] == p and full(r) in bb]
            if not mine:
                continue
            same = sum(all(str(r.get(c)) == str(bb[full(r)].get(c)) for c in cols if c in r)
                       for r in mine)
            lines.append(f"{LABEL[p]} {same} of {len(mine)}")
        if lines:
            out += ["Uniform-demand runs that are identical in the two sets "
                    "(throughput, success rate, collisions and exposure all equal): "
                    + "; ".join(lines) + ".", ""]
    return out


def _dense_table(exp_rows, dense_rows) -> List[str]:
    """60 and 80 robots (experiments_dense.csv) next to 40 robots of the main grid.

    The dense runs use the main grid's design and seeds, without CBS, so the
    40-robot column is the same instances one step lower.
    """
    top = max(r["robots"] for r in exp_rows)
    rows = [r for r in exp_rows if r["robots"] == top] + list(dense_rows)
    robots = sorted({r["robots"] for r in rows})
    planners = [p for p in PLANNERS if any(r["planner"] == p for r in dense_rows)]
    layouts = [l for l in LAYOUT_TITLE if any(r["layout"] == l for r in rows)]
    out = [f"## Higher density — {', '.join(str(n) for n in robots)} robots", "",
           f"{top} robots from `experiments_full.csv`; the rest from "
           "`experiments_dense.csv` (same design and seeds; CBS not run). "
           "Skews, delays and seeds pooled.", ""]
    for metric, title, fmt in (("throughput", "Throughput (orders / 1,000 steps)", "{:.1f}"),
                               ("success_rate", "Planning success rate", "{:.2f}"),
                               ("congestion_exposure", "Congestion exposure", "{:.3f}")):
        out += [f"### {title}", "",
                "| Layout | Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                "|---|---|" + "---|" * len(robots)]
        for layout in layouts:
            stats = _cell_stats([r for r in rows if r["layout"] == layout],
                                metric, ("planner", "robots"))
            for p in planners:
                cells = [(f"{fmt.format(stats[(p, n)][0])} ± {fmt.format(stats[(p, n)][1])}"
                          if (p, n) in stats else "–") for n in robots]
                out.append(f"| {LAYOUT_TITLE[layout].split(' (')[0]} | {LABEL[p]} | "
                           + " | ".join(cells) + " |")
        out.append("")

    key = lambda r: (r["layout"], r["skew"], r["delay"], r["seed"])
    out += ["### Paired differences, all layouts pooled", "",
            "| Robots | Comparison | n | Throughput difference (95% CI) | "
            "Exposure difference (95% CI) | Wilcoxon p (throughput) |",
            "|---|---|---|---|---|---|"]
    for n in robots:
        by: Dict[str, Dict] = defaultdict(dict)
        for r in rows:
            if r["robots"] == n:
                by[r["planner"]][key(r)] = r
        for a, b in (("CA-ECBS", "ECBS"), ("CA-ECBS", "prioritized"), ("prioritized", "ECBS")):
            shared = sorted(set(by[a]) & set(by[b]))
            if not shared:
                continue
            ta = [float(by[a][k]["throughput"]) for k in shared]
            tb = [float(by[b][k]["throughput"]) for k in shared]
            d, ci = mean_ci([x - y for x, y in zip(ta, tb)])
            dx, cix = mean_ci([float(by[a][k]["congestion_exposure"])
                               - float(by[b][k]["congestion_exposure"]) for k in shared])
            _, pv = wilcoxon_signed_rank(ta, tb)
            ptxt = "< 0.0001" if pv < 1e-4 else f"{pv:.4f}"
            out.append(f"| {n} | {LABEL[a]} − {LABEL[b]} | {len(shared)} | "
                       f"{d:+.1f} (±{ci:.1f}) | {dx:+.3f} (±{cix:.3f}) | {ptxt} |")
    out.append("")

    if all(k in rows[0] for k in ("mean_plan_time", "total_expansions", "planning_calls")):
        out += ["### Runtime — CPU time per planning call / high-level nodes per call, "
                "all layouts pooled", "",
                "| Planner | " + " | ".join(f"{n} robots" for n in robots) + " |",
                "|---|" + "---|" * len(robots)]
        for p in planners:
            cells = []
            for n in robots:
                sel = [r for r in rows if r["planner"] == p and r["robots"] == n]
                calls = sum(float(r["planning_calls"]) for r in sel)
                if not calls:
                    cells.append("–")
                    continue
                cpu = sum(float(r["mean_plan_time"]) * float(r["planning_calls"]) for r in sel)
                nodes = sum(float(r["total_expansions"]) for r in sel)
                cells.append(f"{cpu / calls:.3f} s"
                             + (f" / {nodes / calls:.1f}" if nodes else ""))
            out.append(f"| {LABEL[p]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def tables_md(exp_rows, path, sweeps: Optional[Dict[str, List[Dict]]] = None,
              stresses: Optional[Dict[str, List[Dict]]] = None,
              replan_rows: Sequence[Dict] = (),
              compare: Optional[Dict[str, List[Dict]]] = None,
              dense_rows: Sequence[Dict] = ()) -> None:
    """Every number in the report's results section, as Markdown tables."""
    out = ["# Result tables", "",
           "Mean ± 95% CI over all runs in the cell (skews, delays, seeds pooled "
           "unless stated). Generated by `python -m src.experiments.plots`; "
           "do not edit by hand.", ""]
    if exp_rows:
        out += _grid_tables(exp_rows) + _pooled_tables(exp_rows)
        out += _path_cost_table(exp_rows)
        out += _replan_table(exp_rows, list(replan_rows))
    for tag, rows in (stresses or {}).items():
        out += _stress_table(tag, rows)
    for tag, rows in (sweeps or {}).items():
        out += _sweep_table(tag, rows)
    if compare and len(compare) > 1:
        out += _compare_table(compare)
    if exp_rows and dense_rows:
        out += _dense_table(exp_rows, list(dense_rows))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out))


SWEEP_DEFAULTS = ("results/sweep_custom.csv", "results/sweep_warehouse_65x46.csv",
                  "results/sweep_custom_20.csv")
STRESS_DEFAULTS = ("results/stress_warehouse_65x46.csv", "results/stress_custom.csv")
REPLAN_DEFAULTS = ("results/replan_h2.csv", "results/replan_h10.csv")
COMPARE_DEFAULT = "results/zipf_baseline/experiments_full.csv"
DENSE_DEFAULT = "results/experiments_dense.csv"
SOURCE_NAME = {"instacart": "Instacart", "synthetic": "Synthetic Zipf"}


def _demand_label(rows: Sequence[Dict]) -> str:
    """Name a main-grid file by the demand source its non-uniform rows record."""
    sources = sorted({str(r.get("demand_source", "")) for r in rows} - {"uniform", ""})
    return " + ".join(SOURCE_NAME.get(s, s) for s in sources) or "unknown demand"


def _tag(path: str, prefix: str) -> str:
    """results/sweep_custom_20.csv -> custom_20 (names the figure and the table)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MAPForce report figures and tables")
    p.add_argument("--experiments", default="results/experiments_full.csv")
    p.add_argument("--sweep", nargs="*", default=list(SWEEP_DEFAULTS),
                   help="sweep CSVs; each becomes sweep_<name>.png")
    p.add_argument("--stress", nargs="*", default=list(STRESS_DEFAULTS),
                   help="stress-test CSVs; each becomes stress_<name>.png")
    p.add_argument("--replan", nargs="*", default=list(REPLAN_DEFAULTS),
                   help="replanning-interval CSVs (tables only)")
    p.add_argument("--compare", default=COMPARE_DEFAULT,
                   help="main-grid CSV made with another demand model (table only)")
    p.add_argument("--dense", default=DENSE_DEFAULT,
                   help="60 / 80-robot runs (same design, no CBS); adds two figures")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args(argv)
    _style()
    os.makedirs(args.out_dir, exist_ok=True)
    made = []
    out = lambda name: os.path.join(args.out_dir, name)

    rows: List[Dict] = []
    if os.path.exists(args.experiments):
        rows = read_csv(args.experiments)
        metric_by_robots(rows, "throughput", "Orders / 1,000 steps",
                         "Throughput vs fleet size", out("throughput.png"))
        metric_by_robots(rows, "success_rate", "Share of planning calls solved",
                         "Planning success rate vs fleet size", out("success_rate.png"),
                         planners=PLANNERS[:4], ylim=(0, 1.05))
        metric_by_robots(rows, "congestion_exposure", "Mean congestion on planned paths",
                         "Congestion exposure vs fleet size", out("congestion.png"))
        metric_by_robots(rows, "collision_rate", "Collisions / 1,000 robot-steps",
                         "Execution collisions with delays (p = 0.05)",
                         out("collisions.png"), row_filter=lambda r: float(r["delay"]) > 0)
        throughput_by_skew(rows, out("throughput_by_skew.png"))
        made += ["throughput", "success_rate", "congestion", "collisions",
                 "throughput_by_skew"]

    sweeps: Dict[str, List[Dict]] = {}
    for path in args.sweep:
        if os.path.exists(path):
            tag = _tag(path, "sweep_")
            sweeps[tag] = read_csv(path)
            sweep_figures(sweeps[tag], out(f"sweep_{tag}.png"))
            made.append(f"sweep_{tag}")
    stresses: Dict[str, List[Dict]] = {}
    for path in args.stress:
        if os.path.exists(path):
            tag = _tag(path, "stress_")
            stresses[tag] = read_csv(path)
            stress_figure(stresses[tag], out(f"stress_{tag}.png"),
                          where=STRESS_NAME.get(tag, tag))
            made.append(f"stress_{tag}")
    replan_rows = [r for path in args.replan if os.path.exists(path)
                   for r in read_csv(path)]

    compare: Dict[str, List[Dict]] = {}
    if rows and args.compare and os.path.exists(args.compare):
        other = read_csv(args.compare)
        compare = {_demand_label(rows): rows, _demand_label(other): other}

    dense: List[Dict] = []
    if rows and args.dense and os.path.exists(args.dense):
        dense = read_csv(args.dense)
        both = rows + dense
        metric_by_robots(both, "throughput", "Orders / 1,000 steps",
                         "Throughput vs fleet size, 10–80 robots (CBS run up to 40)",
                         out("throughput_dense.png"))
        metric_by_robots(both, "success_rate", "Share of planning calls solved",
                         "Planning success rate, 10–80 robots (CBS run up to 40)",
                         out("success_dense.png"), planners=PLANNERS[:4], ylim=(0, 1.05))
        made += ["throughput_dense", "success_dense"]

    if rows or sweeps or stresses:
        tables_md(rows, out("tables.md"), sweeps, stresses, replan_rows, compare, dense)
        made.append("tables.md")
    print(f"wrote {', '.join(made) or 'nothing (no input CSVs found)'} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
