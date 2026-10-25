"""Gates for the visualiser: it must record the real simulator, not a copy."""

import json
import random

from src.demand.orders import OrderStream, build_demand_model
from src.experiments.runner import make_planners
from src.lifelong.simulator import run_lifelong
from src.planners.congestion import CongestionMap
from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations
from src.viz.render import build_html, grid_payload, record_run

GRID = narrow_aisle_grid(rows=11, cols=15)
SHELVES, STATIONS = shelves_and_stations(GRID)


def _inputs():
    demand = build_demand_model(SHELVES, STATIONS, skew="extreme", seed=0)
    cong = CongestionMap(GRID, SHELVES, STATIONS, demand.shelf_weights)
    free = GRID.free_cells()
    random.Random(1).shuffle(free)
    return demand, cong, free[:6]


def test_replay_matches_the_simulator_exactly():
    demand, cong, starts = _inputs()
    planner = make_planners(cong, time_limit=5.0, window=6)["prioritized"]
    run = record_run(GRID, planner, OrderStream(demand, 0.5, 0), starts,
                     horizon=40, delay_prob=0.05, seed=0, congestion=cong)
    cong.clear_planned()
    direct = run_lifelong(GRID, planner, OrderStream(demand, 0.5, 0), starts,
                          horizon=40, delay_prob=0.05, seed=0, congestion=cong)
    assert len(run["frames"]) == 40
    assert run["frames"][-1]["done"] == direct.completed_orders
    assert run["frames"][-1]["coll"] == direct.collisions
    assert run["calls"] == direct.planning_calls


def test_html_is_self_contained_and_carries_every_panel():
    demand, cong, starts = _inputs()
    payload = grid_payload(GRID, cong)
    payload["runs"] = []
    for name in ("ECBS", "CA-ECBS"):
        cong.clear_planned()
        planner = make_planners(cong, time_limit=5.0, window=6)[name]
        run = record_run(GRID, planner, OrderStream(demand, 0.5, 0), starts,
                         horizon=10, congestion=cong)
        run["planner"] = name
        payload["runs"].append(run)
    html = build_html("t", "s", payload)
    assert "{data}" not in html and "{title}" not in html
    assert "<script src" not in html                  # no CDN, works offline
    blob = html.split("const D = ", 1)[1].split(";\n", 1)[0]
    assert [r["planner"] for r in json.loads(blob)["runs"]] == ["ECBS", "CA-ECBS"]


def test_live_traffic_layer_is_recorded_per_replan():
    """The heatmap's live layer (28 Oct) is the congestion map's planned
    term, snapshotted at every replan, and every frame points at one."""
    demand, cong, starts = _inputs()
    planner = make_planners(cong, time_limit=5.0, window=6)["ECBS"]
    run = record_run(GRID, planner, OrderStream(demand, 0.5, 0), starts,
                     horizon=20, replan_every=5, seed=0, congestion=cong)
    assert len(run["live"]) == 4                      # replans at t = 0, 5, 10, 15
    assert all(0 <= f["li"] < len(run["live"]) for f in run["frames"])
    assert run["frames"][0]["li"] == 0 and run["frames"][-1]["li"] == 3
    assert any(run["live"][0].values())
    assert all(0 < v <= 1 for snap in run["live"] for v in snap.values())
    html = build_html("t", "s", {**grid_payload(GRID, cong), "runs": [run]})
    assert "live traffic" in html
