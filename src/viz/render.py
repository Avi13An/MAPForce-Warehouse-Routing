"""Interactive visualiser: the live demo for the viva.

Owner per the proposal: Bysani (visualiser & demo).

Writes one self-contained HTML file -- no server, no CDN, no build step. Open
it in a browser and it plays back recorded runs: robots moving, the congestion
heatmap underneath, blocked aisles, and running metrics.

The heatmap has two layers, switchable in the page (10 Nov): the static
demand term of the congestion map (which aisles the order data makes busy),
and the live planned-occupancy term (where each panel's robots are planned to
be over the next replanning window), refreshed at every replan. Until 10 Nov
only the static demand layer was drawn.

Planner, layout, robot count and skew are chosen when the file is generated
(command-line flags below); the page replays those runs. Several planners can
be shown **side by side on the same warehouse and order stream** (proposal
objective 5), with one set of controls driving all of them.

    python -m src.viz.render                                   # ECBS vs CA-ECBS
    python -m src.viz.render --planners prioritized CBS ECBS CA-ECBS --robots 20
    python -m src.viz.render --layout warehouse_65x46 --disruptions 6

Recording goes through `run_lifelong(on_frame=...)`, so what you watch is the
exact loop the experiment numbers come from. (Until 10 Nov this module kept
its own copy of that loop, which had drifted: it lacked the assignment,
drop-off and blocked-aisle fixes.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

from ..demand.orders import OrderStream, build_demand_model
from ..experiments.runner import load_layouts, make_planners
from ..lifelong.disruptions import random_aisle_blockages
from ..lifelong.simulator import dock_cells, run_lifelong
from ..planners.common import Cell, GridView
from ..planners.congestion import CongestionMap
from ..tuning.sweep import narrow_aisle_grid, shelves_and_stations

PLANNER_NAMES = ["independent_astar", "prioritized", "CBS", "ECBS", "CA-ECBS"]


def record_run(
    grid: GridView, planner, order_stream: OrderStream,
    start_positions: Sequence[Cell], horizon: int = 120,
    replan_every: int = 5, delay_prob: float = 0.05, seed: int = 0,
    congestion: Optional[CongestionMap] = None, disruptions: Sequence = (),
) -> Dict:
    """Run the real simulator once and keep one frame per timestep."""
    frames: List[Dict] = []
    live: List[Dict[str, float]] = []      # planned-occupancy snapshots

    def on_frame(t, robots, info):
        # The simulator refreshes the congestion map's planned term from every
        # issued plan, so after a replan it holds the fleet's next window.
        if congestion is not None and (info["replanned"] or not live):
            live.append({f"{c[0]},{c[1]}": round(v, 2)
                         for c, v in congestion.planned.items() if v >= 0.01})
        frames.append({
            "t": t,
            "robots": [[r.pos[0], r.pos[1], 1 if r.carrying else 0] for r in robots],
            "goals": [[r.goal[0], r.goal[1]] for r in robots],
            "done": info["done"],
            "coll": info["collisions"],
            "fail": info["failures"],
            "replan": info["replanned"],
            "closed": [list(c) for c in info["closed"]],
            "li": len(live) - 1,
        })

    result = run_lifelong(grid, planner, order_stream, start_positions,
                          horizon=horizon, replan_every=replan_every,
                          delay_prob=delay_prob, seed=seed, congestion=congestion,
                          disruptions=disruptions, on_frame=on_frame)
    return {"frames": frames, "live": live, "calls": result.planning_calls,
            "failures": result.planning_failures, "summary": result.as_row()}


def grid_payload(grid: GridView, congestion: Optional[CongestionMap]) -> Dict:
    heat = {}
    if congestion is not None:
        for y in range(grid.height):
            for x in range(grid.width):
                s = congestion.static.get((x, y), 0.0)   # demand traffic only
                if s > 0:
                    heat[f"{x},{y}"] = round(s, 3)
    return {
        "width": grid.width, "height": grid.height,
        "walls": [[x, y] for y in range(grid.height) for x in range(grid.width)
                  if not grid.passable((x, y))],
        "heat": heat,
    }


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAPForce - {title}</title>
<style>
 body{background:#10130f;color:#e7ece7;font:14px/1.5 system-ui,sans-serif;margin:0;padding:20px}
 h1{font-size:18px;margin:0 0 4px;font-weight:600}
 .sub{color:#96a199;margin-bottom:12px}
 .controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin-bottom:14px}
 .controls label{display:flex;align-items:center;gap:6px;color:#96a199}
 button{background:#1d2320;color:#e7ece7;border:1px solid #394039;border-radius:3px;
   padding:6px 14px;cursor:pointer;font:inherit}
 button:hover{border-color:#57bca7}
 #grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,{min_w}px),1fr));gap:18px}
 .run h2{font-size:15px;margin:0 0 6px;font-weight:600}
 canvas{background:#171c19;border:1px solid #293029;border-radius:4px;width:100%;height:auto;image-rendering:pixelated}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:6px}
 .stat{background:#171c19;border:1px solid #293029;border-radius:3px;padding:4px 8px}
 .stat span{display:block;color:#96a199;font-size:12px}
 .stat b{font-variant-numeric:tabular-nums;color:#57bca7;font-size:16px}
 .legend{color:#96a199;font-size:12px;margin-top:14px}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 12px;vertical-align:middle}
</style></head><body>
<h1>MAPForce &mdash; {title}</h1>
<div class="sub">{subtitle}</div>
<div class="controls">
  <button id="play">Pause</button><button id="rst">Restart</button>
  <label>timestep <input type="range" id="scrub" min="0" value="0" style="width:220px"><b id="tlabel">0</b></label>
  <label>speed <input type="range" id="spd" min="1" max="30" value="10"></label>
  <label>heatmap <select id="heat">
    <option value="both" selected>demand + live traffic</option>
    <option value="demand">demand only</option>
    <option value="live">live traffic only</option>
    <option value="off">off</option></select></label>
  <label><input type="checkbox" id="goals"> show goals</label>
</div>
<div id="grid"></div>
<div class="legend">
  <span class="sw" style="background:#2b332d;margin-left:0"></span>shelf
  <span class="sw" style="background:rgba(222,139,81,.55)"></span>busy aisle (demand)
  <span class="sw" style="background:rgba(122,167,232,.6)"></span>live traffic (planned next window)
  <span class="sw" style="background:#c94f4f"></span>blocked aisle
  <span class="sw" style="background:#57bca7;border-radius:50%"></span>robot (dark dot = carrying)
</div>
<script>
const D = {data};
const N = D.runs[0].frames.length;
const CELL = Math.max(5, Math.min(20, Math.floor({panel_px} / D.width)));
const wall = D.walls;
const PAL = ['#57bca7','#de8b51','#7aa7e8','#d98fc0','#c2cf6a','#e8c35a','#8fd6c5','#e87a7a'];
const grid = document.getElementById('grid');
const panels = D.runs.map(run => {
  const div = document.createElement('div'); div.className = 'run';
  div.innerHTML = `<h2>${run.planner}</h2><canvas></canvas>
    <div class="stats">
      <div class="stat"><span>delivered</span><b data-k="done">0</b></div>
      <div class="stat"><span>per 1,000 steps</span><b data-k="thr">0</b></div>
      <div class="stat"><span>collisions</span><b data-k="coll">0</b></div>
      <div class="stat"><span>failed plans</span><b data-k="fail">0</b></div>
    </div>`;
  grid.appendChild(div);
  const c = div.querySelector('canvas');
  c.width = D.width * CELL; c.height = D.height * CELL;
  const out = {}; div.querySelectorAll('b[data-k]').forEach(b => out[b.dataset.k] = b);
  return {run, x: c.getContext('2d'), out};
});
let i = 0, playing = true, speed = 10;
const scrub = document.getElementById('scrub'); scrub.max = N - 1;

function drawPanel(p){
  const f = p.run.frames[i], x = p.x; if(!f) return;
  x.fillStyle = '#171c19'; x.fillRect(0, 0, D.width*CELL, D.height*CELL);
  const mode = document.getElementById('heat').value;
  if(mode === 'both' || mode === 'demand'){
    for(const k in D.heat){
      const [cx, cy] = k.split(',').map(Number);
      x.fillStyle = 'rgba(222,139,81,' + (Math.min(1, D.heat[k]) * 0.55) + ')';
      x.fillRect(cx*CELL, cy*CELL, CELL, CELL);
    }
  }
  const L = (p.run.live || [])[f.li];
  if(L && (mode === 'both' || mode === 'live')){
    for(const k in L){
      const [cx, cy] = k.split(',').map(Number);
      x.fillStyle = 'rgba(122,167,232,' + (0.15 + Math.min(1, L[k]) * 0.6) + ')';
      x.fillRect(cx*CELL, cy*CELL, CELL, CELL);
    }
  }
  x.fillStyle = '#2b332d';
  for(const [wx, wy] of wall) x.fillRect(wx*CELL, wy*CELL, CELL, CELL);
  x.fillStyle = '#c94f4f';
  for(const [bx, by] of f.closed) x.fillRect(bx*CELL+1, by*CELL+1, CELL-2, CELL-2);
  if(document.getElementById('goals').checked){
    f.goals.forEach(([gx, gy], n) => {
      x.strokeStyle = PAL[n % PAL.length]; x.lineWidth = 1;
      x.strokeRect(gx*CELL+2, gy*CELL+2, CELL-4, CELL-4);
    });
  }
  f.robots.forEach(([rx, ry, car], n) => {
    x.fillStyle = PAL[n % PAL.length];
    x.beginPath(); x.arc(rx*CELL+CELL/2, ry*CELL+CELL/2, CELL*0.36, 0, 6.3); x.fill();
    if(car){ x.fillStyle = '#10130f'; x.fillRect(rx*CELL+CELL/2-2, ry*CELL+CELL/2-2, 4, 4); }
  });
  p.out.done.textContent = f.done;
  p.out.thr.textContent = (1000 * f.done / (f.t + 1)).toFixed(0);
  p.out.coll.textContent = f.coll;
  p.out.fail.textContent = f.fail;
}
function draw(){ panels.forEach(drawPanel); scrub.value = i; document.getElementById('tlabel').textContent = i; }
function tick(){ if(playing){ i = (i + 1) % N; draw(); } setTimeout(tick, 1000 / speed); }
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
document.getElementById('rst').onclick = () => { i = 0; draw(); };
document.getElementById('spd').oninput = e => { speed = +e.target.value; };
scrub.oninput = e => { i = +e.target.value; playing = false; document.getElementById('play').textContent = 'Play'; draw(); };
document.getElementById('heat').onchange = draw;
document.getElementById('goals').onchange = draw;
draw(); tick();
</script></body></html>"""


def build_html(title: str, subtitle: str, payload: Dict) -> str:
    n = len(payload["runs"])
    panel_px = 760 if n == 1 else 600 if n == 2 else 460
    return (HTML.replace("{title}", title)
                .replace("{subtitle}", subtitle)
                .replace("{min_w}", str(min(panel_px, 640)))
                .replace("{panel_px}", str(panel_px))
                .replace("{data}", json.dumps(payload, separators=(",", ":"))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MAPForce visualiser")
    p.add_argument("--planners", nargs="+", default=["ECBS", "CA-ECBS"],
                   choices=PLANNER_NAMES, help="one or more, shown side by side")
    p.add_argument("--planner", choices=PLANNER_NAMES, default=None,
                   help="(old single-planner form; same as --planners X)")
    p.add_argument("--layout", default="custom_narrow_aisle",
                   choices=("custom_narrow_aisle", "warehouse_65x46",
                            "movingai_warehouse", "synthetic"))
    p.add_argument("--robots", type=int, default=20)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--skew", default="extreme", choices=["uniform", "real", "extreme"])
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--disruptions", type=int, default=0,
                   help="number of random blocked-aisle events")
    p.add_argument("--time-limit", type=float, default=5.0, help="CPU s per planning call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/demo.html")
    args = p.parse_args(argv)
    planners_wanted = [args.planner] if args.planner else args.planners

    if args.layout == "synthetic":
        grid = narrow_aisle_grid(rows=13, cols=19)
    else:
        grid = load_layouts(names=(args.layout,))[args.layout]
    shelves, stations = shelves_and_stations(grid)
    free = grid.free_cells()
    random.Random(args.seed).shuffle(free)
    starts = free[: args.robots]

    demand = build_demand_model(shelves, stations, skew=args.skew, seed=args.seed)
    cong = CongestionMap(grid, shelves, stations, demand.shelf_weights)
    events = random_aisle_blockages(grid, args.horizon, args.disruptions,
                                    seed=args.seed, protect=dock_cells(grid, stations)) \
        if args.disruptions else ()

    payload = grid_payload(grid, cong)
    payload["runs"] = []
    for name in planners_wanted:
        print(f"recording {name}, {args.robots} robots, {args.skew} demand...")
        cong.clear_planned()
        planner = make_planners(cong, time_limit=args.time_limit, window=6)[name]
        run = record_run(grid, planner, OrderStream(demand, rate=0.5, seed=args.seed),
                         starts, horizon=args.horizon, delay_prob=args.delay,
                         seed=args.seed, congestion=cong, disruptions=events)
        run["planner"] = (f"{name} — {run['calls'] - run['failures']}/{run['calls']} "
                          f"plans solved")
        payload["runs"].append(run)

    title = " vs ".join(planners_wanted)
    subtitle = (f"{args.layout} {grid.width}×{grid.height} &middot; {args.robots} robots &middot; "
                f"{args.skew} demand ({demand.source}) &middot; delay p={args.delay}"
                + (f" &middot; {args.disruptions} blocked-aisle events" if args.disruptions else "")
                + " &middot; same orders, starts and seed for every panel")
    html = build_html(title, subtitle, payload)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(html) // 1024} KB) -- open it in any browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
