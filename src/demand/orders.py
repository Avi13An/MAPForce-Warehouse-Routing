"""Demand data: real order frequencies -> shelf slotting -> a task stream.

Owner per the proposal: Eeshan (demand data). Written here to unblock the
lifelong loop and the experiment runner; it needs reviewing and taking over.

Three stages, matching Section 4.2:

1. **Product frequencies.** Read from the Instacart Market Basket Analysis
   dataset if it is present under `data/instacart/`, otherwise fall back to a
   Zipf law over synthetic products. `DemandModel.source` records which one
   produced any given run, and the experiment CSVs carry it in their
   `demand_source` column.
2. **Shelf slotting.** Products are assigned to shelf cells. Two policies:
   `random` (the usual assumption) and `popularity_zoned` (hot products
   clustered together), which is what produces the 'extreme' skew level.
3. **Order stream.** Orders are drawn from the frequency distribution and
   turned into pick-and-drop tasks.

The Instacart download is ~600 MB of CSV. Only `order_products__prior.csv`
(product_id counts) and `products.csv` are needed, and only the product-id
column of the first, so the loader streams and counts rather than loading
frames into memory.
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..planners.common import Cell

DEFAULT_INSTACART_DIR = os.path.join("data", "instacart")


# ---------------------------------------------------------------- frequencies
FREQUENCY_CACHE = "product_frequencies.csv"
_RAW_NAME = "order_products__prior.csv"
_memo: Dict[Tuple[str, int], Dict[int, float]] = {}


def _open_raw(data_dir: str):
    """The raw order file as a text stream, from the .csv or any zip holding it."""
    import io
    import zipfile

    plain = os.path.join(data_dir, _RAW_NAME)
    if os.path.exists(plain):
        return open(plain, newline="")
    # Kaggle ships either `order_products__prior.csv.zip` (competition page)
    # or one `archive.zip` holding every table (dataset page). Take the first
    # zip in the folder that contains the file, the conventional name first.
    if not os.path.isdir(data_dir):
        return None
    zips = sorted(n for n in os.listdir(data_dir) if n.lower().endswith(".zip"))
    zips.sort(key=lambda n: n != _RAW_NAME + ".zip")
    for name in zips:
        try:
            zf = zipfile.ZipFile(os.path.join(data_dir, name))
        except (zipfile.BadZipFile, OSError):
            continue
        inner = next((n for n in zf.namelist() if n.endswith(_RAW_NAME)), None)
        if inner is not None:
            return io.TextIOWrapper(zf.open(inner), newline="")
        zf.close()
    return None


def count_instacart_orders(data_dir: str = DEFAULT_INSTACART_DIR) -> Optional[Counter]:
    """How many times each product appears in the prior orders.

    Reads `product_frequencies.csv` (product_id,count) if it exists -- a few
    hundred KB, small enough to commit so teammates never need the raw
    download. Otherwise streams the raw ~550 MB `order_products__prior.csv`
    (or the `.csv.zip` Kaggle ships) once and writes that file.
    """
    cache = os.path.join(data_dir, FREQUENCY_CACHE)
    counts: Counter = Counter()
    if os.path.exists(cache):
        with open(cache, newline="") as f:
            for row in csv.DictReader(f):
                counts[int(row["product_id"])] = int(row["count"])
        return counts or None

    stream = _open_raw(data_dir)
    if stream is None:
        return None
    with stream as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return None
        try:
            col = header.index("product_id")
        except ValueError:
            col = 1
        for row in reader:
            if len(row) > col:
                counts[int(row[col])] += 1
    if not counts:
        return None

    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id", "count"])
        for pid, c in counts.most_common():
            w.writerow([pid, c])
    return counts


def load_instacart_frequencies(
    data_dir: str = DEFAULT_INSTACART_DIR, top_n: int = 2000
) -> Optional[Dict[int, float]]:
    """Product order frequencies from the Instacart dataset, or None if absent.

    Returns the `top_n` most ordered products as a normalised distribution --
    the tail is irrelevant to congestion because those shelves are barely
    visited. Memoised per process: the experiment runner builds a demand model
    for every (layout, skew, seed), and re-reading the file each time would
    dominate the run.
    """
    key = (os.path.abspath(data_dir), top_n)
    if key in _memo:
        return dict(_memo[key])
    counts = count_instacart_orders(data_dir)
    if not counts:
        return None
    top = counts.most_common(top_n)
    total = sum(c for _, c in top)
    _memo[key] = {int(pid): c / total for pid, c in top}
    return dict(_memo[key])


def demand_skew_summary(freqs: Dict[int, float]) -> Dict[str, float]:
    """Numbers for the report's 'does grocery skew transfer?' argument.

    share_top_20pct: fraction of order volume taken by the top 20% of
    products (the Pareto / ABC-analysis figure warehouses quote).
    zipf_exponent: least-squares slope of log(frequency) on log(rank).
    """
    import math

    ranked = sorted(freqs.values(), reverse=True)
    n = len(ranked)
    if n < 2:
        return {"products": n, "share_top_20pct": 1.0, "zipf_exponent": 0.0}
    top = max(1, n // 5)
    share = sum(ranked[:top]) / sum(ranked)
    xs = [math.log(i + 1) for i in range(n)]
    ys = [math.log(v) for v in ranked]
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    return {"products": n, "share_top_20pct": round(share, 4),
            "zipf_exponent": round(-slope, 3)}


def synthetic_frequencies(n_products: int = 2000, exponent: float = 1.0,
                          seed: int = 0) -> Dict[int, float]:
    """Zipf stand-in for the Instacart distribution.

    Real grocery demand is heavy-tailed; a Zipf law reproduces the shape well
    enough to develop against. Clearly marked as synthetic by
    `DemandModel.source` so it can never be mistaken for the real thing in a
    results table.
    """
    raw = {i: 1.0 / ((i + 1) ** exponent) for i in range(n_products)}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# ------------------------------------------------------------------- slotting
@dataclass
class DemandModel:
    """Which shelf gets visited how often, and where the orders come from."""

    shelf_weights: Dict[Cell, float]
    stations: List[Cell]
    source: str                      # "instacart" or "synthetic"
    policy: str                      # slotting policy used
    skew: str                        # "uniform" | "real" | "extreme"
    product_of_shelf: Dict[Cell, int] = field(default_factory=dict)

    def sample_shelf(self, rng: random.Random) -> Cell:
        cells = list(self.shelf_weights)
        weights = [self.shelf_weights[c] for c in cells]
        return rng.choices(cells, weights=weights, k=1)[0]

    def sample_station(self, rng: random.Random) -> Cell:
        return rng.choice(self.stations)

    def provenance(self) -> str:
        """One line for the report, so no figure is ever unattributed."""
        return f"{self.source} frequencies, {self.policy} slotting, {self.skew} skew"


def build_demand_model(
    shelves: Sequence[Cell],
    stations: Sequence[Cell],
    skew: str = "real",
    policy: str = "random",
    data_dir: str = DEFAULT_INSTACART_DIR,
    seed: int = 0,
) -> DemandModel:
    """Assign products to shelves and derive a per-shelf visit frequency.

    skew:
      uniform  -- every shelf equally likely (the control, as in standard MAPF
                  benchmarks, where starts and goals are uniform random)
      real     -- the measured product distribution, slotted by `policy`
      extreme  -- the top products forced into one zone
    """
    shelves = list(shelves)
    stations = list(stations)
    rng = random.Random(seed)

    if not shelves:
        return DemandModel({}, stations, "none", policy, skew)

    if skew == "uniform":
        w = 1.0 / len(shelves)
        return DemandModel({s: w for s in shelves}, stations, "uniform", policy, skew)

    freqs = load_instacart_frequencies(data_dir)
    source = "instacart"
    if freqs is None:
        freqs = synthetic_frequencies(n_products=max(len(shelves), 200))
        source = "synthetic"

    ranked_products = sorted(freqs, key=lambda p: -freqs[p])[: len(shelves)]

    if skew == "extreme" or policy == "popularity_zoned":
        # Hot products go to the shelves nearest one corner, which is what makes
        # a hot zone rather than diffuse popularity.
        anchor = shelves[0]
        ordered_shelves = sorted(
            shelves, key=lambda c: abs(c[0] - anchor[0]) + abs(c[1] - anchor[1])
        )
        policy = "popularity_zoned"
    else:
        ordered_shelves = list(shelves)
        rng.shuffle(ordered_shelves)

    product_of_shelf: Dict[Cell, int] = {}
    raw: Dict[Cell, float] = {}
    for shelf, product in zip(ordered_shelves, ranked_products):
        product_of_shelf[shelf] = product
        raw[shelf] = freqs[product]

    total = sum(raw.values())
    weights = {c: v / total for c, v in raw.items()} if total else {}
    for s in shelves:
        weights.setdefault(s, 0.0)

    return DemandModel(weights, stations, source, policy, skew, product_of_shelf)


# --------------------------------------------------------------- order stream
@dataclass
class Task:
    """One order: go to `shelf`, pick, then carry to `station`."""

    task_id: int
    shelf: Cell
    station: Cell
    released_at: int
    assigned_at: int = -1
    completed_at: int = -1

    @property
    def completed(self) -> bool:
        return self.completed_at >= 0


class OrderStream:
    """A continuous stream of orders drawn from the demand distribution.

    Orders arrive at a fixed rate rather than all at once -- that is what makes
    the problem lifelong rather than a sequence of one-shot MAPF instances, and
    it is what the throughput metric in Section 4.5 is measured against.
    """

    def __init__(self, demand: DemandModel, rate: float = 1.0, seed: int = 0) -> None:
        self.demand = demand
        self.rate = rate                 # expected new orders per timestep
        self.rng = random.Random(seed)
        self._next_id = 0
        self._backlog: float = 0.0

    def release(self, timestep: int) -> List[Task]:
        """Orders arriving at this timestep."""
        self._backlog += self.rate
        out: List[Task] = []
        while self._backlog >= 1.0:
            self._backlog -= 1.0
            out.append(Task(
                task_id=self._next_id,
                shelf=self.demand.sample_shelf(self.rng),
                station=self.demand.sample_station(self.rng),
                released_at=timestep,
            ))
            self._next_id += 1
        return out
