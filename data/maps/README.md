# Maps

| File | Size | Free cells | What it is |
|---|---|---|---|
| `warehouse-10-20-10-2-1.map` | 161 × 63 | 5,699 | The real `warehouse-10-20-10-2-1` map from the MovingAI MAPF benchmark suite (Stern et al., 2019). Layout key `movingai_warehouse`. |
| `warehouse-65x46-handbuilt.map` | 65 × 46 | 1,391 | A smaller warehouse written by hand in the MovingAI file format. It is **not** a MovingAI benchmark map. Layout key `warehouse_65x46`. |

## Where the real map comes from

movingai.com was not reachable from the machine that added the map, so the
file was taken from the LaCAM3 repository, which redistributes the MovingAI
maps unchanged: `scripts/map/warehouse-10-20-10-2-1.map` in
https://github.com/Kei18/lacam3, commit
`1c3ecf8ade40298f16047544634e9c0c1cf3b93c`.

- SHA-256: `c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0`
- Header `height 63`, `width 161`; 5,699 free cells. These match the size
  published for this map in the MAPF literature (for example, 161 × 63 in
  arXiv:2407.09451, Section 3.1).

To check it against the official copy, download `mapf-map.zip` from
https://movingai.com/benchmarks/mapf/index.html and compare hashes.

## Why there are two warehouse maps

Until 30 Sep 2026 the file named `warehouse-10-20-10-2-1.map` was the
65 × 46 hand-built map, and every result labelled "MovingAI warehouse" came
from it. It was renamed to say what it is, and the results made on it were
relabelled `warehouse_65x46` (see `results/README.md`). The real benchmark
map was added under its proper name, and the main experiment grid was run on
it as a third layout.

## How the loader reads a map

`src/environment/map_loader.py`:

- `.`, `G` and `S` are free; anything else (`@`, `T`, `O`, `W`) is blocked.
- A **pick spot** is a free cell beside a blocked cell that is not part of
  the map's outer border wall. The real map is fenced by a ring of `T`; without
  that exclusion every cell along the edge would have become a pick spot.
- **Packing stations** are every 4th cell of the second-to-last row
  (16 on the 65 × 46 map, 40 on the real map).
