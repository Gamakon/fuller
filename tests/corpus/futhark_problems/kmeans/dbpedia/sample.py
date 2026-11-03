#!/usr/bin/env python3
"""THE K-MEANS FITNESS DATA (Andrew, 2026-11-03): 1,000 row ids drawn at random (seed 7013) from UltraDim's
dbpedia-openai-1M-angular embeddings (990,000 rows of 1,536 f32), and 5 consecutive rows read from each, 5,000 rows.
Writes `row_ids.txt` (the 1,000 start rows) and `sample_5000x1536.in` (Futhark's binary format, [5000][1536]f32, the
vectors as stored: already unit length). Deterministic: the same ids and
bytes every run.

    python3 -I sample.py [/path/to/dbpedia-openai-1M-angular.parquet]
"""
import os
import struct
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/dev/minkymorgan/qdrant/benchmark_data/dbpedia-openai-1M-angular.parquet")
STARTS, RUN, SEED = 1000, 5, 7013

f = pq.ParquetFile(SRC)
n = f.metadata.num_rows
rng = np.random.default_rng(SEED)
starts = np.sort(rng.choice(n - RUN + 1, STARTS, replace=False))
np.savetxt(os.path.join(HERE, "row_ids.txt"), starts, fmt="%d", header=f"{STARTS} start rows of {RUN} consecutive rows each, seed {SEED}, from {os.path.basename(SRC)} ({n} rows)")

# read row groups once; take the runs from each group's slice
bounds = np.cumsum([0] + [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)])
rows = np.zeros((STARTS * RUN, 1536), dtype=np.float32)
k = 0
for g in range(f.metadata.num_row_groups):
    lo, hi = bounds[g], bounds[g + 1]
    mine = starts[(starts >= lo) & (starts < hi)]
    if len(mine) == 0:
        continue
    tbl = f.read_row_group(g, columns=["vector"])
    vec = np.stack(tbl.column("vector").to_numpy(zero_copy_only=False)).astype(np.float32)
    for s in mine:
        a = s - lo
        if a + RUN > len(vec):  # a run crossing a row-group boundary: read the rest from the next group
            nxt = f.read_row_group(g + 1, columns=["vector"]).column("vector").to_numpy(zero_copy_only=False)
            vec2 = np.concatenate([vec[a:], np.stack(nxt[: a + RUN - len(vec)]).astype(np.float32)])
            rows[k : k + RUN] = vec2
        else:
            rows[k : k + RUN] = vec[a : a + RUN]
        k += RUN
assert k == STARTS * RUN, k
with open(os.path.join(HERE, "sample_5000x1536.in"), "wb") as out:
    out.write(b"b\x02\x02 f32" + struct.pack("<qq", rows.shape[0], rows.shape[1]))
    out.write(rows.tobytes())
norms = np.linalg.norm(rows, axis=1)
print(f"rows {rows.shape}, finite {np.isfinite(rows).all()}, norm min {norms.min():.4f} max {norms.max():.4f} (unit vectors: {np.allclose(norms, 1, atol=1e-3)})")
