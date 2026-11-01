"""Score generated blocks against the real ones with faxl-core's fitness.py, imported VERBATIM from their handover
(never copied or edited), plus the distribution statistics of their README table.

    python3 score.py WORK [--controls] [--seeds 5] SET.gen.bin ...

The fitness is Python (scikit-learn 1.8 HistGradientBoostingClassifier, scipy): acceptable as the scorer for now.
`objective` is their number to minimise: the worse |AUC - 0.5| of the 12-feature and the raw-64 discriminators,
each a 3-fold cross-validated AUC (held-out folds). The spread is over classifier seeds (the `seed` argument of
block_auc/raw_auc: it draws the balanced subsample and the classifier's random_state).

Statistics (our definitions; faxl-core did not state theirs): z = the block standardised by its own mean and std;
spikes = cells with |z| > 2.5 a block; kurtosis = scipy's excess kurtosis of z, its p50 and p99 over blocks; rails =
cells at nibble level 0 or 15 of the block's own grid.
"""
import pathlib
import struct
import sys

import numpy as np
from scipy import stats

FAXL = pathlib.Path("/Users/andrewmorgan/Dev/gamakon/faxl-core/docs/virtual-blocks")
sys.path.insert(0, str(FAXL))
import fitness  # noqa: E402  (faxl-core's canonical fitness, verbatim)


def read_fut(path):
    """One f32 array in Futhark's binary format."""
    b = pathlib.Path(path).read_bytes()
    assert b[0:1] == b"b" and b[1] == 2, f"{path}: not Futhark binary format v2"
    rank, ty = b[2], b[3:7]
    assert ty == b" f32", f"{path}: type {ty!r}"
    dims = struct.unpack(f"<{rank}q", b[7:7 + 8 * rank])
    n = int(np.prod(dims))
    body = b[7 + 8 * rank:]
    # An executable's standard output ends with a newline after the value; nothing else may follow it.
    assert len(body) >= 4 * n and not body[4 * n:].strip(), f"{path}: {len(body) - 4 * n} bytes after the value"
    return np.frombuffer(body[:4 * n], dtype="<f4").reshape(dims)


def dist_stats(X, scale=None, bias=None):
    X = np.asarray(X, dtype=np.float64)
    Z = (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-12)
    k = stats.kurtosis(Z, axis=1)
    out = {"spikes": float((np.abs(Z) > 2.5).sum(1).mean()),
           "kurt_p50": float(np.percentile(k, 50)), "kurt_p99": float(np.percentile(k, 99))}
    if scale is not None:
        q = np.round((X - bias[:, None]) / scale[:, None])
        out["rails"] = float(((q == 0) | (q == 15)).sum(1).mean())
    return out


def auc_spread(R, G, seeds):
    rows = []
    for s in seeds:
        a1, _ = fitness.block_auc(R, G, seed=s)
        a2, _ = fitness.raw_auc(R, G, seed=s)
        rows.append((a1, a2, max(abs(a1 - 0.5), abs(a2 - 0.5))))
    return np.array(rows)


def fmt(rows):
    m, sd = rows.mean(0), rows.std(0, ddof=1) if len(rows) > 1 else np.zeros(3)
    return (f"summary AUC {m[0]:.4f} +- {sd[0]:.4f}, raw AUC {m[1]:.4f} +- {sd[1]:.4f}, "
            f"objective {m[2]:.4f} +- {sd[2]:.4f} (n={len(rows)} classifier seeds)")


def main(argv):
    work = pathlib.Path(argv[0])
    seeds = range(5)
    files = []
    controls = False
    i = 1
    while i < len(argv):
        if argv[i] == "--controls":
            controls = True
        elif argv[i] == "--seeds":
            seeds = range(int(argv[i + 1]))
            i += 1
        else:
            files.append(argv[i])
        i += 1
    d = np.load(FAXL / "fitness_data.npz")
    R, s, b = d["real_blocks"], d["real_scale"], d["real_bias"]
    print(f"real (12000 blocks): {dist_stats(R, s, b)}")
    if controls:
        # Their two controls, exactly as fitness.py's __main__ builds them.
        rng = np.random.default_rng(1)
        bad = (rng.standard_normal(R.shape) * R.std(1, keepdims=True) + R.mean(1, keepdims=True)).astype(np.float32)
        obj, det = fitness.objective(R, bad)
        print(f"control gaussian: objective {obj:.4f} {det}")
        half = len(R) // 2
        obj2, det2 = fitness.objective(R[:half], R[half:])
        print(f"control real vs real: objective {obj2:.4f} {det2}")
        print(f"control real vs real, spread: {fmt(auc_spread(R[:half], R[half:], seeds))}")
    for f in files:
        G = read_fut(work / f)
        assert G.shape == R.shape, f"{f}: shape {G.shape}"
        obj, det = fitness.objective(R, G)
        print(f"{f}: canonical objective {obj:.4f} {det}")
        print(f"{f}: {fmt(auc_spread(R, G, seeds))}")
        print(f"{f}: {dist_stats(G, s, b)}")


if __name__ == "__main__":
    main(sys.argv[1:])
