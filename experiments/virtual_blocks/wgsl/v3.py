exec(open('proto.py').read())
span = 15*np.abs(S)/std
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 12
edges = np.quantile(span[A], np.linspace(0, 1, NB+1)); edges[0] = 0; edges[-1] = 1e9
def binof(sp): return np.clip(np.searchsorted(edges, sp, side='right') - 1, 0, NB-1)
# level-space position t = q in 0..15 (0 is the pinned extreme); fit per span bin the quantile table of q for the
# non-pinned cells (drop exactly one level-0 cell per block, the pinned one)
def tables(idx):
    T = np.zeros((NB, 257))
    for k in range(NB):
        rows = idx[binof(span[idx]) == k]
        qq = q[rows].copy()
        # remove one zero per row
        first0 = np.argmax(qq == 0, axis=1); keep = np.ones_like(qq, bool); keep[np.arange(len(rows)), first0] = False
        vals = qq[keep] + rng.random(keep.sum()) - 0.5      # dequantise jitter so the table is continuous
        T[k] = np.quantile(vals, np.linspace(0, 1, 257))
    return T
T = tables(A)
def gen(idx):
    sc, b = S[idx], B[idx]; k = binof(span[idx])
    u = rng.random((len(idx), 64))
    lv = np.empty_like(u)
    for kk in range(NB):
        r = k == kk; lv[r] = np.interp(u[r], np.linspace(0, 1, 257), T[kk])
    qq = np.clip(np.round(lv), 0, 15)
    lane = rng.integers(0, 64, len(idx)); qq[np.arange(len(idx)), lane] = 0
    return qq * sc[:, None] + b[:, None]
score(gen(A), f'V3 level tables per span bin ({NB} bins)')
