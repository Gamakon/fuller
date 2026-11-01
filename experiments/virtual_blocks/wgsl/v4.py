exec(open('proto.py').read())
span = 15*np.abs(S)/std
NB = int(sys.argv[1]); fix = int(sys.argv[2])
edges = np.quantile(span[A], np.linspace(0, 1, NB+1)); edges[0] = 0; edges[-1] = 1e9
def binof(sp): return np.clip(np.searchsorted(edges, sp, side='right') - 1, 0, NB-1)
Z = (R - mean[:, None]) / std[:, None]
T = np.zeros((NB, 513))
for k in range(NB):
    rows = A[binof(span[A]) == k]
    qq = q[rows]; first0 = np.argmax(qq == 0, axis=1)
    keep = np.ones_like(qq, bool); keep[np.arange(len(rows)), first0] = False
    T[k] = np.quantile(Z[rows][keep], np.linspace(0, 1, 513))
def gen(idx):
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]; k = binof(span[idx])
    u = rng.random((len(idx), 64)); z = np.empty_like(u)
    for kk in range(NB):
        r = k == kk; z[r] = np.interp(u[r], np.linspace(0, 1, 513), T[kk])
    lane = rng.integers(0, 64, len(idx)); ar = np.arange(len(idx))
    for it in range(fix + 1):
        x = m[:, None] + s[:, None] * z
        qq = np.clip(np.round((x - b[:, None]) / sc[:, None]), 0, 15); qq[ar, lane] = 0
        y = qq * sc[:, None] + b[:, None]
        if it < fix:   # nudge z so the quantised block's mean and std move toward the stored m_mean, m_std
            ym, ys = y.mean(1), y.std(1)
            z = (z - (ym - m)[:, None] / s[:, None]) * (s / ys)[:, None]
    return y
score(gen(A), f'V4 z tables per span bin ({NB}), {fix} mean/std fixes')
