exec(open('proto.py').read())
span = 15*np.abs(S)/std
NB = int(sys.argv[1]); fix = int(sys.argv[2]); P15 = 0.839
edges = np.quantile(span[A], np.linspace(0, 1, NB+1)); edges[0] = 0; edges[-1] = 1e9
def binof(sp): return np.clip(np.searchsorted(edges, sp, side='right') - 1, 0, NB-1)
Z = (R - mean[:, None]) / std[:, None]
G_ = np.linspace(0, 1, 513)
T = np.zeros((NB, 513))
for k in range(NB):
    rows = A[binof(span[A]) == k]; qq = q[rows]; ar = np.arange(len(rows))
    keep = np.ones_like(qq, bool)
    keep[ar, np.argmax(qq == 0, axis=1)] = False
    keep[ar, np.argmax(qq == qq.max(1, keepdims=True), axis=1)] = False
    T[k] = np.quantile(Z[rows][keep], G_)
def gen(idx):
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]; k = binof(span[idx]); n = len(idx); ar = np.arange(n)
    top = np.where(rng.random(n) < P15, 15, 14)
    # truncation: levels in [0, top] <=> z in [zlo, zhi] (oriented by the sign of the scale)
    za = (b + (-0.5) * sc - m) / s; zb = (b + (top + 0.5) * sc - m) / s
    zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
    u = rng.random((n, 64)); z = np.empty_like(u)
    for kk in range(NB):
        r = k == kk
        Flo = np.interp(zlo[r], T[kk], G_); Fhi = np.interp(zhi[r], T[kk], G_)
        uu = Flo[:, None] + u[r] * (Fhi - Flo)[:, None]
        z[r] = np.interp(uu, G_, T[kk])
    la = rng.integers(0, 64, n); lb = (la + 1 + rng.integers(0, 63, n)) % 64
    for it in range(fix + 1):
        x = m[:, None] + s[:, None] * z
        qq = np.clip(np.round((x - b[:, None]) / sc[:, None]), 0, top[:, None]); qq[ar, la] = 0; qq[ar, lb] = top
        y = qq * sc[:, None] + b[:, None]
        if it < fix:
            ym, ys = y.mean(1), y.std(1)
            z = (z - (ym - m)[:, None] / s[:, None]) * (s / ys)[:, None]
    return y
if __name__ == '__main__' and 'diag' not in sys.argv: score(gen(A), f'V5 pin both rails, truncated tables ({NB}), {fix} fixes')
