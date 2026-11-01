exec(open('proto.py').read())
# pooled z of non-extreme cells (exclude level-0 cells), as a quantile table
Z = (R - mean[:, None]) / std[:, None]
zpool = Z[q != 0]
tab = np.quantile(zpool, np.linspace(0, 1, 1025))
def gen(idx, pin=True, tab=tab):
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]
    u = rng.random((len(idx), 64))
    z = np.interp(u, np.linspace(0, 1, len(tab)), tab)
    x = m[:, None] + s[:, None] * z
    qq = np.clip(np.round((x - b[:, None]) / sc[:, None]), 0, 15)
    if pin:
        lane = rng.integers(0, 64, len(idx)); qq[np.arange(len(idx)), lane] = 0
    return qq * sc[:, None] + b[:, None]
score(R[A], 'control real A vs real B')
score(gen(A, pin=False), 'V1 pooled z, no pin')
score(gen(A), 'V2 pooled z, pin one lane at level 0')
