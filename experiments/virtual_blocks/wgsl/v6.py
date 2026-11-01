# Feedback-free: v5's draw with NO correction pass, then ONE analytic correction from a table of the expected
# (mean shift, std ratio) of the quantised block, indexed by (span bin, position of the grid around the mean).
import sys; sys.argv = ['v5', '16', '0', 'diag']
exec(open('v5.py').read())
NZ = int(__import__('os').environ.get('NZ', '16'))
def feats_(idx, top):
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]
    za = (b - 0.5 * sc - m) / s; zb = (b + (top + 0.5) * sc - m) / s
    return binof(span[idx]), np.minimum(za, zb), np.maximum(za, zb)
def gen0(idx, rng_):
    global rng
    rng = rng_; return gen(idx)
# fit: generate half A with fix=0 under one seed, measure the realised shift and ratio per block
r0 = np.random.default_rng(101)
Y = gen0(A, r0)
qtop = np.clip(np.round((Y - B[A][:, None]) / S[A][:, None]), 0, 15).max(1)
k, zl, zh = feats_(A, qtop)
dm = (Y.mean(1) - mean[A]) / std[A]; ds = Y.std(1) / std[A]
el = np.quantile(zl, np.linspace(0, 1, NZ + 1)); eh = np.quantile(zh, np.linspace(0, 1, NZ + 1))
il = np.clip(np.searchsorted(el, zl) - 1, 0, NZ - 1); ih = np.clip(np.searchsorted(eh, zh) - 1, 0, NZ - 1)
TM = np.zeros((NB, NZ, NZ)); TS = np.ones((NB, NZ, NZ))
for a in range(NB):
    for i in range(NZ):
        for j in range(NZ):
            sel = (k == a) & (il == i) & (ih == j)
            if sel.sum() >= 5: TM[a, i, j] = dm[sel].mean(); TS[a, i, j] = ds[sel].mean()
print('cells filled', (TS != 1).mean())
# generate with the analytic correction: z' = (z - TM) / TS, one shot, then quantise
def gen_corr(idx, rng_):
    global rng
    rng = rng_
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]; kk = binof(span[idx]); n = len(idx); ar = np.arange(n)
    top = np.where(rng.random(n) < P15, 15, 14)
    za = (b - 0.5 * sc - m) / s; zb = (b + (top + 0.5) * sc - m) / s
    zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
    u = rng.random((n, 64)); z = np.empty_like(u)
    for kq in range(NB):
        r = kk == kq
        Flo = np.interp(zlo[r], T[kq], G_); Fhi = np.interp(zhi[r], T[kq], G_)
        z[r] = np.interp(Flo[:, None] + u[r] * (Fhi - Flo)[:, None], G_, T[kq])
    i_ = np.clip(np.searchsorted(el, zlo) - 1, 0, NZ - 1); j_ = np.clip(np.searchsorted(eh, zhi) - 1, 0, NZ - 1)
    z = (z - TM[kk, i_, j_][:, None]) / TS[kk, i_, j_][:, None]
    la = rng.integers(0, 64, n); lb = (la + 1 + rng.integers(0, 63, n)) % 64
    qq = np.clip(np.round((m[:, None] + s[:, None] * z - b[:, None]) / sc[:, None]), 0, top[:, None]); qq[ar, la] = 0; qq[ar, lb] = top
    return qq * sc[:, None] + b[:, None]
score(gen0(A, np.random.default_rng(7)), 'fix0 (no correction)')
score(gen_corr(A, np.random.default_rng(7)), f'analytic one-shot correction, table {NB}x{NZ}x{NZ}')
