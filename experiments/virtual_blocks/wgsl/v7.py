# Feedback-free by STRATIFIED draws: lane l takes stratum pi(l) = (a*l + c) mod 64 (a odd, a and c hashed per block),
# u = (pi(l) + jitter)/64 inside the truncated range, so the block's 64 values follow the table's quantiles and its
# mean and std land without any cross-lane correction. Then quantise and pin as v5.
import sys, os; sys.argv = ['v5', '16', '0', 'diag']
exec(open('v5.py').read())
FIX = int(os.environ.get('FIX', '0')); JIT = float(os.environ.get('JIT', '1.0')); ROUNDS = int(os.environ.get('ROUNDS', '3'))
def gen_strat(idx):
    m, s, sc, b = mean[idx], std[idx], S[idx], B[idx]; kk = binof(span[idx]); n = len(idx); ar = np.arange(n)
    top = np.where(rng.random(n) < P15, 15, 14)
    za = (b - 0.5 * sc - m) / s; zb = (b + (top + 0.5) * sc - m) / s
    zlo, zhi = np.minimum(za, zb), np.maximum(za, zb)
    st = np.broadcast_to(np.arange(64), (n, 64)).copy()
    for rnd in range(ROUNDS):     # a keyed bijection on 6 bits: xor key, odd multiply, xorshift
        k1 = rng.integers(0, 64, n)[:, None]; a = (2 * rng.integers(0, 32, n) + 1)[:, None]
        st = ((st ^ k1) * a) & 63
        st = st ^ (st >> 3)
    u = (st + 0.5 + JIT * (rng.random((n, 64)) - 0.5)) / 64
    z = np.empty_like(u)
    for kq in range(NB):
        r = kk == kq
        Flo = np.interp(zlo[r], T[kq], G_); Fhi = np.interp(zhi[r], T[kq], G_)
        z[r] = np.interp(Flo[:, None] + u[r] * (Fhi - Flo)[:, None], G_, T[kq])
    la = rng.integers(0, 64, n); lb = (la + 1 + rng.integers(0, 63, n)) % 64
    for it in range(FIX + 1):
        qq = np.clip(np.round((m[:, None] + s[:, None] * z - b[:, None]) / sc[:, None]), 0, top[:, None]); qq[ar, la] = 0; qq[ar, lb] = top
        y = qq * sc[:, None] + b[:, None]
        if it < FIX:
            z = (z - ((y.mean(1) - m) / s)[:, None]) * (s / y.std(1))[:, None]
    return y
score(gen_strat(A), f'stratified (6-bit bijection, {ROUNDS} rounds), FIX={FIX}, jitter={JIT}')
