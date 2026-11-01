import sys, numpy as np
sys.path.insert(0, '/Users/andrewmorgan/Dev/gamakon/faxl-core/docs/virtual-blocks')
from fitness import objective
d = np.load('/Users/andrewmorgan/Dev/gamakon/faxl-core/docs/virtual-blocks/fitness_data.npz')
R = d['real_blocks'].astype(np.float64); S = d['real_scale'].astype(np.float64); B = d['real_bias'].astype(np.float64)
n = len(R); A = np.arange(0, n, 2); Bh = np.arange(1, n, 2)       # fit and generate on A's params, compare with B
mean = R.mean(1); std = R.std(1)
q = np.clip(np.round((R - B[:, None]) / S[:, None]), 0, 15)
def stats_(X, s, b):
    qq = np.clip(np.round((X - b[:, None]) / s[:, None]), 0, 15)
    Z = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    k = ((Z**4).mean(1) - 3)
    return dict(rails=((qq == 0) | (qq == 15)).sum(1).mean(), kurt50=np.median(k), kurt99=np.percentile(k, 99))
def score(G, tag, idx=A):
    o, det = objective(R[Bh], G)
    print(f"{tag}: objective {o:.4f} {det} {stats_(G, S[idx], B[idx])}", flush=True)
rng = np.random.default_rng(7)
