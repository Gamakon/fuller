exec(open('v4.py').read().replace("score(gen(A)", "G=(gen(A)) #"))
from fitness import feats
from scipy.stats import ks_2samp
fr, fg = feats(R[Bh]), feats(G)
names = ['norm','mean','std','skew','kurt','maxabs','o0','o15','o31','o47','o63','nearzero']
for i, nm in enumerate(names): print(nm, round(ks_2samp(fr[:, i], fg[:, i]).statistic, 4))
qg = np.clip(np.round((G - B[A][:, None]) / S[A][:, None]), 0, 15)
print('n_maxlevel real', np.bincount((q[Bh]==q[Bh].max(1,keepdims=True)).sum(1))[:8]/len(Bh))
print('n_maxlevel gen ', np.bincount((qg==qg.max(1,keepdims=True)).sum(1))[:8]/len(A))
print('max level real', np.bincount(q[Bh].max(1).astype(int),minlength=16)[12:]/len(Bh), 'gen', np.bincount(qg.max(1).astype(int),minlength=16)[12:]/len(A))
print('n0 real', np.bincount((q[Bh]==0).sum(1))[:5]/len(Bh), 'gen', np.bincount((qg==0).sum(1))[:5]/len(A))
