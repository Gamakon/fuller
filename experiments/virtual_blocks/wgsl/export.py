"""Fit the nibble generator's tables on the EVEN real blocks (half A) and write the GPU inputs:
tables.f32 = [P15, edges[0..NB+1], T[NB][513]] and params_A.f32 = per block (m_mean, m_std, scale, bias)."""
import sys; sys.argv = ['v5', '16', '4', 'diag']
exec(open('v5.py').read())
tab = np.concatenate([[P15], edges.clip(max=1e30), T.ravel()]).astype(np.float32)
tab.tofile('tables.f32')
np.column_stack([mean[A], std[A], S[A], B[A]]).astype(np.float32).tofile('params_A.f32')
print('NB', NB, 'tables', tab.size, 'blocks', len(A))
