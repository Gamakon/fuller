import random
random.seed(7)
def N(s,*c): return (s,list(c))
def L(f,v): return N(f'literal.{f}={v}')
def widx(): return N('index.of_col',N('axis.x_wid',N('builtin.workgroup_id')))
def lidx(): return N('index.of_col',N('axis.x_lid',N('builtin.local_invocation_id')))
def tk(): return N('safe.tile_index',N('safe.tile_counter'))
def tk2(): return N('safe.tile_index2',N('safe.tile_counter2'))
def ct(): return N('safe.counter_index',N('safe.counter'))
def pidx(x): return N('safe.pidx64',x)
def pa_get(x): return N('parr.get_f',N('parr.a'),pidx(x))
def pua_get(x): return N('parr.get_i',N('parr.ua'),pidx(x))
def rf(n): return N('reg.get_f',N(f'reg.f{n}'))
def ri(n): return N('reg.get_i',N(f'reg.i{n}'))
def setf(r,g,v): return N(f'reg.set_f_{g}',N(f'reg.f{r}'),v)
def seti(r,g,v): return N(f'reg.set_i_{g}',N(f'reg.i{r}'),v)
def add(a,b): return N('index.add',a,b)
def mul(a,b): return N('index.mul',a,b)
def fadd(a,b): return N('arith.add',a,b)
def lb(n): return N('bound.x_of_uniform',N('uniform.of_lit',L('lit4096',n)))
def lbu(n): return N('bound.of_uniform',N('uniform.of_lit',L('lit4096',n)))
def PADV(): return N('index.or',N('convert.bits_to_index',N('bits.shl',L('bits',65535),L('count',16))),L('index',65535))
def blk(g,st):
    if len(st)<=4:
        return N('block.one_'+g if len(st)==1 else f'block.seq{len(st)}_{g}',*st)
    return N(f'block.seq4_{g}',*st[:3],N('stmt.block_'+g,blk(g,st[3:])))
def QW(k): return N('load.queries',add(mul(widx(),L('index',512)),k))
def dot(a,b): return N('geom.dot',a,b)
NQ=lambda: N('compare.lt',widx(),N('index.of_count',N('load.cfg_n_queries')))
def swap_body(fc,ic,ft,it):
    cond=N('compare.gt',rf(fc),pa_get(tk2()))
    st=[setf(ft,'C2',pa_get(tk2())),seti(it,'C2',pua_get(tk2())),
        N('parr.set_f_C2',N('parr.a'),pidx(tk2()),rf(fc)),
        N('parr.set_i_C2',N('parr.ua'),pidx(tk2()),ri(ic)),
        setf(fc,'C2',rf(ft)),seti(ic,'C2',ri(it))]
    return N('block.one_C2',N('stmt.if_C2',cond,blk('C2',st)))
def ins_loop(*a): return N('loop.tilex_C1',lb(64),swap_body(*a))
# A
A=N('loop.items_boundx',lbu(64),blk('L1',[N('parr.set_f_L1',N('parr.a'),pidx(ct()),L('real','-3e+38')),N('parr.set_i_L1',N('parr.ua'),pidx(ct()),PADV())]))
# B
loop1=N('loop.tilex_C0',lb(512),blk('C1',[setf(0,'C1',fadd(rf(0),dot(QW(tk()),QW(tk()))))]))
s4=N('loop.tilex_C1',lb(512),blk('C2',[
  N('reg.set_v_C2',N('reg.v0'),N('load.keys',add(mul(N('index.and',add(tk(),lidx()),L('index',524287)),L('index',512)),tk2()))),
  setf(1,'C2',fadd(rf(1),dot(N('reg.get_v',N('reg.v0')),QW(tk2())))),
  setf(2,'C2',fadd(rf(2),dot(N('reg.get_v',N('reg.v0')),N('reg.get_v',N('reg.v0')))))]))
sq=lambda: N('arith.sqrt',N('arith.mul',rf(0),rf(2)))
s6=N('stmt.if_C1',N('compare.gt',sq(),L('real',0)),blk('C1',[setf(3,'C1',N('arith.div',rf(1),sq()))]))
s7=N('stmt.if_C1',N('compare.gt',rf(3),pa_get(L('index',63))),blk('C1',[
  setf(4,'C1',rf(3)),seti(1,'C1',add(tk(),lidx())),seti(2,'C1',L('index',0)),ins_loop(4,1,5,3)]))
body2=[N('reg.set_f_C1',N('reg.f1'),L('real',0)),N('reg.set_f_C1',N('reg.f2'),L('real',0)),seti(0,'C1',L('index',0)),s4,
       setf(3,'C1',L('real','-3e+38')),s6,s7]
cond2=N('compare.lt',add(tk(),lidx()),N('index.of_count',N('load.cfg_arrived_end')))
loop2=N('loop.tilex_strided_C0',N('bound.x_of_uniform',N('uniform.cfg_arrived_end')),L('lit4096',16),blk('C1',[N('stmt.if_C1',cond2,blk('C1',body2))]))
B=N('stmt.if_L0',NQ(),blk('C0',[loop1,loop2]))
# C
wi=lambda: N('safe.wg_of',add(mul(lidx(),L('index',64)),ct()))
C=N('loop.items_boundx',lbu(64),blk('L1',[N('wg.set_L1',N('wg.a'),wi(),pa_get(ct())),N('wg.set_u_L1',N('wg.ua'),wi(),pua_get(ct()))]))
D=N('stmt.barrier_L0')
body=blk('L0',[A,B,C,D])
# E
c1=N('loop.tilex_C0',lb(64),blk('C1',[N('parr.set_f_C1',N('parr.a'),pidx(tk()),L('real','-3e+38')),N('parr.set_i_C1',N('parr.ua'),pidx(tk()),PADV())]))
wgi=lambda: N('safe.wg_of',tk())
m_if=N('stmt.if_C1',N('logic.and',N('compare.ne',ri(4),PADV()),N('compare.gt',rf(6),pa_get(L('index',63)))),
  blk('C1',[setf(7,'C1',rf(6)),seti(5,'C1',ri(4)),seti(6,'C1',L('index',0)),ins_loop(7,5,8,7)]))
c2=N('loop.tilex_C0',lb(1024),blk('C1',[
  setf(6,'C1',N('wg.get_safe',N('wg.a'),wgi())),
  seti(4,'C1',N('wg.get_u_safe',N('wg.ua'),wgi())),m_if]))
def store(g,off,idxv):
    return N(f'store.cands_{g}',idxv,pua_get(idxv2(off)))
def base(): return mul(widx(),L('index',64))
stores=[]
def mk(g,off):
    if off==0:
        i=add(base(),lidx()); v=lidx()
    else:
        i=add(add(base(),L('index',off)),lidx()); v=add(L('index',off),lidx())
    return N(f'store.cands_{g}',i,pua_get(v))
c3,c4,c5=mk('C0',0),mk('C0',16),mk('C0',32)
o=mk('O0',48)
epi=N('block.one_O0',N('stmt.if_O0',NQ(),N('block.seq4_O0',N('stmt.block_C0',N('block.seq3_C0',c1,c2,c3)),c4,c5,o)))
# entries
ing=N('kernel.entry_safe',N('grid.of',L('wgdim',1),L('wgdim',1),L('blk',1),L('blk',1)),N('block.one_L0',N('stmt.return_L0')),N('fallback'))
grid=N('grid.with_wg',N('grid.of',L('wgdim',16),L('wgdim',1),L('blk',1),L('blk',1)),
  N('wg.decls_cons',N('wg.decl_f',N('wg.a'),L('wglen',1024)),N('wg.decls_one',N('wg.decl_u',N('wg.ua'),L('wglen',1024)))))
sea=N('kernel.entry_safe',grid,body,epi)
root=N('kernel.module_2',ing,sea)
q=[root];orf=[]
while q:
    s,c=q.pop(0);orf.append(s);q.extend(c)
F=sum(1 for s in orf if not (s.startswith(('literal.','reg.f','reg.i','reg.v','builtin.','safe.tile_counter','safe.counter','stmt.return','stmt.barrier','fallback','parr.a','parr.ua','wg.a','wg.ua','load.cfg','uniform.cfg','reg.get_')) ) or s.startswith('reg.get_'))
# arity-based F: recompute precisely
import re
F=0
def walk(n):
    global F
    if n[1]: F+=1
    for x in n[1]: walk(x)
walk(root)
pool=['literal.real=%s'%v for v in ['0.5','-12.25','873.15','2','-0.125','31.5','1e+10']]+\
 ['literal.index=%d'%v for v in [3,512,77,4096,1000]]+['literal.count=%d'%v for v in [5,64,300]]+\
 ['literal.bits=%d'%v for v in [255,65535,1024]]+['literal.wgdim=%d'%v for v in [4,32,99,256]]+\
 ['literal.blk=%d'%v for v in [1,2,4,8]]+['literal.lit64=%d'%v for v in [7,20,46]]+['literal.lit4096=%d'%v for v in [128,1700,2958]]+\
 ['literal.wglen=%d'%v for v in [292,3939]]+['literal.flag=0','literal.flag=1','literal.int=-308496','literal.int=42','literal.unroll=4','literal.unroll=2']+\
 ['reg.f%d'%i for i in (3,9,13,40)]+['reg.i%d'%i for i in (2,5,11,25)]+['reg.v%d'%i for i in (1,4,9)]+['reg.b%d'%i for i in (0,3)]+['reg.n%d'%i for i in (1,7)]+\
 ['builtin.global_invocation_id','builtin.local_invocation_id','builtin.workgroup_id','builtin.num_workgroups','builtin.item','builtin.tile','builtin.local_invocation_index','builtin.subgroup_size',
  'load.cfg_seed_slot','load.cfg_stage','load.cfg_arrived_begin','load.cfg_pass_index','load.cfg_c_max','uniform.cfg_m','uniform.cfg_d','uniform.cfg_stage' if False else 'uniform.cfg_r_rows',
  'safe.counter','safe.tile_counter','safe.tile_counter2','stmt.return_L0','stmt.barrier_L0','parr.b','parr.ua','wg.b','wg.ua','bound.m','bound.d','uniform.subgroup_size','load.phi_len']
pad=[];prev=None
for _ in range(3*F+1):
    while True:
        s=random.choice(pool)
        if s!=prev and not(pad and s.split('=')[0]==pad[-1].split('=')[0] and len(pad)>1 and pad[-2].split('=')[0]==s.split('=')[0]): break
    pad.append(s);prev=s
out=[]
for i,s in enumerate(orf): out.append(f'{i} {s}')
out.append('ORF END')
for j,s in enumerate(pad): out.append(f'{len(orf)+j} {s}')
open('/tmp/gepgen/out.txt','w').write('\n'.join(out))
print(len(orf),F,len(pad))
