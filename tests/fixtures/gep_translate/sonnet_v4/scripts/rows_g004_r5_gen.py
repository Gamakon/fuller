import random
def n(name,*c): return (name,list(c))
wid=n('index.of_col',n('axis.x_wid',n('builtin.workgroup_id')))
lid=n('index.of_col',n('axis.x_lid',n('builtin.local_invocation_id')))
li=lambda v:n(f'literal.index={v}')
tile=lambda:n('safe.tile_index',n('safe.tile_counter'))
tile2=lambda:n('safe.tile_index2',n('safe.tile_counter2'))
cnt=lambda:n('safe.counter_index',n('safe.counter'))
pidx=lambda x:n('safe.pidx64',x)
pag=lambda i:n('parr.get_f',n('parr.a'),pidx(i))
pug=lambda i:n('parr.get_i',n('parr.ua'),pidx(i))
CONST=lambda:n('index.or',n('convert.bits_to_index',n('bits.shl',n('literal.bits=65535'),n('literal.count=16'))),n('subgroup.inclusive_mul',li(65535)))
bx=lambda v:n('bound.x_of_uniform',n('uniform.of_lit',n(f'literal.lit4096={v}')))
bu=lambda v:n('bound.of_uniform',n('uniform.of_lit',n(f'literal.lit4096={v}')))
gf=lambda r:n('reg.get_f',n('reg.'+r)); gi=lambda r:n('reg.get_i',n('reg.'+r)); gv=lambda r:n('reg.get_v',n('reg.'+r))
sf=lambda R,r,v:n(f'reg.set_f_{R}',n('reg.'+r),v)
si=lambda R,r,v:n(f'reg.set_i_{R}',n('reg.'+r),v)
NEG=lambda:n('literal.real=-3e+38')
kl=lambda:index_add(tile(),lid)
def index_add(a,b): return n('index.add',a,b() if callable(b) else b)
def qidx(k): return n('index.add',n('index.mul',wid,li(512)),k)
def dot(a,b): return n('geom.dot',a,b)
def insert(f,i,ft,it,fi_first):
    ck=lambda:tile2()
    s=[sf('C2',ft,pag(ck())),si('C2',it,pug(ck())),
       n('parr.set_f_C2',n('parr.a'),pidx(ck()),gf(f)),
       n('parr.set_i_C2',n('parr.ua'),pidx(ck()),gi(i))]
    a=si('C2',i,gi(it)); b=sf('C2',f,gf(ft))
    s+= [a,b] if fi_first else [b,a]
    cond=n('compare.gt',gf(f),pag(ck()))
    return n('loop.tilex_C1',bx(64),n('block.one_C2',n('stmt.if_C2',cond,n('block.seq4_C2',s[0],s[1],s[2],n('stmt.block_C2',n('block.seq3_C2',s[3],s[4],s[5]))))))
# A
A=n('loop.items_boundx',bu(64),n('block.seq2_L1',
  n('parr.set_f_L1',n('parr.a'),pidx(cnt()),NEG()),
  n('parr.set_i_L1',n('parr.ua'),pidx(cnt()),CONST())))
# B
B1=n('loop.tilex_C0',bx(512),n('block.one_C1',sf('C1','f0',n('arith.add',gf('f0'),dot(n('load.queries',qidx(tile())),n('load.queries',qidx(tile())))))))
inner=n('loop.tilex_C1',bx(512),n('block.seq3_C2',
  n('reg.set_v_C2',n('reg.v0'),n('load.keys',n('index.add',n('index.mul',n('index.and',n('index.add',tile(),lid),li(524287)),li(512)),tile2()))),
  sf('C2','f1',n('arith.add',gf('f1'),dot(gv('v0'),n('load.queries',qidx(tile2()))))),
  sf('C2','f2',n('arith.add',gf('f2'),dot(gv('v0'),gv('v0'))))))
sq=lambda:n('arith.sqrt',n('arith.mul',gf('f0'),gf('f2')))
s5=n('stmt.if_C1',n('compare.gt',sq(),n('literal.real=0')),n('block.one_C1',sf('C1','f3',n('arith.div',gf('f1'),sq()))))
s6=n('stmt.if_C1',n('compare.gt',gf('f3'),pag(li(63))),n('block.seq3_C1',
   sf('C1','f4',gf('f3')),si('C1','i1',n('index.add',tile(),lid)),insert('f4','i1','f5','i3',True)))
ifb=n('stmt.if_C1',n('compare.lt',n('index.add',tile(),lid),n('index.of_count',n('load.cfg_arrived_end'))),
  n('block.seq4_C1',sf('C1','f1',n('literal.real=0')),sf('C1','f2',n('literal.real=0')),inner,
    n('stmt.block_C1',n('block.seq3_C1',sf('C1','f3',NEG()),s5,s6))))
B2=n('loop.tilex_strided_C0',n('bound.x_of_uniform',n('uniform.cfg_arrived_end')),n('literal.lit4096=16'),n('block.one_C1',ifb))
nq=lambda:n('compare.lt',wid,n('index.of_count',n('load.cfg_n_queries')))
B=n('stmt.if_L0',nq(),n('block.seq2_C0',B1,B2))
# C
wgi=lambda:n('safe.wg_of',n('index.add',n('index.mul',lid,li(64)),cnt()))
C=n('loop.items_boundx',bu(64),n('block.seq2_L1',
  n('wg.set_L1',n('wg.a'),wgi(),pag(cnt())),
  n('wg.set_u_L1',n('wg.ua'),wgi(),pug(cnt()))))
D=n('stmt.barrier_L0')
# epilogue
c1=n('loop.tilex_C0',bx(64),n('block.seq2_C1',
  n('parr.set_f_C1',n('parr.a'),pidx(tile()),NEG()),
  n('parr.set_i_C1',n('parr.ua'),pidx(tile()),CONST())))
def merge():
    ins=n('stmt.if_C1',n('logic.and',n('compare.ne',gi('i4'),CONST()),n('compare.gt',gf('f6'),pag(li(63)))),
      n('block.seq3_C1',sf('C1','f7',gf('f6')),si('C1','i5',gi('i4')),insert('f7','i5','f8','i7',False)))
    return n('loop.tilex_C0',bx(1024),n('block.seq3_C1',
      sf('C1','f6',n('wg.get_safe',n('wg.a'),n('safe.wg_of',tile()))),
      si('C1','i4',n('wg.get_u_safe',n('wg.ua'),n('safe.wg_of',tile()))),ins))
def st(R,off):
    base=n('index.mul',wid,li(64))
    idx=n('index.add',n('index.add',base,li(off)),lid) if off else n('index.add',base,lid)
    v=pug(n('index.add',li(off),lid) if off else lid)
    return n(f'store.cands_{R}',idx,v)
c4,c5,c6,o=st('C0',0),st('C0',16),st('C0',32),st('O0',48)
epi=n('block.one_O0',n('stmt.if_O0',nq(),n('block.seq4_O0',n('stmt.block_C0',n('block.seq4_C0',c1,merge(),merge(),c4)),c5,c6,o)))
grid=n('grid.with_wg',n('grid.of',n('literal.wgdim=16'),n('literal.wgdim=1'),n('literal.blk=1'),n('literal.blk=1')),
  n('wg.decls_cons',n('wg.decl_f',n('wg.a'),n('literal.wglen=1024')),n('wg.decls_one',n('wg.decl_u',n('wg.ua'),n('literal.wglen=1024')))))
search=n('kernel.entry_safe',grid,n('block.seq4_L0',A,B,C,D),epi)
ing=n('kernel.entry_safe',n('grid.of',n('literal.wgdim=1'),n('literal.wgdim=1'),n('literal.blk=1'),n('literal.blk=1')),n('block.one_L0',n('stmt.return_L0')),n('fallback'))
root=n('kernel.module_2',ing,search)
out=[];q=[root]
while q:
    nd=q.pop(0);out.append(nd[0]);q+=nd[1]
F=sum(1 for _ in [0])
# arity>=1 count
def cnt_f(nd): return (1 if nd[1] else 0)+sum(cnt_f(c) for c in nd[1])
F=cnt_f(root)
random.seed(7)
pool=['reg.f%d'%i for i in range(8,40)]+['reg.i%d'%i for i in range(8,60)]+['reg.v%d'%i for i in range(2,30)]+['reg.n%d'%i for i in range(16)]+['reg.b%d'%i for i in range(1,12)]+['builtin.global_invocation_id','builtin.local_invocation_id','builtin.workgroup_id','builtin.local_invocation_index','builtin.subgroup_size','stmt.return_L0','stmt.barrier_L0','load.cfg_stage','load.cfg_seed_slot','load.cfg_c_max','load.cfg_pass_index','load.cfg_n_passes','load.cfg_arrived_begin','safe.tile_counter2','safe.counter','wg.b','wg.va','parr.b','parr.d','parr.x','uniform.cfg_m','uniform.cfg_d','uniform.nwg_x']
gen=[lambda:'literal.real=%g'%random.uniform(-900,900),lambda:'literal.index=%d'%random.randint(0,900000),lambda:'literal.count=%d'%random.randint(0,900000),lambda:'literal.bits=%d'%random.randint(0,900000),lambda:'literal.wgdim=%d'%random.randint(1,1000),lambda:'literal.blk=%d'%random.choice([1,2,4,8]),lambda:'literal.lit64=%d'%random.randint(1,64),lambda:'literal.lit4096=%d'%random.randint(1,4096),lambda:'literal.flag=%d'%random.randint(0,1),lambda:'literal.int=%d'%random.randint(-900000,900000),lambda:'literal.unroll=%d'%random.choice([2,4,8]),lambda:'literal.wglen=%d'%random.randint(1,8000)]
pad=[]
while len(pad)<3*F+1:
    s=random.choice(pool) if random.random()<.55 else random.choice(gen)()
    if pad and pad[-1]==s: continue
    pad.append(s)
i=0
for s in out: print(i,s);i+=1
print('ORF END')
for s in pad: print(i,s);i+=1
import sys;print('ORF',len(out),'F',F,file=sys.stderr)
