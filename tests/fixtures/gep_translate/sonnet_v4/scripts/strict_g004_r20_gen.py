import random
from collections import deque
AR={}
def reg(names,a):
    for n in names.split(): AR[n]=a
reg("kernel.module_2 grid.with_wg wg.decls_cons wg.decl_f wg.decl_u loop.tile_L0 block.seq2_L2 index.or index.mul stmt.if_L0 compare.lt index.add index.and loop.tilex_C1 loop.tile_C1 loop.tile_C0 loop.tilex_C0 stmt.if_C1 stmt.if_C2 stmt.if_O0 block.seq2_C1 reg.set_f_C1 reg.set_i_C1 reg.set_f_C2 reg.set_i_C2 reg.set_v_C2 arith.add arith.mul geom.dot compare.gt compare.ne logic.and parr.get_f parr.get_i wg.get_safe wg.get_u_safe store.cands_C0 store.cands_O0",2)
reg("kernel.entry_safe parr.set_f_L2 parr.set_i_L2 parr.set_f_C1 parr.set_i_C1 parr.set_f_C2 parr.set_i_C2 wg.set_L2 wg.set_u_L2 loop.tilex_strided_C0 block.seq3_C1 block.seq3_C2 block.seq3_C0",3)
AR["block.seq3_C1"]=3;AR["block.seq3_C2"]=3;AR["block.seq3_C0"]=3
reg("block.seq4_L0 block.seq4_C1 block.seq4_C2 block.seq4_O0 grid.of",4)
reg("wg.decls_one block.one_L0 block.one_C0 block.one_C1 block.one_C2 block.one_O0 safe.pidx64 safe.tile_index safe.tile_index2 safe.wg_of index.of_col axis.x_wid axis.x_lid index.of_count bound.x_of_uniform uniform.of_lit stmt.block_C1 stmt.block_C2 stmt.block_C0 reg.get_f reg.get_i reg.get_v arith.inverse_sqrt load.keys load.queries",1)
reg("stmt.return_L0 stmt.barrier_L0 fallback wg.a wg.ua parr.a parr.ua safe.tile_counter safe.tile_counter2 builtin.workgroup_id builtin.local_invocation_id load.cfg_n_queries load.cfg_arrived_end uniform.cfg_arrived_end reg.f0 reg.f1 reg.f2 reg.f3 reg.f4 reg.f5 reg.f6 reg.f7 reg.i1 reg.i2 reg.i3 reg.i4 reg.i5 reg.i6 reg.i7 reg.v0",0)
def N(s,*c):
    if s.startswith("literal."): assert not c; return (s,[])
    assert AR[s]==len(c),(s,len(c)); return (s,list(c))
def L(n): return N("literal.index=%d"%n)
def pidx(x): return N("safe.pidx64",x)
col=lambda: N("index.of_col",N("axis.x_lid",N("builtin.local_invocation_id")))
widx=lambda: N("index.of_col",N("axis.x_wid",N("builtin.workgroup_id")))
tk=lambda: N("safe.tile_index",N("safe.tile_counter"))
tk2=lambda: N("safe.tile_index2",N("safe.tile_counter2"))
NEG=lambda: N("literal.real=-3e+38")
BIG=lambda: N("index.or",N("index.mul",L(65535),L(65536)),L(65535))
R=lambda n: N("reg."+n)
gf=lambda n: N("reg.get_f",R(n)); gi=lambda n: N("reg.get_i",R(n)); gv=lambda n: N("reg.get_v",R(n))
pa=lambda: N("parr.a"); pua=lambda: N("parr.ua")
def chain(st,r):
    if len(st)==1: return N("block.one_"+r,st[0])
    if len(st)<=4: return N("block.seq%d_%s"%(len(st),r),*st)
    return N("block.seq4_"+r,st[0],st[1],st[2],N("stmt.block_"+r,chain(st[3:],r)))
def setf(r,n,e): return N("reg.set_f_"+r,R(n),e)
def seti(r,n,e): return N("reg.set_i_"+r,R(n),e)
ingest=N("kernel.entry_safe",N("grid.of",N("literal.wgdim=1"),N("literal.wgdim=1"),N("literal.blk=1"),N("literal.blk=1")),N("block.one_L0",N("stmt.return_L0")),N("fallback"))
grid=N("grid.with_wg",N("grid.of",N("literal.wgdim=16"),N("literal.wgdim=1"),N("literal.blk=1"),N("literal.blk=1")),
  N("wg.decls_cons",N("wg.decl_f",N("wg.a"),N("literal.wglen=1024")),N("wg.decls_one",N("wg.decl_u",N("wg.ua"),N("literal.wglen=1024")))))
def reset(r,rr):
    return N("loop.tile_"+r,N("literal.lit64=64"),N("block.seq2_"+rr,
        N("parr.set_f_"+rr,pa(),pidx(tk()),NEG()),N("parr.set_i_"+rr,pua(),pidx(tk()),BIG())))
loopA=N("loop.tile_L0",N("literal.lit64=64"),N("block.seq2_L2",N("parr.set_f_L2",pa(),pidx(tk()),NEG()),N("parr.set_i_L2",pua(),pidx(tk()),BIG())))
# insertion into pa/pua by k2
ins=N("stmt.if_C2",N("compare.gt",gf("f3"),N("parr.get_f",pa(),pidx(tk2()))),
  chain([N("reg.set_f_C2",R("f4"),N("parr.get_f",pa(),pidx(tk2()))),
         N("reg.set_i_C2",R("i3"),N("parr.get_i",pua(),pidx(tk2()))),
         N("parr.set_f_C2",pa(),pidx(tk2()),gf("f3")),
         N("parr.set_i_C2",pua(),pidx(tk2()),gi("i1")),
         N("reg.set_f_C2",R("f3"),gf("f4")),
         N("reg.set_i_C2",R("i1"),gi("i3"))],"C2"))
kl=lambda: N("index.add",tk(),col())
dotloop=N("loop.tilex_C1",N("bound.x_of_uniform",N("uniform.of_lit",N("literal.lit4096=512"))),
  N("block.seq3_C2",
    N("reg.set_v_C2",R("v0"),N("load.keys",N("index.add",N("index.mul",N("index.and",kl(),L(524287)),L(512)),tk2()))),
    N("reg.set_f_C2",R("f0"),N("arith.add",gf("f0"),N("geom.dot",gv("v0"),N("load.queries",N("index.add",N("index.mul",widx(),L(512)),tk2()))))),
    N("reg.set_f_C2",R("f1"),N("arith.add",gf("f1"),N("geom.dot",gv("v0"),gv("v0"))))))
ifK=N("stmt.if_C1",N("compare.lt",kl(),N("index.of_count",N("load.cfg_arrived_end"))),
  chain([setf("C1","f0",N("literal.real=0")),setf("C1","f1",N("literal.real=0")),dotloop,
    setf("C1","f2",NEG()),
    N("stmt.if_C1",N("compare.gt",gf("f1"),N("literal.real=0")),N("block.one_C1",setf("C1","f2",N("arith.mul",gf("f0"),N("arith.inverse_sqrt",gf("f1")))))),
    N("stmt.if_C1",N("compare.gt",gf("f2"),N("parr.get_f",pa(),pidx(L(63)))),
      chain([setf("C1","f3",gf("f2")),seti("C1","i1",kl()),seti("C1","i2",L(0)),
        N("loop.tile_C1",N("literal.lit64=64"),N("block.one_C2",ins))],"C1"))],"C1"))
loopK=N("loop.tilex_strided_C0",N("bound.x_of_uniform",N("uniform.cfg_arrived_end")),N("literal.lit4096=16"),N("block.one_C1",ifK))
nq=lambda: N("compare.lt",widx(),N("index.of_count",N("load.cfg_n_queries")))
ifB=N("stmt.if_L0",nq(),N("block.one_C0",loopK))
wgi=lambda: N("safe.wg_of",N("index.add",N("index.mul",col(),L(64)),tk()))
loopC=N("loop.tile_L0",N("literal.lit64=64"),N("block.seq2_L2",
  N("wg.set_L2",N("wg.a"),wgi(),N("parr.get_f",pa(),pidx(tk()))),
  N("wg.set_u_L2",N("wg.ua"),wgi(),N("parr.get_i",pua(),pidx(tk())))))
body=N("block.seq4_L0",loopA,ifB,loopC,N("stmt.barrier_L0"))
def ins2(j,quirk=False):
    return N("stmt.if_C1",N("compare.gt",gf("f6"),N("parr.get_f",pa(),pidx(L(j)))),
      chain([N("reg.set_f_C1",R("f7"),N("parr.get_f",pa(),pidx(L(0 if quirk else j)))),
             N("reg.set_i_C1",R("i7"),N("parr.get_i",pua(),pidx(L(j)))),
             N("parr.set_f_C1",pa(),pidx(L(j)),gf("f6")),
             N("parr.set_i_C1",pua(),pidx(L(j)),gi("i5")),
             N("reg.set_f_C1",R("f6"),gf("f7")),
             N("reg.set_i_C1",R("i7"),gi("i7")) if False else N("reg.set_i_C1",R("i5"),gi("i7"))],"C1"))
sts=[setf("C1","f6",gf("f5")),seti("C1","i5",gi("i4")),seti("C1","i6",L(0))]+[ins2(j,j==28) for j in range(64)]
# fix: first statement of each ins is f7 = pa[j]; fine
mergeloop=N("loop.tilex_C0",N("bound.x_of_uniform",N("uniform.of_lit",N("literal.lit4096=1024"))),
  N("block.seq3_C1",
    N("reg.set_f_C1",R("f5"),N("wg.get_safe",N("wg.a"),N("safe.wg_of",tk()))),
    N("reg.set_i_C1",R("i4"),N("wg.get_u_safe",N("wg.ua"),N("safe.wg_of",tk()))),
    N("stmt.if_C1",N("logic.and",N("compare.ne",gi("i4"),BIG()),N("compare.gt",gf("f5"),N("parr.get_f",pa(),pidx(L(63))))),chain(sts,"C1"))))
def cst(off,last=False):
    idx=N("index.add",N("index.mul",widx(),L(64)),col()) if off==0 else N("index.add",N("index.add",N("index.mul",widx(),L(64)),L(off)),col())
    pi=col() if off==0 else N("index.add",L(off),col())
    return N("store.cands_O0" if last else "store.cands_C0",idx,N("parr.get_i",pua(),pidx(pi)))
epi=N("block.one_O0",N("stmt.if_O0",nq(),N("block.seq4_O0",
   N("stmt.block_C0",N("block.seq3_C0",reset("C0","C1"),mergeloop,cst(0))),cst(16),cst(32),cst(48,True))))
search=N("kernel.entry_safe",grid,body,epi)
root=N("kernel.module_2",ingest,search)
order=[];q=deque([root])
while q:
    n=q.popleft();order.append(n[0]);q.extend(n[1])
F=sum(1 for s in order if not s.startswith("literal.") and AR[s]>=1)
rng=random.Random(7)
def lit():
    k=rng.choice(["real","index","count","bits","int","flag","wgdim","blk","unroll","lit64","lit4096","wglen"])
    v={"real":lambda:"%.6g"%rng.uniform(-900,900),"index":lambda:rng.randint(0,900000),"count":lambda:rng.randint(0,900000),
       "bits":lambda:rng.randint(0,900000),"int":lambda:rng.randint(-900000,900000),"flag":lambda:rng.randint(0,1),
       "wgdim":lambda:rng.randint(1,1024),"blk":lambda:rng.choice([1,2,4,8]),"unroll":lambda:rng.choice([2,4,8]),
       "lit64":lambda:rng.randint(1,64),"lit4096":lambda:rng.randint(1,4096),"wglen":lambda:rng.randint(1,8192)}[k]()
    return "literal.%s=%s"%(k,v)
fixed=["builtin.global_invocation_id","builtin.local_invocation_id","builtin.workgroup_id","builtin.num_workgroups","builtin.item","builtin.tile","builtin.subgroup_size","builtin.subgroup_invocation_id",
 "load.cfg_seed_slot","load.cfg_stage","load.cfg_arrived_begin","load.cfg_arrived_end","load.cfg_n_queries","load.cfg_pass_index","load.cfg_n_passes","load.cfg_c_max","uniform.cfg_m" if False else "uniform.cfg_d",
 "stmt.return_L0","stmt.barrier_L0","safe.counter","safe.tile_counter","safe.tile_counter2","safe.tile_counter3","bound.m","bound.d","bound.rows","load.phi_len","load.h_len",
 "wg.b","wg.c","wg.d","wg.va","wg.vb","wg.v2a","wg.ub","wg.uc","wg.ud","wg.au_a","wg.ai_a","parr.b","parr.c","parr.d","parr.v","parr.i","parr.m","parr.n","parr.vq","parr.q","uniform.cfg_r_rows","uniform.cfg_r_off","uniform.nwg_x","uniform.subgroup_size"]
regs=["reg.f%d"%i for i in range(0,64)]+["reg.i%d"%i for i in range(0,64)]+["reg.v%d"%i for i in range(0,32)]+["reg.n%d"%i for i in range(16)]+["reg.b%d"%i for i in range(16)]+["reg.w%d"%i for i in range(8)]+["reg.u%d"%i for i in range(8)]
regs=[r for r in regs if r not in("reg.f32","reg.i32")]+["reg.f32","reg.i32"]
pad=[];prev=None;pk=[]
while len(pad)<3*F+1:
    r=rng.random()
    if r<0.3: s,k=lit(),"lit"
    elif r<0.65: s,k=rng.choice(regs),"reg"
    else: s,k=rng.choice(fixed),"fix"
    if s==prev or pk[-2:]==[k,k] : continue
    pad.append(s);prev=s;pk.append(k)
with open("/tmp/gengep/out.txt","w") as f:
    for i,s in enumerate(order): f.write("%d %s\n"%(i,s))
    f.write("ORF END\n")
    for j,s in enumerate(pad): f.write("%d %s\n"%(len(order)+j,s))
print(len(order),F,len(pad))
