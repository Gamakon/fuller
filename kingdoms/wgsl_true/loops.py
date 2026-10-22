"""Batch 3: fuel strata, loops and counters, uniform provenance, the barrier."""
from lib import *

BOUND_CLASSES = (6, 12, 18, 24)
UCMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}
UOPS = {"add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})", "div": "({0} / {1})", "mod": "({0} % {1})", "min": "min({0}, {1})",
        "max": "max({0}, {1})", "shl": "({0} << {1})", "shr": "({0} >> {1})", "and": "({0} & {1})", "or": "({0} | {1})"}


def declare(iface):
    for c in BOUND_CLASSES:
        T(f"u32.boundx{c}", f"an expression loop bound capped at 2^{c} (trip class {c // 6}); any invocation's value")
        T(f"u32.boundu{c}", f"a loop bound capped at 2^{c} that is the same in every invocation of the workgroup")
        register_twin(f"u32.boundx{c}")
        SAFE_ARGS.add(f"u32.boundu{c}")
    for d in range(1, 6):
        T(f"u32.tcounter{d}", f"the counter k{d} of a loop whose bound is a literal or uniform: uniform")
        T(f"u32.xcounter{d}", f"the counter kx{d} of a loop whose bound is any expression, or that breaks: divergent")
        SAFE_ARGS.add(f"u32.tcounter{d}")
        SAFE_ARGS.add(f"u32.xcounter{d}")
        for p in ("n", "k", "nx", "kx", "kc", "kxc"):
            decl(f"var {p}{d}: u32 = 0u;")
    T("ureg<u32>.uniform", "a named u32 register U0..U7 that holds the same value in every invocation")
    SAFE_ARGS.add("ureg<u32>.uniform")
    for n in range(8):
        decl(f"var U{n}: u32 = 0u;")


def loop_text(d1, divergent, bound="{0}", stride=None, body="{1}"):
    """A counted loop. naga writes a `for` with a continuing step as `while(true) { if (!loop_init) { k = k + 1 } ... }`, and
    Apple's compiler hangs the dispatch on that form (docs/reports/2026-10-21_metal_hang_k2), so the loop is a plain `loop`
    with no continuing block. The step is taken on a shadow counter `kc` before the body: `k = kc` at the top, the test
    `k >= n` before the body, `kc = k + step` next, so `k` is the trip inside the body, the final value after the loop (n
    after the last trip, the trip of a `break`), and a `continue` still takes the step. Measured 2026-10-22 against the
    earlier spelling, where the counter started at -1 and the step came before the test (`k = k + 1u; if (k >= n)`):
    volumepassage at 64 x 64 one submit 0.957x of its time (the test of a counter from 0 before the step lets Apple's
    compiler see the trip count); it terminates on the hang reproducer."""
    n, k, kc = (f"nx{d1}", f"kx{d1}", f"kxc{d1}") if divergent else (f"n{d1}", f"k{d1}", f"kc{d1}")
    head = f"{kc} = 0u; loop {{ {k} = {kc}; if ({k} >= {n}) {{ break; }} "
    if stride is None:
        return f"{n} = {bound}; {head}{kc} = {k} + 1u; {body} }}"
    return f"{{ let st = {stride}; {n} = {bound}; {head}{kc} = {k} + st; {body} }} }}"


def build(iface):
    # ---- bounds
    for c in BOUND_CLASSES:
        cap = 2 ** c
        R(f"bound.of_index_{c}", f"u32.boundx{c}", ["u32.index"], f"min({{0}}, {cap}u)", "the cap is the row's text, so every expression-bounded loop has a known trip bound", twin=True)
        R(f"bound.of_uniform_{c}", f"u32.boundu{c}", ["u32.uniform"], f"min({{0}}, {cap}u)", "a uniform bound: the trip count is the same in every invocation")
        R(f"own.u32_boundx{c}", f"u32.boundx{c}", [], "1u", "a bound of one trip")
        R(f"own.u32_boundx{c}w", f"u32.boundx{c}w", [], "1u")
        R(f"own.u32_boundu{c}", f"u32.boundu{c}", [], "1u")
    # ---- counters
    for d in range(1, 6):
        R(f"safe.tile_counter{d}", f"u32.tcounter{d}", [], f"k{d}", "0u before its loop, the trip inside, the final value after")
        R(f"safe.xtile_counter{d}", f"u32.xcounter{d}", [], f"kx{d}")
        R(f"index.of_tcounter{d}", "u32.index", [f"u32.tcounter{d}"], "{0}", twin=True)
        R(f"index.of_xcounter{d}", "u32.index", [f"u32.xcounter{d}"], "{0}", twin=True)
        R(f"uniform.of_tcounter{d}", "u32.uniform", [f"u32.tcounter{d}"], "{0}", "the counter of a literal- or uniform-bounded loop is the same in every invocation")
    # ---- uniform provenance
    for f in iface["fields"]:
        R(f"uniform.{f}", "u32.uniform", [], f"cfg.{f}", "a field is the same in every invocation")
    R("uniform.lit", "u32.uniform", ["u32.lit4096"], "lit_u32({0})", "the runtime literal: a u32.uniform is never a const-expression (uniform arithmetic of literals could overflow at shader creation)")
    for c in "xyz":
        R(f"uniform.wid_{c}", "u32.uniform", ["vec3<u32>.wid"], "{0}." + c, "the workgroup id is uniform")
        R(f"uniform.nwg_{c}", "u32.uniform", ["vec3<u32>.nwg"], "{0}." + c)
    for op, sp in UOPS.items():
        R(f"uniform.{op}", "u32.uniform", ["u32.uniform", "u32.uniform"], sp, "arithmetic of uniforms is uniform")
    R("own.u32_uniform", "u32.uniform", [], "lit_u32(1u)", "a literal is uniform")
    R("index.of_uniform", "u32.index", ["u32.uniform"], "{0}", "one way only", twin=True)
    for c, op in UCMP.items():
        R(f"uflag.{c}", "bool.uflag", ["u32.uniform", "u32.uniform"], "({0} " + op + " {1})")
    R("uflag.and", "bool.uflag", ["bool.uflag", "bool.uflag"], "({0} && {1})")
    R("uflag.or", "bool.uflag", ["bool.uflag", "bool.uflag"], "({0} || {1})")
    R("uflag.not", "bool.uflag", ["bool.uflag"], "(!{0})")
    R("own.bool_uflag", "bool.uflag", [], "lit_bool(true)")
    R("flag.of_uflag", "bool.flag", ["bool.uflag"], "{0}", "one way only: a uniform predicate is a predicate", twin=True)
    for n in range(8):
        R(f"ureg.U{n}", "ureg<u32>.uniform", [], f"U{n}")
    R("uniform.get_U", "u32.uniform", ["ureg<u32>.uniform"], "{0}", "a uniform register's only writers are L rungs from uniform sources")
    for x, d, k in rungs("L"):
        s = S(x, d, k)
        sfx = f"_L{d}_f{k}"
        R("ureg.set" + sfx, s, ["ureg<u32>.uniform", "u32.uniform"], "{0} = {1};", "L rungs only, from a uniform value only")
        R("stmt.if_uflag" + sfx, s, ["bool.uflag", B("L", d, k)], "if ({0}) { {1} }", "the body keeps the L rung: every invocation takes the same branch")
        R("stmt.if_else_uflag" + sfx, s, ["bool.uflag", B("L", d, k), B("L", d, k)], "if ({0}) { {1} } else { {2} }")
        R("stmt.barrier" + sfx, s, [], "workgroupBarrier();", "at L rungs only: every enclosing condition and bound is uniform by the types")
        R("stmt.storage_barrier" + sfx, s, [], "storageBarrier();", "a control barrier affecting the storage address space; L rungs only")
        if iface["has_subgroup_barrier"]:
            R("stmt.subgroup_barrier" + sfx, s, [], "subgroupBarrier();", "present only with the subgroup-barrier feature")
    # ---- loops
    for d in range(0, 5):
        d1 = d + 1
        for k in STRATA:
            # lit64 bound
            for x in "LCOW":
                if d1 not in DEPTH[x] or k + 1 > K:
                    continue
                div = x != "L"
                R(f"loop.tile_{x}{d}_f{k}", S(x, d, k + 1), ["u32.lit64", B(x, d1, k)], loop_text(d1, div), "a literal bound raises the stratum by one")
                if x != "W" or DEPTH["S"]:
                    R(f"loop.tile_strided_{x}{d}_f{k}", S(x, d, k + 1), ["u32.lit64", "u32.lit4096", B(x, d1, k)], loop_text(d1, div, stride="{1}", body="{2}"), "the stride form")
            for c in BOUND_CLASSES:
                cls = c // 6
                if k + cls > K:
                    continue
                if d1 in DEPTH["L"]:
                    R(f"loop.tileu_{c}_L{d}_f{k}", S("L", d, k + cls), [f"u32.boundu{c}", B("L", d1, k)], loop_text(d1, False), "a uniform bound: the body keeps the L rung")
                    R(f"loop.tileu_strided_{c}_L{d}_f{k}", S("L", d, k + cls), [f"u32.boundu{c}", "u32.lit4096", B("L", d1, k)], loop_text(d1, False, stride="{1}", body="{2}"))
                for x in "LC":
                    R(f"loop.tilex_{c}_{x}{d}_f{k}", S(x, d, k + cls), [f"u32.boundx{c}", B("C", d1, k)], loop_text(d1, True), "an expression bound differs between invocations: the body is a C block")
                    R(f"loop.tilex_strided_{c}_{x}{d}_f{k}", S(x, d, k + cls), [f"u32.boundx{c}", "u32.lit4096", B("C", d1, k)], loop_text(d1, True, stride="{1}", body="{2}"))
                R(f"loop.tilex_{c}_W{d}_f{k}", S("W", d, k + cls), [f"u32.boundx{c}w", B("W", d1, k)], loop_text(d1, True))
                if DEPTH["S"]:
                    # a pool interface (s9 batch 9): the strided write-phase loop (an invocation's own elements t, t + T, ...)
                    R(f"loop.tilex_strided_{c}_W{d}_f{k}", S("W", d, k + cls), [f"u32.boundx{c}w", "u32.lit4096", B("W", d1, k)], loop_text(d1, True, stride="{1}", body="{2}"))
                for x in "LC":
                    R(f"loop.tilex_brk_{c}_{x}{d}_f{k}", S(x, d, k + cls), [f"u32.boundx{c}", BB(d1, k)], loop_text(d1, True), "a loop that breaks binds kx")
                R(f"loop.tilex_brk_{c}_W{d}_f{k}", S("W", d, k + cls), [f"u32.boundx{c}w", WBB(d1, k)], loop_text(d1, True))
            if k + 1 <= K:
                for x in "LC":
                    R(f"loop.tile_brk_{x}{d}_f{k}", S(x, d, k + 1), ["u32.lit64", BB(d1, k)], loop_text(d1, True), "a loop that breaks binds kx: after it the counter differs between invocations")
                R(f"loop.tile_brk_W{d}_f{k}", S("W", d, k + 1), ["u32.lit64", WBB(d1, k)], loop_text(d1, True))
            if d1 in DEPTH["S"]:
                # the solo phase's loops (s9 batch 9): one invocation, the body stays S
                if k + 1 <= K:
                    R(f"loop.tile_S{d}_f{k}", S("S", d, k + 1), ["u32.lit64", B("S", d1, k)], loop_text(d1, True), "a literal bound inside the solo phase")
                    R(f"loop.tile_strided_S{d}_f{k}", S("S", d, k + 1), ["u32.lit64", "u32.lit4096", B("S", d1, k)], loop_text(d1, True, stride="{1}", body="{2}"))
                    R(f"loop.tile_brk_S{d}_f{k}", S("S", d, k + 1), ["u32.lit64", SBB(d1, k)], loop_text(d1, True))
                for c in BOUND_CLASSES:
                    if k + c // 6 > K:
                        continue
                    R(f"loop.tilex_{c}_S{d}_f{k}", S("S", d, k + c // 6), [f"u32.boundx{c}", B("S", d1, k)], loop_text(d1, True))
                    R(f"loop.tilex_strided_{c}_S{d}_f{k}", S("S", d, k + c // 6), [f"u32.boundx{c}", "u32.lit4096", B("S", d1, k)], loop_text(d1, True, stride="{1}", body="{2}"))
                    R(f"loop.tilex_brk_{c}_S{d}_f{k}", S("S", d, k + c // 6), [f"u32.boundx{c}", SBB(d1, k)], loop_text(d1, True))
