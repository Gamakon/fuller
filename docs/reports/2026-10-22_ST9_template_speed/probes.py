"""Measurement probes: text edits of the encoded volumepassage (never kernels of the kingdom), one difference from the
shipped shader at a time. python3 probes.py ENCODED.wgsl OUTDIR"""
import re, sys, os
src = open(sys.argv[1]).read()
out = sys.argv[2]
os.makedirs(out, exist_ok=True)

def bare(s):
    return re.sub(r"(?<!fn )lit_(?:u32|f32|i32|bool|f16)\(([^()]*)\)", r"\1", s)

LOOP = re.compile(r"(k\w*) = 4294967295u; loop \{ \1 = \1 \+ 1u; if \(\1 >= (n\w*)\) \{ break; \} ")
def loop_for(s):
    s2, n = LOOP.subn(lambda m: f"{m[1]} = 0u; for (; {m[1]} < {m[2]}; {m[1]} = {m[1]} + 1u) {{ ", s)
    assert n == 2, n
    return s2

def bound_lit(s):
    # the counted loops' bounds read as their literals (n1 = 50u, n2 = 7u)
    vals = dict(re.findall(r"(n\d+) = (\d+u);", s))
    assert len(vals) == 2, vals
    s2, n = re.subn(r"(k\d+) (>=|<) (n\d+)\b", lambda m: f"{m[1]} {m[2]} {vals[m[3]]}", s)
    assert n == 2
    return s2

def zero(s):
    return re.sub(r"(?<!fn )zero_vec3_f32\(\)", "vec3<f32>()", s)

def axes(s):
    return s.replace("col_axis(gid)", "gid.x").replace("row_axis(gid)", "gid.y")

def early(s):
    a = "let nwg: vec3<u32> = select(nwg_tile, dispatch_tile.grid.xyz, DISPATCH_TILED);\n"
    assert a in s
    return s.replace(a, a + "if (gid.x >= textureDimensions(output).x || gid.y >= textureDimensions(output).y) { return; }\n")

def notile(s):
    s = s.replace("@builtin(global_invocation_id) gid_tile: vec3<u32>", "@builtin(global_invocation_id) gid: vec3<u32>")
    s = s.replace("@builtin(workgroup_id) wid_tile: vec3<u32>", "@builtin(workgroup_id) wid: vec3<u32>")
    s = s.replace("@builtin(num_workgroups) nwg_tile: vec3<u32>", "@builtin(num_workgroups) nwg: vec3<u32>")
    for l in ["let wid: vec3<u32> = select(wid_tile, wid_tile + dispatch_tile.origin.xyz, DISPATCH_TILED);\n",
              "let gid: vec3<u32> = select(gid_tile, gid_tile + dispatch_tile.origin.xyz * vec3<u32>(WG_X, WG_Y, 1u), DISPATCH_TILED);\n",
              "let nwg: vec3<u32> = select(nwg_tile, dispatch_tile.grid.xyz, DISPATCH_TILED);\n",
              "@group(3) @binding(0) var<uniform> dispatch_tile: DispatchTile;\n", "override DISPATCH_TILED: bool = true;\n"]:
        assert l in s, l
        s = s.replace(l, "")
    return s

probes = {"encoded": src, "bare": bare(src), "for": loop_for(src), "boundlit": bound_lit(src), "zero": zero(src), "axes": axes(src),
          "notile": notile(src)}
probes["all"] = bare(bound_lit(loop_for(zero(axes(notile(src))))))
probes["all_but_for"] = bare(bound_lit(zero(axes(notile(src)))))
for k, v in probes.items():
    open(f"{out}/{k}.wgsl", "w").write(v)
    print(k, len(v))

# loop spellings without a continuing block (naga writes no loop_init for any of them)
BODY_END = re.compile(r"(k\d+) = 4294967295u; loop \{ \1 = \1 \+ 1u; if \(\1 >= (n\d+)\) \{ break; \} ")
def loop_shadow(s):
    # the step on a shadow counter kc: k = kc at the top (the trip inside, the final value after, a continue takes the step)
    s2, n = BODY_END.subn(lambda m: f"kc{m[1][1:]} = 0u; loop {{ {m[1]} = kc{m[1][1:]}; if ({m[1]} >= {m[2]}) {{ break; }} kc{m[1][1:]} = {m[1]} + 1u; ", s)
    assert n == 2, n
    a = "var k1: u32 = 0u;\n"
    assert a in s2
    return s2.replace(a, a + "var kc1: u32 = 0u;\nvar kc2: u32 = 0u;\n")

def loop_end(s):
    # the step at the end of the body (a probe only: a `continue` in the body would skip it); one loop a call
    m = BODY_END.search(s)
    k, nn = m[1], m[2]
    j, depth = m.end(), 1
    while depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    body = s[m.end():j - 1]
    return s[:m.start()] + f"{k} = 0u; loop {{ if ({k} >= {nn}) {{ break; }} {body}{k} = {k} + 1u; }}" + s[j:]

def loop_end_all(s):
    while BODY_END.search(s):
        s = loop_end(s)
    return s

for k, v in {"shadow": loop_shadow(src), "end": loop_end_all(src)}.items():
    open(f"{out}/{k}.wgsl", "w").write(v)
    print(k, len(v))
