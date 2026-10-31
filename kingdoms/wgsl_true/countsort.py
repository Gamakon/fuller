"""Compiler-assigned slots for the scatter (docs/DESIGN_inrange_index_types.md section 10, approved 2026-10-31).

A gene names the cluster a value belongs to; it never names a slot. The compiler assigns the slot when the gene is
expressed into WGSL, from a deterministic counting-sort scaffold it owns:

* the entries `hist`, `offsets` and `starts` are the scaffold's (the gene cannot write them): the counts per cluster
  per block, the prefix sums, and each cluster's first slot;
* the scatter entry's head and prologue are the scaffold's: they compute the invocation's point `cs_v`, its cluster
  `cs_a`, whether it stores (`cs_ok`) and its slot `cs_slot = start[a] + prefix[block][a] + rank`, the rank being the
  number of earlier points of the block with the same cluster, in index order (a stable rank: no atomics, the same slot
  on every run, each cluster's points in index order, so the mean sums them in A7's order);
* the gene's only data-indexed store is `store.cluster_<R>(u32.cluster, value)`: no position argument. `u32.cluster`
  has one producer, the point's own cluster `cs_a`: a store under any other cluster has no slot the counting sort
  counted, and two invocations could then meet in one cell.

The scaffold comes in a fixed set of correct variants (the layout of the scratch, the methods of hist, offsets and
starts, the scatter's workgroup width and rank method). The gene chooses one through a typed knob: `kernel.hist`,
whose terminals are the variants (`cs.<variant>`, spelled as the variant's module-scope text and hist entry). The
other pieces of the same variant are spelled by the markers `<cs:offsets>`, `<cs:starts>`, `<cs:scatter_head>` and
`<cs:scatter_prologue>`, which the renderer (TrueKingdom::render) resolves from the module's one knob, as it resolves
`<declarations>` from the tree. One knob per module, so every piece of one module is one variant's.

Outside the scatter entry the cluster store's names are declared and never set (`cs_ok` stays false), so a cluster
store a gene places in another entry stores nothing.

The variants' texts are the reference kernels' own (the segmented form, Sonnet's reply 2 and the seed-1 champion,
read from their committed files) and the compiler's scatter prologues below; `scaffold.tsv` holds them, one row per
(variant, piece). The witness (`witness`, and `python3 countsort.py DIR` on a written table) refuses a table where a
durable binding is written other than through an own-index store or the cluster store, or a scaffold differs from a
variant byte for byte.
"""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: E402,F403

REPO = os.environ.get("FULLER_REPO") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCES = {
    "seg": "problems/kmeans_update_segmented/segmented.wgsl",
    "r2": "tests/corpus/kmeans_model_replies/reply2_g0_r4.wgsl",
    "champ": "docs/kmeans_champion_s1/champion.wgsl",
    "none": "problems/kmeans_update_segmented/gather.wgsl",
}
PIECES = ("module", "hist", "offsets", "starts", "scatter_head", "scatter_prologue", "source_hist", "source_offsets", "source_starts", "source_scatter")
# the identity a gene's layout needs on each scaffold entry type (a head holds at least one function)
SAME = {"kernel.hist": "kernel.cs_hist_same", "kernel.offsets": "kernel.cs_offsets_same", "kernel.starts": "kernel.cs_starts_same"}


def fresh(t):
    """The champion's own share names (`st7_s0`, the share pass's) spelled as the scaffold's (`cs_st7_s0`): a share
    name in a row's spelling would not be fresh for the share pass."""
    return re.sub(r"\bst(\d+)_s(\d+)\b", r"cs_st\1_s\2", t)


NAMES = ("cs_v", "cs_a", "cs_ok", "cs_slot")
MODULE_IDS = ("@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, "
              "@builtin(local_invocation_index) lidx: u32, @builtin(num_workgroups) nwg: vec3<u32>")


def enabled(iface):
    return bool(iface.get("counting_sort"))


def entries(iface):
    """The module's entries the scaffold owns or shapes: hist, offsets, starts (owned) and the scatter (head and prologue)."""
    return tuple(iface["counting_sort"]["entries"]) if enabled(iface) else ()


def entry_text(src, name):
    """The text of entry `name` in `src`: from its `@compute` line through the brace that closes its body (the reader,
    true_reader.rs `entry_text`, cuts the same way)."""
    m = re.search(r"@compute[^\n]*\n\s*fn " + re.escape(name) + r"\(", src)
    if not m:
        raise SystemExit(f"our software has a bug: no entry {name}")
    i = src.index("{", src.index(")", m.end()))
    depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start():j + 1]
    raise SystemExit(f"our software has a bug: entry {name} does not close")


def module_text(src):
    """The module-scope text a source declares between its last binding and its first entry that no kingdom prologue
    holds: reply 2's workgroup array and its two helpers. The segmented form has none; the champion's are the
    kingdom's own prologue and pool, so it has none of its own."""
    head = src[:src.index("@compute")]
    lines = head.split("\n")
    last = max(i for i, l in enumerate(lines) if l.startswith("@group("))
    return "\n".join(lines[last + 1:]).strip("\n") + "\n"


def norm(t):
    """Whitespace collapsed: the reader matches an entry against a piece in this form."""
    return " ".join(t.split())


def head(wx, wy):
    knobs = (f"const KNOB_WCOL: u32 = {wx}u; const KNOB_WROW: u32 = {wy}u; const KNOB_BCOL: u32 = 1u; const KNOB_BROW: u32 = 1u; "
             f"const KNOB_KTILE: u32 = 1u; const KNOB_THREADS: u32 = {wx}u * {wy}u;\n")
    return f"@compute @workgroup_size({wx}, {wy}, 1)\nfn scatter({MODULE_IDS}) {{\n" + knobs


# The scatter prologue of the segmented layout ([counts nb k][prefixes nb k][lens k][starts k][ids n], nb = ceil(n /
# 256)): one invocation per point, any workgroup width, the rank counted over the block's earlier points in the
# assignments buffer (the segmented form's method). Every subscript is in range of its binding by the contract's
# lengths (cs_v < n, cs_u < cs_v, cs_a < k); the store's own clamp keeps it in range whatever the data.
SEG_PROLOGUE = (
    "{ let cs_n = params.x; let cs_k = params.y; let cs_nb = (cs_n + 255u) / 256u;\n"
    "cs_v = gid.x;\n"
    "cs_ok = cs_v < cs_n;\n"
    "if (cs_ok) { cs_a = assignments[cs_v]; cs_ok = cs_a < cs_k; }\n"
    "if (cs_ok) {\n"
    "let cs_b = cs_v / 256u;\n"
    "var cs_rank: u32 = 0u;\n"
    "for (var cs_i: u32 = 0u; cs_i < 256u; cs_i = cs_i + 1u) { let cs_u = cs_b * 256u + cs_i; if (cs_u < cs_v) { if (assignments[cs_u] == cs_a) { cs_rank = cs_rank + 1u; } } }\n"
    "cs_slot = 2u * cs_nb * cs_k + 2u * cs_k + scratch[2u * cs_nb * cs_k + cs_k + cs_a] + scratch[cs_nb * cs_k + cs_b * cs_k + cs_a] + cs_rank;\n"
    "} }"
)

# The scatter prologue of reply 2's layout ([counts, then prefixes in place, nb k][lens k][starts k][ids n]): a
# workgroup of 64 x 2 = 128 invocations, two per block of 256 points, the block's labels staged in its workgroup array
# `sh_a` (reply 2's module text declares it), the rank counted there (reply 2's method). `scratch_ok()` and
# `nblocks()` are reply 2's helpers.
R2_PROLOGUE = (
    "{ let cs_n = params.x; let cs_k = params.y; let cs_w = wid.x; let cs_b = cs_w / 2u;\n"
    "let cs_a0 = cs_b * 256u + lidx; let cs_a1 = cs_a0 + 128u;\n"
    "sh_a[lidx] = select(0xffffffffu, assignments[min(cs_a0, arrayLength(&assignments) - 1u)], cs_a0 < cs_n);\n"
    "sh_a[lidx + 128u] = select(0xffffffffu, assignments[min(cs_a1, arrayLength(&assignments) - 1u)], cs_a1 < cs_n);\n"
    "workgroupBarrier();\n"
    "cs_v = cs_w * 128u + lidx;\n"
    "let cs_p = (cs_w & 1u) * 128u + lidx;\n"
    "cs_a = sh_a[cs_p];\n"
    "cs_ok = cs_v < cs_n && scratch_ok() && cs_a < cs_k;\n"
    "if (cs_ok) {\n"
    "var cs_rank: u32 = 0u;\n"
    "for (var cs_q: u32 = 0u; cs_q < 256u; cs_q = cs_q + 1u) { if (cs_q >= cs_p) { break; } if (sh_a[cs_q] == cs_a) { cs_rank = cs_rank + 1u; } }\n"
    "let cs_cb = nblocks() * cs_k;\n"
    "cs_slot = cs_cb + 2u * cs_k + scratch[cs_cb + cs_k + cs_a] + scratch[cs_b * cs_k + cs_a] + cs_rank;\n"
    "} }"
)


# The scatter prologue of the variant with no counting sort (A7's gather in the module form, whose hist, offsets,
# starts and scatter are empty): no invocation has a slot, so a cluster store stores nothing.
NONE_PROLOGUE = "{ cs_ok = false; }"


def variants():
    """Every scaffold variant: name -> {piece: text}. The segmented layout's hist, offsets and starts come from the
    segmented form or the champion (the champion was evolved from it and writes the same cells), in every combination,
    with a scatter of width 64, 128 or 256; reply 2's layout is reply 2's own. Ordered by name."""
    src = {k: open(os.path.join(REPO, p)).read() for k, p in SOURCES.items()}
    out = {}
    # the scatter texts the reader recognises for each layout: the sources' own scatter entries
    seg_scatters = "\n\n".join(entry_text(src[s], "scatter") for s in ("seg", "champ"))

    def pieces(h, o, s):
        e = {"hist": entry_text(src[h], "hist"), "offsets": entry_text(src[o], "offsets"), "starts": entry_text(src[s], "starts")}
        return {**{p: fresh(t) for p, t in e.items()}, **{f"source_{p}": t for p, t in e.items()}}

    for h, o, s in itertools.product(("seg", "champ"), repeat=3):
        for w in (64, 128, 256):
            out[f"seg_h{h}_o{o}_s{s}_w{w}"] = dict(module="", **pieces(h, o, s), scatter_head=head(w, 1), scatter_prologue=SEG_PROLOGUE,
                                                   source_scatter=seg_scatters if w == 256 else "")
    out["r2"] = dict(module=module_text(src["r2"]), **pieces("r2", "r2", "r2"), scatter_head=head(64, 2), scatter_prologue=R2_PROLOGUE,
                     source_scatter=entry_text(src["r2"], "scatter"))
    out["none"] = dict(module="", **pieces("none", "none", "none"), scatter_head=head(256, 1), scatter_prologue=NONE_PROLOGUE,
                       source_scatter=entry_text(src["none"], "scatter"))
    return dict(sorted(out.items()))


def knob_text(v):
    return (v["module"] + "\n" if v["module"].strip() else "") + v["hist"] + "\n"


def declare(iface):
    if not enabled(iface):
        return
    T("u32.cluster", "the cluster of the invocation's own point (the scaffold's cs_a): the only cluster a cluster store takes")
    for n in NAMES:
        decl(f"var {n}: {'bool' if n == 'cs_ok' else 'u32'} = {'false' if n == 'cs_ok' else '0u'};")


def build(iface):
    """The scaffold's rows: the knob (`kernel.hist`'s terminals), the offsets and starts terminals, the scatter entry,
    the cluster and point of the invocation, and the cluster store at every L and C rung."""
    if not enabled(iface):
        return
    cs = iface["counting_sort"]
    ids = cs["ids"]
    pool = bool(iface.get("wg_pool"))
    for name, v in variants().items():
        R(f"cs.{name}", "kernel.hist", [], knob_text(v), f"the counting-sort scaffold {name}: its module text and hist entry (the knob: the module's other scaffold pieces are this variant's)")
    R("cs.offsets", "kernel.offsets", [], "<cs:offsets>\n", "the scaffold's offsets entry, of the module's variant (the knob)")
    R("cs.starts", "kernel.starts", [], "<cs:starts>\n", "the scaffold's starts entry, of the module's variant (the knob)")
    for t, n in SAME.items():
        R(n, t, [t], "{0}", "the identity: a gene's layout holds at least one function, and a scaffold entry has none of its own")
    from start import body_type
    args = (["u32.wgext"] if pool else []) + [body_type()]
    text = ("// pool {0} words\n" if pool else "") + "<cs:scatter_head>// counting sort: " + " ".join(NAMES) + "\n<declarations>\n<cs:scatter_prologue>\n{%d}\n}\n" % (len(args) - 1)
    R("kernel.entry_scatter", "kernel.scatter", args, text, "the scatter entry: the scaffold's head and prologue (the slot of the invocation's point), then the gene's body")
    R("cluster.own", "u32.cluster", [], "cs_a", "the cluster of the invocation's own point, as the scaffold read it")
    R("cluster.point", "u32.index", [], "cs_v", "the invocation's own point, as the scaffold numbered it", twin=True)
    for x, d, k in rungs("LC"):
        R(f"store.cluster_{ids}_{x}{d}_f{k}", S(x, d, k), ["u32.cluster", fam(x, "u32.index")], store_text(ids),
          "the gene's only data-indexed store: the value at the slot the scaffold assigned the invocation's point (no position argument)")


def store_text(ids):
    return f"{{ _ = {{0}}; if (cs_ok) {{ {ids}[min(cs_slot, arrayLength(&{ids}) - 1u)] = {{1}}; }} }}"


def write(outdir, iface):
    """scaffold.tsv: one row per (variant, piece), the text escaped as functions.tsv escapes it."""
    if not enabled(iface):
        return
    with open(os.path.join(outdir, "scaffold.tsv"), "w") as f:
        f.write("variant\tpiece\ttext\n")
        for name, v in variants().items():
            for p in PIECES:
                f.write(f"{name}\t{p}\t{esc(v[p])}\n")


# ---------------------------------------------------------------- the witness

def durable(iface):
    """The bindings a kernel writes that outlive the dispatch: the read_write storage bindings of the prologue."""
    return re.findall(r"var<storage,\s*read_write>\s+(\w+)\s*:", iface["prologue_top"])


def writes(text, b):
    """Every write of binding `b` in `text`: a subscript `b[...]` followed by `=` (not `==`), or an atomic on `&b[`."""
    out = []
    for m in re.finditer(r"\b" + re.escape(b) + r"\[", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"[": 1, "]": -1}.get(text[j], 0)
            j += 1
        rest = text[j:].lstrip()
        if rest.startswith("=") and not rest.startswith("=="):
            out.append(text[m.start():j])
    out += re.findall(r"atomic\w+\(\s*&" + re.escape(b) + r"\[", text)
    return out


def witness(iface, rows=None):
    """Section 10's witness: every durable write goes through an own-index store or the cluster store and through no
    other row; every scaffold is one of the compiler's variants, byte for byte."""
    if not enabled(iface):
        return []
    rows = ROWS if rows is None else rows
    cs = iface["counting_sort"]
    ids = cs["ids"]
    vs = variants()
    problems = []
    for n, r in rows.items():
        for b in durable(iface):
            w = writes(r["spell"], b)
            # a knob row is a variant's own text (checked byte for byte below): the scaffold's writes are the compiler's
            if not w or (n.startswith("cs.") and n[3:] in vs):
                continue
            if b == ids and n.startswith(f"store.cluster_{ids}_") and r["spell"] == store_text(ids) and r["args"][0] == "u32.cluster":
                continue
            if r["args"][:2] == [f"u32.threadrow_{b}", f"u32.threadcol_{b}"] and n.startswith(f"store.{b}"):
                # section 9's own-index store: the cell is the invocation's own by its argument types (the row and
                # column of the invocation's own cell, skeleton.py store_text)
                continue
            problems.append(f"{n} writes the durable binding {b} ({w[0]}) other than through an own-index store or the cluster store")
        for m in re.findall(r"<cs:(\w+)>", r["spell"]):
            if m not in PIECES or (n, m) not in {("cs.offsets", "offsets"), ("cs.starts", "starts"), ("kernel.entry_scatter", "scatter_head"), ("kernel.entry_scatter", "scatter_prologue")}:
                problems.append(f"{n} spells the scaffold marker <cs:{m}>, which only the scaffold's own rows spell")
        if re.search(r"\b(cs_v|cs_a|cs_ok|cs_slot)\s*=[^=]", r["spell"]):
            problems.append(f"{n} assigns a name of the scaffold: only the scaffold's prologue does")
    produced = {}
    for n, r in rows.items():
        produced.setdefault(r["out"], []).append(n)
    want = {f"cs.{k}" for k in vs} | {SAME["kernel.hist"]}
    if set(produced.get("kernel.hist", [])) != want:
        problems.append(f"kernel.hist (the knob) has producers {sorted(set(produced.get('kernel.hist', [])) ^ want)} beside or short of the variants")
    for k, v in vs.items():
        r = rows.get(f"cs.{k}")
        if r and (r["spell"] != knob_text(v) or r["args"]):
            problems.append(f"cs.{k} is not the variant's module text and hist entry byte for byte")
    for t, p in (("kernel.offsets", ["cs.offsets", SAME["kernel.offsets"]]), ("kernel.starts", ["cs.starts", SAME["kernel.starts"]]), ("u32.cluster", ["cluster.own"])):
        if produced.get(t, []) != p:
            problems.append(f"{t} has producers {produced.get(t, [])}, not only {p}")
    for t, n in SAME.items():
        if n in rows and (rows[n]["spell"] != "{0}" or rows[n]["args"] != [t]):
            problems.append(f"{n} is not the identity of {t}")
    if sorted(produced.get("kernel.scatter", [])) != ["kernel.entry_scatter", "own.kernel_scatter"]:
        problems.append(f"kernel.scatter has producers {produced.get('kernel.scatter', [])}")
    return problems


def check_dir(d, iface):
    """`python3 countsort.py DIR`: the scaffold.tsv of a written table is the compiler's variants byte for byte."""
    want = "variant\tpiece\ttext\n" + "".join(f"{n}\t{p}\t{esc(v[p])}\n" for n, v in variants().items() for p in PIECES)
    got = open(os.path.join(d, "scaffold.tsv")).read()
    return [] if got == want else [f"{d}/scaffold.tsv differs from the compiler's variants"]


if __name__ == "__main__":
    from interface import INTERFACES
    d = sys.argv[1]
    iface = INTERFACES[os.path.basename(os.path.normpath(d))]
    ps = check_dir(d, iface)
    for p in ps:
        print("PROBLEM:", p)
    sys.exit(1 if ps else 0)
