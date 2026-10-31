"""The entry-start type (2026-10-31): a workgroup phase that opens an entry's body spells no barrier before it.

Every pool phase row (pool.py) is `workgroupBarrier(); <phase> workgroupBarrier(); <rest>`. The leading barrier orders
the phase after whatever the block ran before it. When the phase is the first statement of an entry nothing ran before
it in the dispatch (workgroup memory is zero-initialised before the entry, visible to every invocation), so the barrier
only costs time: measured on reply 2's scatter entry, 0.0272 ms with it against 0.0266 ms (the source) without it
(fbench/r2b.log).

The rule is a type. An entry's body is `block.start`, not `block.L0_f<K>`; its rows are:

| Row | Spelling | Why it is legal |
|---|---|---|
| `block.start_body(block.L0_f<K>)` | `{0}` | any body, its phases keep their leading barrier |
| `own.block_start` | `{}` | the empty body |
| `wg.<kind>_start(...)` | the phase row `wg.<kind>_L0_f<K>`'s spelling without `workgroupBarrier(); ` | nothing runs before it |

`block.start` is an argument of the entry rows only, so a first-phase row stands nowhere but at the start of an entry.

`witness` proves from the emitted text: (0) every other row that spells a barrier opens with it (or is the bare
barrier statement), so no phase without its leading barrier stands at any other type; (1) the rows giving `block.start` are those three forms, each first-phase row
the exact text of its phase row less the leading barrier, with the same arguments; (2) only `kernel.entry_*` rows take
`block.start`, as their body, and their text before the body (the grid's knobs, the passes header, `<declarations>`)
names no workgroup variable, no helper that reaches one, and no barrier; (3) the function-scope declarations name no
workgroup variable and no such helper. The share pass binds only rows that read no workgroup memory (its context-free
rule, src/wgsl/true_share.rs, tested), so its `let`s ahead of the body never read the pool either.
"""
import re

from lib import *

START = "block.start"
LEAD = "workgroupBarrier(); "


def enabled():
    """An interface whose entries can open with a pool phase: phases exist at the entry body's rung L0."""
    return 0 in DEPTH["S"]


def body_type():
    return START if enabled() else B("L", 0, K)


def first_phase(name):
    """`wg.<kind>_L0_f<K>` -> `wg.<kind>_start`."""
    base, sep, _ = name.rpartition(f"_L0_f{K}")
    assert sep, name
    return base + "_start"


def declare(iface):
    if enabled():
        T(START, "the body of a module entry: it runs first in its dispatch, so a workgroup phase opening it needs no barrier before it")


def build(iface):
    if enabled():
        R("block.start_body", START, [B("L", 0, K)], "{0}", "an entry body that does not open with a first-phase row: its phases keep their barriers")
        R("own.block_start", START, [], "{}", "the empty entry body")


# ---------------------------------------------------------------- the witness

def _workgroup_names(prologue):
    """The workgroup variables the prologue declares and every helper whose body reaches one (transitively)."""
    names = set(re.findall(r"var<workgroup>\s+(\w+)\s*:", prologue))
    bodies = {}
    for m in re.finditer(r"\bfn\s+(\w+)\s*\(", prologue):
        start = prologue.find("{", m.end())
        depth, j = 0, start
        while True:
            depth += {"{": 1, "}": -1}.get(prologue[j], 0)
            j += 1
            if depth == 0:
                break
        bodies[m.group(1)] = prologue[start:j]
    changed = True
    while changed:
        changed = False
        for f, body in bodies.items():
            if f not in names and any(re.search(r"\b" + re.escape(n) + r"\b", body) for n in names):
                names.add(f)
                changed = True
    return names


def _touches(text, wg):
    return any(re.search(r"\b" + re.escape(n) + r"\b", text) for n in wg) or "workgroupBarrier" in text or "workgroupUniformLoad" in text


def witness_rows(rows, prologue, decls):
    """The rules of the module docstring over rows {name: {out, args, spell}}, the module prologue and the
    function-scope declarations. Returns the problems."""
    problems = []
    takers = [n for n, r in rows.items() if START in r["args"]]
    givers = [n for n, r in rows.items() if r["out"] == START]
    # rule 0 holds in a pool table (pool.py's phases); the stratum-map tables (wg.py) have their own barrier rules
    pool = re.search(r"var<workgroup>\s+wg_pool\s*:", prologue) is not None
    for n, r in rows.items():
        sp = r["spell"]
        if pool and "workgroupBarrier" in sp and r["out"] != START and not sp.startswith(LEAD) and sp != LEAD.strip():
            problems.append(f"row {n} spells a phase with no barrier before it outside {START}")
    if not givers and not takers:
        return problems
    wg = _workgroup_names(prologue)
    if not wg:
        problems.append("block.start in a table whose prologue declares no workgroup variable")
    for n in givers:
        r = rows[n]
        if n == "block.start_body":
            if r["args"] != [B("L", 0, K)] or r["spell"] != "{0}":
                problems.append(f"row {n} is not {B('L', 0, K)} spelled {{0}}")
        elif n == "own.block_start":
            if r["args"] or r["spell"] != "{}":
                problems.append(f"row {n} is not the empty body")
        elif n.startswith("wg.") and n.endswith("_start"):
            p = rows.get(n.removesuffix("_start") + f"_L0_f{K}")
            if p is None or p["out"] != S("L", 0, K):
                problems.append(f"row {n} has no phase row at L0_f{K}")
            elif p["args"] != r["args"] or LEAD + r["spell"] != p["spell"]:
                problems.append(f"row {n} is not its phase row less the leading barrier")
            elif r["spell"].startswith("workgroupBarrier"):
                problems.append(f"row {n} still opens with a barrier")
        else:
            problems.append(f"row {n} gives {START} and is none of its three forms")
    for n in takers:
        r = rows[n]
        i = len(r["args"]) - 1
        if not n.startswith("kernel.entry_") or r["args"].index(START) != i:
            problems.append(f"row {n} takes {START} and is not an entry taking it as its body")
            continue
        if any(not (a.startswith("grid.") or a == "u32.passes") for a in r["args"][:i]):
            problems.append(f"entry {n} takes an argument before its body that is not its grid or passes: {r['args'][:i]}")
        at = r["spell"].find("{" + str(i) + "}")
        if at < 0 or _touches(r["spell"][:at], wg):
            problems.append(f"entry {n}'s text before its body touches workgroup memory or a barrier")
    if _touches(decls, wg):
        problems.append("the function-scope declarations touch workgroup memory or a barrier")
    return problems


def witness(iface):
    prologue = iface.get("prologue_top", "") + "\n".join(PROLOGUE_TOP) + "\n" + "\n".join(HELPERS[n] for n in sorted(HELPERS))
    return witness_rows(ROWS, prologue, "\n".join(DECLS))


def witness_committed(table_dir):
    """The witness over a COMMITTED table: `python3 start.py mod/kmeans_update_segmented`."""
    unesc = lambda s: s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    rows = {}
    for line in open(os.path.join(table_dir, "functions.tsv")).read().split("\n")[1:]:
        if line:
            p = line.split("\t")
            rows[p[1]] = dict(out=p[3], args=p[4].split(";") if p[4] else [], spell=unesc(p[5]))
    meta = dict(line.rstrip("\n").split("\t", 1) for line in open(os.path.join(table_dir, "interface.tsv")) if "\t" in line)
    global K
    K = int(meta["fuel_budget"])
    return witness_rows(rows, open(os.path.join(table_dir, "prologue.wgsl")).read(), open(os.path.join(table_dir, "declarations.wgsl")).read())


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    probs = witness_committed(d if os.path.isabs(d) else os.path.join(HERE, d))
    for p in probs:
        print("PROBLEM:", p)
    print(f"start witness: {len(probs)} problems")
    sys.exit(1 if probs else 0)
