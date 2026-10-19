#!/usr/bin/env python3
"""Fill docs/prompts/gep_translate.md's placeholders from the kingdom's own tables and a
problem, and write one full prompt per program. Text only; no kernel runs.
Usage: build_gep_translate.py --problem <dir> --out <dir> <program.wgsl>..."""
import argparse, csv, os, re, sys
F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = os.path.join(F, "kingdoms/safewgsl")

def tsv(path):
    rows = [r for r in csv.reader(open(path), delimiter="\t") if r]
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]

def dual_table():
    out = []
    for t in tsv(os.path.join(K, "types.tsv")):
        lit = t.get("literal", "-")
        out.append(f"{t['ref']} {t['dual']} — {t['meaning']}" + (f" — literal {lit}" if lit and lit != "-" else ""))
    return "\n".join(out)

def symbol_table():
    by = {}
    for r in tsv(os.path.join(K, "functions.tsv")):
        if r["class"] in ("load", "store", "arg") and r["instance"] in ("buffer", "uniform", "local", "<name>", "branch"):
            continue  # the problem's own rows are in A7
        note = (" — " + r["note"]) if r.get("note") else ""
        by.setdefault(r["class"], []).append(f"{r['class']}.{r['instance']} : {r['in'].replace(' ', ', ') if r['in'] != '-' else '(none)'} -> {r['out']} [{r['arity']}]{note}")
    return "\n\n".join(f"class {c}\n" + "\n".join(v) for c, v in by.items())

def interface(problem):
    toml = open(os.path.join(problem, "problem.toml")).read()
    names = re.findall(r'^\s*name\s*=\s*"([^"]+)"', toml, re.M)
    text = open(os.path.join(problem, "contract.md")).read()
    cfg = re.search(r"struct Cfg \{([^}]*)\}", text)
    fields = [f.split(":")[0].strip() for f in (cfg.group(1).split(",") if cfg else []) if f.strip()]
    lines = ["Bindings (group 0) and the rows each gives:"]
    for n in names:
        if n == "cfg":
            continue
        lines.append(f"- `{n}`: `load.{n} : u32.index -> <the element type: f32 buffers .real, u32 buffers .index>` [1]; "
                     f"a store `store.{n}_<rung> : u32.index, <element type> -> stmt.<rung>` [2] where the buffer is read_write; "
                     f"storage atomics `atomic.{n}_<fun>` on a u32 buffer declared atomic.")
    lines.append("Uniform fields of cfg, each a terminal: " + ", ".join(f"`load.cfg_{f} : -> u32.count`" for f in fields)
                 + "; as a loop bound the same field is `uniform.cfg_<field> : -> u32.uniform`.")
    lines.append("Entry points: `ingest`, then `search`, in that order: the root is `kernel.module_2` with two `kernel.entry_safe` children.")
    lines.append("\nThe problem's contract (the bindings' meaning):\n\n" + text)
    return "\n".join(lines)

def pretty(sexpr):
    """One node a line, indented by depth, a literal kept on its line."""
    toks = re.findall(r'\(|\)|[^\s()]+', sexpr); out = []; depth = 0; i = 0
    while i < len(toks):
        t = toks[i]
        if t == "(" and toks[i + 1].startswith("literal.") and i + 3 < len(toks) and toks[i + 3] == ")":
            out.append("  " * depth + f"({toks[i+1]} {toks[i+2]})"); i += 4; continue
        if t == "(":
            out.append("  " * depth + "(" + toks[i + 1]); depth += 1; i += 2; continue
        if t == ")":
            out[-1] += ")"; depth -= 1; i += 1; continue
        out.append("  " * depth + t); i += 1
    return "\n".join(out)

def parse_sexpr(sexpr):
    toks = re.findall(r'\(|\)|[^\s()]+', sexpr); i = 0
    def node():
        nonlocal i
        if toks[i] == "(":
            i += 1; name = toks[i]; i += 1; kids = []
            while toks[i] != ")": kids.append(node())
            i += 1; return [name, kids]
        i += 1; return [toks[i - 1], []]
    return node()

def fold_literals(n):
    """(literal.form VALUE) as one symbol literal.form=VALUE."""
    name, kids = n
    if name.startswith("literal.") and len(kids) == 1 and not kids[0][1]:
        v = kids[0][0]
        try:
            f = float(v); v = repr(int(f)) if f == int(f) and abs(f) < 1e15 else ("%g" % f)
        except ValueError: pass
        return [f"{name}={v}", []]
    return [name, [fold_literals(c) for c in kids]]

def level_order(root):
    out = []; level = [root]
    while level:
        out.extend(level); level = [c for n in level for c in n[1]]
    return out

def literal_ranges():
    r = {}
    for t in tsv(os.path.join(K, "types.tsv")):
        lit = t.get("literal", "-")
        if lit and lit != "-": r[t["form"]] = lit
    return r

def padding(n, rng, extra_terminals):
    """n varied valid terminals: arity-0 rows of the table (not fallback), literals of many forms
    with values in range, and the problem's own terminals; no symbol twice in a row."""
    pool = [f"{r['class']}.{r['instance']}" for r in tsv(os.path.join(K, "functions.tsv"))
            if r["arity"] == "0" and r["class"] != "literal" and "<" not in r["instance"] and r["instance"] not in ("buffer", "uniform", "local")]
    pool = sorted(set(pool + extra_terminals) - {"fallback"})
    ranges = literal_ranges(); forms = sorted(ranges)
    def lit():
        f = rng.choice(forms); spec = ranges[f]
        m = re.match(r"^(-?[0-9.e+]+)\.\.=(-?[0-9.e+]+)$", spec)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            if f in ("real",): v = "%g" % rng.uniform(max(lo, -1e3), min(hi, 1e3))
            else: v = str(rng.randint(int(lo), int(min(hi, 1 << 20))))
        else:
            vals = re.findall(r"-?[0-9.]+", spec); v = rng.choice(vals) if vals else "1"
        return f"literal.{f}={v}"
    out = []
    while len(out) < n:
        sym = lit() if rng.random() < 0.4 else rng.choice(pool)
        if out and (sym == out[-1] or sym.split("=")[0] == out[-1].split("=")[0]): continue
        out.append(sym)
    return out

def karva_lines(sexpr, rng, extra_terminals, arity_of):
    root = fold_literals(parse_sexpr(sexpr))
    orf = level_order(root)
    f = sum(1 for n in orf if n[1])
    lines = [f"{i} {n[0]}" for i, n in enumerate(orf)] + ["ORF END"]
    pad = padding(3 * f + 1, rng, extra_terminals)
    lines += [f"{len(orf) + i} {sym}" for i, sym in enumerate(pad)]
    return "\n".join(lines), len(orf), f

def example(program, chromosome):
    import random
    sx = open(chromosome).read().strip()
    extra = sorted(set(re.findall(r"\b(?:load|uniform)\.[a-z0-9_]+(?=[\s)])", sx)) - {n for n in re.findall(r"\((load\.[a-z0-9_]+)", sx)})
    gene, n_orf, f = karva_lines(sx, random.Random(20261019), extra, None)
    return ("Program (WGSL):\n\n```wgsl\n" + open(program).read().strip() + "\n```\n\nIts tree (brackets only to show the shape):\n\n```\n"
            + pretty(sx) + f"\n```\n\nIts gene in Karva, the form you return: {n_orf} symbols in the ORF, F = {f} of them of arity 1 or more, so 3 × {f} + 1 = {3*f+1} padding terminals:\n\n```karva\n" + gene + "\n```")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--problem", required=True); ap.add_argument("--out", required=True); ap.add_argument("--example-program", required=True); ap.add_argument("--example-chromosome", required=True); ap.add_argument("programs", nargs="+")
    a = ap.parse_args()
    md = open(os.path.join(F, "docs/prompts/gep_translate.md")).read()
    head = md.split("## PART A — the fixed header (cached)", 1)[1].split("## PART B — the per-call challenge", 1)[0]
    part_b = md.split("### B1. The challenge", 1)[1].split("### B2.", 1)[0]
    head = re.sub(r"<!-- builder:.*?-->\n?", "", head, flags=re.S)
    head = head.replace("{{DUAL_TABLE}}", dual_table()).replace("{{SYMBOL_TABLE}}", symbol_table()).replace("{{INTERFACE}}", interface(a.problem)).replace("{{FULL_EXAMPLE}}", example(a.example_program, a.example_chromosome))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "header.txt"), "w").write(head.strip() + "\n")
    for p in a.programs:
        name = os.path.splitext(os.path.basename(p))[0]
        task = part_b.replace("{{PROBLEM}}", os.path.basename(a.problem.rstrip("/"))).replace("{{ENTRIES}}", "`ingest`, `search`").replace("{{PROGRAM}}", open(p).read().strip())
        open(os.path.join(a.out, f"{name}.task.txt"), "w").write("Translate the program into a safeWGSL chromosome.\n" + task.strip() + "\n")
    print(f"header {len(head)} chars (~{len(head)//4} tokens); {len(a.programs)} tasks in {a.out}")

if __name__ == "__main__":
    main()
