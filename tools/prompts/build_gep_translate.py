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

def example(program, chromosome):
    return ("Program (WGSL):\n\n```wgsl\n" + open(program).read().strip() + "\n```\n\nIts chromosome:\n\n```chromosome\n"
            + pretty(open(chromosome).read().strip()) + "\n```")

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
