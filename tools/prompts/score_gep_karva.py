#!/usr/bin/env python3
"""Decode a model's Karva gene (prompt v3, format A9) by the table's arities, level by level,
and score it against the reader's tree. Usage: score_gep_karva.py <reply.txt> <truth.safe.sexpr> <storage buffer names, comma-separated>"""
import csv, os, re, sys
F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUFFERS = set()
ARITY = {}
for r in csv.reader(open(os.path.join(F, "kingdoms/safewgsl/functions.tsv")), delimiter="\t"):
    if r and r[0].startswith("F"):
        ARITY.setdefault(r[1] + "." + r[2], set()).add(int(r[6]))
def arity(sym):
    """The table's arity; the problem's own rows by their template: load.<buffer> 1, a store 2,
    a uniform field 0; a literal and `untranslatable` 0. None for a name the table lacks."""
    base = sym.split("=")[0]
    if base.startswith("literal.") or base in ("untranslatable", "fallback") or base.startswith("href"): return 0
    if base in ARITY: return max(ARITY[base]) if len(ARITY[base]) > 1 else next(iter(ARITY[base]))
    if re.match(r"^(load|uniform)\.cfg_\w+$", base) or re.match(r"^uniform\.\w+$", base): return 0
    if re.match(r"^store\.\w+_[LCO]\d$", base): return 2
    if re.match(r"^load\.\w+$", base): return 1 if base[5:] in BUFFERS else 0
    return None
def gene_of(reply):
    # Forgiving of format: text before the gene, the tag on the fence line or the next.
    m = re.search(r"```\s*karva\s*\n(.*?)```", reply, re.S) or re.search(r"```\s*\n\s*karva\s*\n(.*?)```", reply, re.S)
    if not m: return None, None
    syms, orf_mark = [], None
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line: continue
        if line.upper().startswith("ORF END"): orf_mark = len(syms); continue
        parts = line.split(None, 1)
        syms.append(parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else line)
    return syms, orf_mark
def decode(syms):
    """Level order: the root, then each read symbol's children from the next unread positions."""
    unknown = []
    nodes = [[s, []] for s in syms]
    if not nodes: return None, 0, unknown
    queue = [0]; nxt = 1; qi = 0; short = 0
    while qi < len(queue):
        i = queue[qi]; qi += 1
        a = arity(nodes[i][0])
        if a is None: unknown.append((i, nodes[i][0])); a = 0
        for _ in range(a):
            if nxt >= len(nodes): short += 1; nodes[i][1].append(["(missing)", []]); continue
            nodes[i][1].append(nodes[nxt]); queue.append(nxt); nxt += 1
    return nodes[0], nxt, unknown, short
def truth_tree(sexpr):
    toks = re.findall(r'\(|\)|[^\s()]+', sexpr); i = 0
    def node():
        nonlocal i
        if toks[i] == "(":
            i += 1; name = toks[i]; i += 1; kids = []
            while toks[i] != ")": kids.append(node())
            i += 1
            if name.startswith("literal.") and len(kids) == 1 and not kids[0][1]:
                return [f"{name}={float(kids[0][0])!r}", []]
            return [name, kids]
        i += 1; return [toks[i - 1], []]
    return node()
def normlit(n):
    name = n[0]
    if "=" in name and name.startswith("literal."):
        f, v = name.split("=", 1)
        try: name = f"{f}={float(v)!r}"
        except ValueError: pass
    return [name, [normlit(c) for c in n[1]]]
def walk(n, path="root"):
    yield path, n
    for k, c in enumerate(n[1]): yield from walk(c, f"{path}/{k}")
def size(n): return 1 + sum(size(c) for c in n[1])
if __name__ == "__main__":
    reply, truth = open(sys.argv[1]).read(), open(sys.argv[2]).read()
    BUFFERS.update(sys.argv[3].split(","))
    syms, mark = gene_of(reply)
    if syms is None: print("no karva block"); sys.exit()
    got, used, unknown, short = decode(syms); got = normlit(got); want = truth_tree(truth)
    gm = dict(walk(got)); same = sum(1 for p, n in walk(want) if p in gm and gm[p][0] == n[0])
    first = next((p for p, n in walk(want) if p not in gm or gm[p][0] != n[0]), None)
    print(f"gene {len(syms)} symbols; decoded ORF {used} (marked ORF END at {mark}); tail read for {max(0, used - (mark or used))} child(ren); missing children {short}; "
          f"unknown names {len(unknown)}; nodes {size(got)} (truth {size(want)}); same symbol at the same place {same}/{size(want)} = {100*same/size(want):.1f} %; identical {got == want}")
    if first: print(f"first difference at {first}: truth {dict(walk(want))[first][0]}, got {gm[first][0] if first in gm else '(missing)'}")
    for i, u in unknown[:6]: print(f"  unknown {u} at gene position {i}")
