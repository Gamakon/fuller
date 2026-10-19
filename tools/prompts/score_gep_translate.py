#!/usr/bin/env python3
"""Score a model's chromosome (gep_translate format A9) against the reader's tree.
Usage: score_gep_translate.py <reply.txt> <truth.safe.sexpr>"""
import csv, os, re, sys
F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARITY = {}
for r in csv.reader(open(os.path.join(F, "kingdoms/safewgsl/functions.tsv")), delimiter="\t"):
    if r and r[0].startswith("F"):
        ARITY.setdefault(r[1] + "." + r[2], set()).add(int(r[6]))
PROBLEM = re.compile(r"^(load\.(keys|queries|index_u32|index_f32|cands)|store\.(index_u32|index_f32|cands)_[LCO]\d|atomic\.(index_u32|cands)_\w+|load\.cfg_\w+|uniform\.cfg_\w+)$")

def tokens(s): return re.findall(r'\(|\)|"[^"]*"|[^\s()]+', s)
def parse(s):
    t = tokens(s); i = 0
    def node():
        nonlocal i
        if t[i] == "(":
            i += 1; name = t[i]; i += 1; kids = []
            while t[i] != ")": kids.append(node())
            i += 1; return (name, kids)
        i += 1; return (t[i - 1], [])
    n = node(); return n, i == len(t)
def block(reply, tag):
    m = re.search(r"```" + tag + r"\s*\n(.*?)```", reply, re.S); return m.group(1) if m else None
def walk(n, path="root"):
    yield path, n
    for k, c in enumerate(n[1]): yield from walk(c, f"{path}/{k}")
def size(n): return 1 + sum(size(c) for c in n[1])

reply, truth = open(sys.argv[1]).read(), open(sys.argv[2]).read()
body = block(reply, "chromosome"); notes = block(reply, "notes")
if body is None: print("no chromosome block"); sys.exit()
def by_indent(text):
    """The tree the indentation shows: a line is a node (its first symbol, any rest of the line its own
    complete children), a more-indented following line a child of the nearest less-indented line."""
    root = None; stack = []
    for line in text.split("\n"):
        if not line.strip(): continue
        ind = len(line) - len(line.lstrip(" "))
        body_l = line.strip()
        opened = body_l.startswith("(")
        inner = body_l.lstrip("(").rstrip(")")
        toks = tokens("(" + inner + ")") if opened else [inner]
        if opened:
            n, _ = parse("(" + inner + ")" if inner.count("(") == inner.count(")") else "(" + inner.split()[0] + ")")
            if inner.count("(") != inner.count(")"): n = (inner.split()[0], [])
        else:
            n = (inner, [])
        while stack and stack[-1][0] >= ind: stack.pop()
        if stack: stack[-1][1][1].append(n)
        else: root = n
        stack.append((ind, n))
    return root
short = body.count("(") - body.count(")")
intended = by_indent(body) if short else None
if short > 0:
    print(f"unbalanced: {short} closing bracket(s) missing; scored with them added at the end")
    body = body + ")" * short
try: got, whole = parse(body)
except Exception as e: print("parse failed:", e); sys.exit()
want, _ = parse(truth)
def norm(n):
    name, kids = n
    if re.match(r'^-?[0-9.]+(e[-+]?[0-9]+)?$', name):
        name = repr(float(name))
    return (name, [norm(c) for c in kids])
got, want = norm(got), norm(want)
if intended is not None:
    it = norm(intended); im = dict(walk(it))
    from collections import Counter as C2
    def key2(n): return (n[0], tuple(key2(c) for c in n[1]))
    def subs2(n):
        yield key2(n)
        for c in n[1]: yield from subs2(c)
    gs2 = C2(subs2(it)); f2 = 0
    for k in subs2(want):
        if gs2[k] > 0: gs2[k] -= 1; f2 += 1
    same2 = sum(1 for p, n in walk(want) if p in im and im[p][0] == n[0])
    print(f"INTENDED (read by indentation): nodes {size(it)}; same symbol at the same place {same2}/{size(want)} = {100*same2/size(want):.1f} %; truth subtrees found intact {f2}/{size(want)} = {100*f2/size(want):.1f} %")
lit = lambda n: n[0].startswith("literal.")
unknown = wrong_arity = 0; examples = []
def value(x): return re.match(r'^-?[0-9.]+(e[-+]?[0-9]+)?$', x[0]) is not None or x[0].startswith('"')
for p, n in walk(got):
    if value(n): continue
    if n[0] == "untranslatable" or (lit(n) and len(n[1]) == 1 and not n[1][0][1]): continue
    if lit(n) or n[0] in ("fallback",) or n[0].startswith("href"): continue
    if n[0] not in ARITY and not PROBLEM.match(n[0]):
        unknown += 1; examples.append(f"unknown {n[0]} at {p}") if len(examples) < 6 else None
    elif n[0] in ARITY and len(n[1]) not in ARITY[n[0]]:
        wrong_arity += 1; examples.append(f"arity {n[0]} has {len(n[1])}, table {sorted(ARITY[n[0]])} at {p}") if len(examples) < 6 else None
gm = dict(walk(got)); same = sum(1 for p, n in walk(want) if p in gm and gm[p][0] == n[0])
first = next((p for p, n in walk(want) if p not in gm or gm[p][0] != n[0]), None)
print(f"parsed {'whole' if whole else 'PARTIAL'}; nodes {size(got)} (truth {size(want)}); "
      f"unknown names {unknown}; wrong arity {wrong_arity}; same symbol at the same place {same}/{size(want)} = {100*same/size(want):.1f} %; "
      f"identical tree {got == want}; untranslatable {sum(1 for _, n in walk(got) if n[0]=='untranslatable')}")
from collections import Counter
def key(n): return (n[0], tuple(key(c) for c in n[1]))
def subs(n):
    yield key(n)
    for c in n[1]:
        yield from subs(c)
gs = Counter(subs(got)); ws = list(subs(want))
whole_sub = 0
for k in ws:
    if gs[k] > 0: gs[k] -= 1; whole_sub += 1
gb = Counter(n[0] for _, n in walk(got)); wb = Counter(n[0] for _, n in walk(want))
bag = sum((gb & wb).values())
print(f"truth subtrees found intact {whole_sub}/{len(ws)} = {100*whole_sub/len(ws):.1f} %; symbols in common (as a bag) {bag}/{size(want)} = {100*bag/size(want):.1f} %")
if first: print(f"first difference at {first}: truth {dict(walk(want))[first][0]}, got {gm[first][0] if first in gm else '(missing)'}")
for e in examples: print("  " + e)
print("notes:", (notes or "(none)").strip()[:400].replace("\n", " | "))
