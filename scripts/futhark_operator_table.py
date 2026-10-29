#!/usr/bin/env python3
"""The per-operator table of futhark_evolve runs (docs/PLAN_futhark_fitness_fixes.md section I), from each run's own
ledger (OUT/ledger.tsv), with no new evaluation.

For every child (a ledger line with parents), by its operator:
  * unseen output: its output was not seen on any earlier line of the run;
  * better than parent: its fitness is below its best parent's (lower is better);
  * same output as parent: its output equals a parent's;
  * all zeros: its output is all zeros on every case.

The output is the ledger's `behaviour` column (the hash of the outputs on every case) when the ledger has one; a
ledger without it uses its `quality_inputs` column (every quality input, exact), which is one value an output, and
says so. All zeros needs the ledger's `all_zero` column; without it the cell is `n/a`.

  python3 -I scripts/futhark_operator_table.py OUT [OUT ...]      (writes OUT/operator_table.tsv for each run)

The table printed is every run side by side, one line a run and operator.
"""
import csv
import sys
from collections import OrderedDict, defaultdict


def table(out):
    path = f"{out}/ledger.tsv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise SystemExit(f"{path}: an empty ledger")
    key = "behaviour" if "behaviour" in rows[0] else "quality_inputs"
    zero = "all_zero" if "all_zero" in rows[0] else None
    by_id = {}
    seen = set()
    stats = OrderedDict()
    for r in rows:
        b = r[key]
        parents = [] if r["parents"] in ("-", "") else [p for p in r["parents"].split(",") if p]
        if parents and r["operator"] != "elite":
            s = stats.setdefault(r["operator"], defaultdict(int))
            ps = [by_id[p] for p in parents if p in by_id]
            s["children"] += 1
            s["unseen"] += b not in seen
            if ps:
                s["with_parent"] += 1
                s["better"] += float(r["fitness"]) < min(float(p["fitness"]) for p in ps)
                s["same"] += any(b == p[key] for p in ps)
            if zero:
                s["zeros"] += r[zero] in ("1", "yes", "true")
        seen.add(b)
        by_id.setdefault(r["id"], r)
    lines = []
    for op, s in stats.items():
        n, np_ = s["children"], max(s["with_parent"], 1)
        lines.append(OrderedDict([
            ("run", out), ("operator", op), ("children", n),
            ("unseen_output", f"{s['unseen'] / n:.4f}"),
            ("better_than_parent", f"{s['better'] / np_:.4f}"),
            ("same_output_as_parent", f"{s['same'] / np_:.4f}"),
            ("all_zeros", f"{s['zeros'] / n:.4f}" if zero else "n/a"),
            ("output_from", key),
        ]))
    with open(f"{out}/operator_table.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0].keys()) if lines else ["run"], delimiter="\t")
        w.writeheader()
        w.writerows(lines)
    return lines


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    allrows = []
    for out in sys.argv[1:]:
        allrows.extend(table(out))
    if not allrows:
        raise SystemExit("no children in these ledgers")
    cols = list(allrows[0].keys())
    print("\t".join(cols))
    for r in allrows:
        print("\t".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
