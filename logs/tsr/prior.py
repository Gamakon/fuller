"""What the UNTYPED engine did on these problems, from the record.

Reads the race ledgers under hff/notebooks/sr_logs/*/race_ledger.json and
reports, per problem, whether a prior untyped race solved it. This is context
for the TSR measurements -- cited as a prior result, never re-run.
"""
import glob
import json
import pathlib

WANT = [
    "feynman_I_15_10",
    "feynman_I_26_2",
    "feynman_I_48_2",
    "feynman_I_15_3t",
    "feynman_II_11_27",
    "feynman_II_24_17",
    "feynman_III_4_32",
    "strogatz_bacres1",
]

rows: dict[str, list[tuple[str, bool]]] = {w: [] for w in WANT}
for path in glob.glob("/Users/andrewmorgan/Dev/gamakon/hff/notebooks/sr_logs/*/race_ledger.json"):
    race = pathlib.Path(path).parent.name
    try:
        d = json.loads(pathlib.Path(path).read_text())
    except Exception:
        continue
    for key, v in d.items():
        name = key.split("|")[0]
        if name not in rows or not isinstance(v, dict):
            continue
        solved = v.get("solution")
        if solved is None:
            solved = v.get("solved")
        if solved is None:
            continue
        rows[name].append((race, bool(solved)))

for name in WANT:
    hits = rows[name]
    if not hits:
        print(f"{name:20} no prior race record found")
        continue
    ok = sum(s for _, s in hits)
    where = ", ".join(f"{r}={'solved' if s else 'no'}" for r, s in sorted(hits))
    print(f"{name:20} solved in {ok} of {len(hits)} prior untyped races   [{where}]")
