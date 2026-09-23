"""Append a Note (src/evolve/card.rs :: Note) to a run card.

usage: python3 note.py <card.json> <recovered:true|false|none> <testing> <outcome>
"""
import datetime
import json
import pathlib
import sys

card = pathlib.Path(sys.argv[1])
rec = {"true": True, "false": False, "none": None}[sys.argv[2]]
c = json.loads(card.read_text())
c.setdefault("notes", []).append(
    {
        "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "testing": sys.argv[3],
        "outcome": sys.argv[4],
        "recovered": rec,
    }
)
card.write_text(json.dumps(c, indent=2) + "\n")
print(f"{card}: {len(c['notes'])} note(s)")
