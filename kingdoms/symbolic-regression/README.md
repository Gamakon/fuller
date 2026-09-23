# Symbolic Regression — the live kingdom

The op set the engine runs today, and the first kingdom to get a typing system
beyond "everything is a float".

## The measurement the typing rests on

All 133 SRBench ground-truth models, parsed from each dataset's
`metadata.yaml`; 128 parse to sympy.

| | count |
|---|---|
| **directly nested transcendental pairs** `f(g(...))` | **0** |
| laws at max transcendental depth 0 | 78 |
| depth 1 | 47 |
| depth 2 | 3 |
| depth 3 or more | **0** |

Function totals across all true laws:

```
sqrt 26 · cos 21 · sin 21 · exp 10 · arcsin 2 · tanh 1 · arccos 1
log 0 · tan 0 · abs 0
```

`log`, `tan` and `abs` never appear in a true law. Our reported models are full
of them; that is a separate finding and nothing yet acts on it.

The three depth-2 laws reach depth 2 THROUGH ARITHMETIC, not by direct
nesting — which is why arithmetic must propagate depth rather than reset it.

## The typing rules

See `docs/SPEC_typed_transcendental_depth.md` for the full design and
`symbols.md` for the table.

- `F` — a plain float: variable, constant, or arithmetic over plain floats
- `T1` — output of one transcendental over plain floats
- `T2` — output of a transcendental over something already `T1`
- no `T3`, and **that absence is the depth rule**

Arithmetic ABSORBS (yields the highest depth among its inputs). Transcendentals
RAISE, and have no row accepting `T2`.

## Validation, before any engine work

| check | result |
|---|---|
| all 128 parseable true laws type cleanly | **128 OK, 0 rejected** |
| depth distribution under the rules | 78 / 47 / 3 — matches |
| our 8 reported models: do the bloated ones type? | **7 of 8 ILLEGAL** (213-412 chars) |
| the one legal exception | 60 chars, depth 1 — the run that scored test R² 1.000000 |
| share of a live population's sub-expressions that would be illegal | **23%** (400 real fold events) |

The 23% is biased: it samples the fold operator's targets, which are the shapes
the fold already hunts. A uniform gene dump would give a truer figure.

## Status

Specified and validated against data. Implementation and the A/B are in flight.
Measurements land in `measurements/` with their run cards.
