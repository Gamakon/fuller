# fuller

**An e-graph engine for making symbolic expressions smaller — provably without changing what they compute.**

Named after Buckminster Fuller's *ephemeralization*: doing more with less. `fuller`
takes a symbolic expression (a GEP chromosome, a SymPy expression, a Brainfuck
program) into an [egglog](https://github.com/egraphs-good/egglog) e-graph,
saturates a bounded ruleset, and extracts the smallest equivalent form — with a
data-gated or exact equivalence check so a rewrite can never silently change
behaviour.

It is a Rust crate exposed to Python via [PyO3](https://github.com/PyO3/pyo3).
It was written to replace SymPy in the inner loop of a symbolic-regression
engine (the [HFF](https://github.com/Gamakon/HFF) equation-recovery work),
where SymPy's complex-domain assumptions, native signal handling, and
exponential simplification paths made expression cleanup unreliable. egglog is
the right substrate: deterministic, bounded, declarative, and fast (the same
engine family behind [Herbie](https://herbie.uwplse.org/)).

## The vision

**Machine-discovered expressions should be reported in their simplest form.**
Physical laws are compact; expressions produced by evolutionary search are
not. They arrive with constants absorbed into coefficients (`√(NA/r⁴)`
concealing a physical constant), redundant structure, and noise terms that fit
the sample rather than the underlying relationship. Much of the gap between an
expression that fits the data and the law itself is this excess structure.
fuller reduces it — Buckminster Fuller's ephemeralization, *doing more with
less*, applied to equations.

**Representation should be evolvable.** A constant can appear in a genome two
ways: as a symbol (`pi`, `G`, `k_e`) or as a number (`3.14159…`, `6.674e-11`).
The `snap` ⇄ `concretize` pair makes that choice a reversible mutation, so a
population carries both representations and selection determines which
survives. When the symbolic form wins, the discovered expression matches the
reference form directly rather than being recovered by post-hoc pattern
matching.

**Rewriting belongs inside the evolutionary loop, not after it.** Classical GP
treats simplification as post-processing, because unsound rewrites inside the
loop would corrupt the population. fuller's rewrites are proved equivalent (or
data-gated to preserve behaviour), which removes that constraint: denoise,
snap, and physics-prior restructuring can act as genetic operators, and a
simplification, once found, is propagated by crossover — a Lamarckian
mechanism made sound by proof of equivalence.

**Any grammar with a cost model and an equivalence check can be minimised.**
Symbolic regression checks equivalence against data; Brainfuck checks it
exactly, output-by-output. The engine is agnostic to the choice, and the two
included targets demonstrate one of each. The same machinery applies to SQL,
regex, sorting networks, or compiler IR — any setting where a smaller,
provably equivalent program is useful.

## How it works — the round trip

A karva gene leaves the GEP population, is rewritten in an e-graph under
proved-equivalence rules, evaluated on data, and returns to the population as
a valid gene:

```mermaid
flowchart TD
    G["Karva gene<br/>(GEP population)"] -->|"expressed via ORF"| X["EXPRESS codec<br/>pset name → semantic id"]
    X --> A["AST<br/>(Math · Brainfuck · …)"]
    A -->|"saturate bounded ruleset"| E[("e-graph<br/>equivalence classes")]
    E -->|"extract k candidates"| C["Candidate forms<br/>(provably equivalent)"]
    C -->|"compile + run on data"| M["Per-candidate metrics<br/>train · val · size"]
    M -->|"HFF TrueNorth angle"| T{{"tournament"}}
    T -->|winner| W["Winning form"]
    W -->|"R² guard:<br/>behaviour unchanged?"| Y["COMPRESS codec<br/>semantic id → pset name"]
    Y --> R["Rebuild karva<br/>(ORF pads fixed geometry)"]
    R -->|"selection & crossover"| G

    classDef gep fill:#e8f0fe,stroke:#4a6fa5,color:#1a2b4a
    classDef egraph fill:#efe8fe,stroke:#7a5fa5,color:#2b1a4a
    classDef data fill:#e8fef0,stroke:#4aa56f,color:#1a4a2b
    classDef codec fill:#fef3e0,stroke:#c08a3e,color:#4a331a
    class G,R gep
    class A,E,C egraph
    class M,T,W data
    class X,Y codec
```

**Blue** is the GEP domain (geppy genes), **purple** the e-graph domain (every
rewrite provably equivalent), **green** the data domain (candidate behaviour
measured on held data). Step by step:

| # | Stage | Operation | What happens / why it works |
|---|---|---|---|
| 1 | **Karva gene** (fixed head+tail token string) | expressed via the ORF | GEP decoding reads tokens until the expression tree closes — the open reading frame. Everything past the ORF is non-coding and silent. |
| 2 | Math AST | saturate bounded ruleset | The expressed tree is lifted into an egglog e-graph and a capped rewrite schedule runs — deterministic and bounded. |
| 3 | E-graph (equivalence classes) | extract k candidates | The e-graph holds every equivalent form compactly in shared classes; only k variants are materialised. Each is provably equivalent by construction. |
| 4 | Candidate forms | compile + run on data | Each candidate is compiled and evaluated on train/val rows; behaviour is measured directly. |
| 5 | Per-candidate metrics (train · val · size) | HFF TrueNorth angle | Each candidate's metric vector becomes one angular fitness score, using the same scoring geometry as the host GA. |
| 6 | Scores | tournament | Candidates compete; the instrumented e-class tournament selects the winner on measured behaviour rather than the cost model alone. |
| 7 | Winning form | R² guard | The winner is retained only if behaviour on the data is unchanged within tolerance; a rewrite cannot silently change what the gene computes. |
| 8 | Rebuilt karva tokens | re-encode into the fixed head/tail geometry | **This step is only sound in karva, because of the ORF.** A simplified form is usually *shorter* than the gene it replaces — in tree-GP that is a differently-shaped genome and breaks the genetic operators. In karva the shorter coding region ends earlier; the fixed-length head and tail are padded with terminals *past the ORF*, which are non-coding by definition, so the padding cannot change the expressed tree. The rebuilt gene is a structural drop-in of identical length. |
| 9 | **Karva gene** (back in the population) | selection & crossover | The simplification is now heritable: crossover propagates it and selection judges it. Rewriting operates inside the loop rather than as post-processing. |

The open reading frame serves two purposes here: it allows any equivalent
form — shorter, restructured, constant-snapped — to re-enter a fixed-geometry
genome legally, and it guarantees the padding is inert (non-coding by
definition). Without it, step 8 would require variable-length genomes, and
rewriting would revert to a post-processing step.

The same cycle applies to **Brainfuck** programs. The middle of the loop
(steps 2–7) is target-agnostic — swap the grammar and the cost model — with
one difference: because a Brainfuck program's behaviour is a decidable
input→output function, the equivalence guard at step 7 is exact rather than
statistical. `bf_simplify` implements that path, and any target with decidable
equivalence (SQL, regex, compiler IR) fits the same cycle.

### The symbol-table contract (the two codecs)

The soundness of the round trip depends on the two codec steps (orange nodes).
fuller does not hardcode a symbol table — the host declares one, and the two
codecs enforce it in both directions:

**EXPRESS** (gene → AST): the host supplies each karva call with a mapping
from its own token names to fuller's semantic ids,

```python
functions = {                     # token_name -> (semantic_id, arity)
    "protected_div_zero": ("protected_div", 2),
    "_pset_square":       ("pow2", 1),
    "_raw_div":           ("div", 2),
    ...
}
```

so the e-graph reasons over semantic ids (`div`, `pow2`), independent of the
host's naming. Distinct semantics remain distinct: a protected division and a
raw division map to different ids, so no rule can substitute one for the
other.

**COMPRESS** (winning form → gene): every candidate fuller emits is rendered
using only token names from that same mapping, so decoding back into the
host's pset always resolves. Any form the host's symbol table cannot express
is reported in the `inexpressible` field of the result rather than
approximated. The host's decode tables may also be a superset of what its GA
samples (decode-only registration), which keeps raw operators available to
rewrites without entering random gene generation.

See [`docs/USAGE.md`](docs/USAGE.md) for the full contract.

## What it does

- **Denoise** — shrink an expression to an equivalent smaller one, gated by an
  R² check against your data so the rewrite provably preserves behaviour on the
  points you care about.
- **Snap ⇄ concretize** — flip named constants (`pi`, `G`, `k_e`, …) to their
  numeric values and back. The pair works as GEP mutation operators, letting a
  population evolve the *representation* (symbolic vs numeric) under selection.
- **Prove equivalence** — `equals` / `proves_equal` decide whether two
  expressions are the same via equality saturation, not sampling.
- **SymPy bridge** — `to_math` / `from_math` convert SymPy ⇄ the internal `Math`
  form losslessly (real-domain ops), so `fuller` can be used within an existing
  SymPy pipeline.
- **Brainfuck simplifier** — the same machinery applied to a second, exact
  target (`bf_simplify`), where equivalence is decidable output-by-output.

## Install

Requires a Rust toolchain (`rustup install stable`) and Python ≥ 3.9. Built with
[maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release      # builds the Rust extension and installs `fuller`
```

## Quick start

```python
import sympy as sp
import fuller

r = sp.Symbol("r")

# SymPy <-> internal Math form
fuller.to_math(sp.pi * r**2)          # '(Mul (Var "pi") (Pow2 (Var "r")))'

# Prove two forms equal (equality saturation, not sampling)
fuller.equals(sp.pi * r**2, sp.pi * r * r)     # True

# Denoise a Math expression against data: shrink it while preserving R².
rows = [{"x": v} for v in (-2.0, 1.0, 3.0, 5.0)]
fuller.denoise('(Add (Mul (Var "x") (Num 1.0)) (Mul (Num 0.0) (Var "y")))', rows)
# -> {'expr': '(Var "x")', 'cost': 1, 'changed': True}

# Same engine, a different target: simplify a Brainfuck program.
fuller.bf_simplify("+++[-]")     # {'source': '...', 'op_count': ..., 'changed': ...}
```

For the GEP/karva-chromosome API (`denoise_karva`, `snap_karva`,
`concretize_karva`, `physics_mutate_karva`, `eclass_extract_hff_instrumented`),
see [`docs/USAGE.md`](docs/USAGE.md).

## Two proven targets: symbolic regression and Brainfuck

fuller is not tied to one language. The core is *target-agnostic*: give it a
grammar, a cost model, and a way to check equivalence, and it will minimise
programs in that grammar. Two targets are included, chosen because they check
equivalence in the two fundamentally different ways available:

**Symbolic regression (equivalence checked against data).** Evolved GEP
expressions are noisy and bloated — `x·1 + 0·y`, `√(NA/r⁴)·…` where a physical
constant is absorbed into a coefficient, `Abs(a^1.5)` where the domain is
positive. fuller folds these to the reduced form (`x`, `1/r²·√NA`, `a^1.5`) and
keeps the rewrite **only if R² does not drop on your data**. This is the job
it was written for: replacing SymPy in the extraction step of a
symbolic-regression engine, cheaply and deterministically, so the discovered
expression is reported in its simplest equivalent form. The `snap_karva` / `concretize_karva`
pair goes further — letting a population evolve *whether a constant is symbolic
or numeric*, so selection recovers the reference form of a law in a single run.

**Brainfuck (equivalence checked exactly).** A Brainfuck program's behaviour is
a decidable input→output function, so equivalence is *exact* — no data, no
tolerance. `bf_simplify` collapses run-length redundancy (`+++++-----` → nothing,
`[-]` clear-loops) while proving the simplified program produces identical output
on every input. It demonstrates the same engine in the setting where behaviour
preservation is decided exactly rather than statistically — and the same
technique applies to any target with decidable equivalence (SQL, regex, sorting
networks, compiler IR).

## Watching a fit — `hff-watch`

A fit writes a **telemetry stream**: one JSON object per line, versioned and
numeric, at the beat the progress report already runs on. `hff-watch` repaints
from it. The engine's prose log is unchanged and is still what an offline
diagnosis reads; the stream exists because an operator cannot read a scrolling
report of a 450-second search.

```bash
cargo build --release --features gpu --bin hff-watch

# a live fit, or a live cascade of many, as it runs
./target/release/hff-watch --follow <your-run>/stream.jsonl

# a finished run: name the directory, not the file inside it
./target/release/hff-watch --file <your-run>

# the same state as text, for a pipe, a log, or a second watcher
./target/release/hff-watch --dump <your-run>
```

`<your-run>` is wherever the fit was told to write (`EVOLVE_TELEMETRY_FILE`).
Run output is not versioned — see [`experiments/README.md`](experiments/README.md)
for the full set of environment variables and where a run puts things.

![hff-watch, the main view](docs/img/hff-watch-main.png)

The verdict is the SEARCH's answer, in capitals, first on the line: this fit
ran its full 180 s and **did not** reach a law, which is a different reading
from the stream being finished — the badge beside it says that. Below it the
two islands, the cohort table (`c2800` has gone EXTINCT and keeps its row until
its history ages out), the discoveries the fit made as it ran, and the model on
one line at the foot.

Note `log10 p -10.30` in red: above the stop bar of −19, so the p half was
never met. The viewer colours it against the bar the ENGINE sent, not one of
its own, so a run that moves the bar is still read correctly.

### The model pane, and the hall of fame

A fit writes a model record every time it betters itself, so the stream holds
the whole ladder. `m` opens it and `↑`/`↓` walk it — the pane redraws for the
selected entry:

![the model pane with the hall of fame](docs/img/hff-watch-model-hall-of-fame.png)

Fifty-three entries here, falling from 2.508e-2 at generation 1139 to 1.079e-2
at 3423 — the search's own account of how it got where it stopped, which is
what tells you whether it converged steadily or jumped. The model beside it is
indented on its own brackets; as one line it runs several hundred characters
off the edge.

### The same state as text

`--dump` is the same state machine as the screen, printed instead of drawn, so
a dump and a view can never disagree. It needs no terminal, which is how a
script and an operator watch the same fit.

**A law that was found:**

```
LAW FOUND  (early_stop · the fit met BOTH halves of its stop bar)
FINISHED  strogatz_lv2.tsv  seed 7014  gen 177  6/60 s
  best hff 0.000000e0   1-R2 train 3.564e-14   val 3.675e-14   log10 p —
  intake-0     rows    2000  best 0.000000e0
  champion-0   rows    2000  best 1.132101e-1

  cohort    born      rows      best hff     best ever        gain
       0       0      2400    0.000000e0    0.000000e0         NEW
     100     100      1600   1.923106e-1   1.923106e-1         NEW

  DISCOVERIES · 0 snapped · 82 folded · 0 dropped
  gen    177  fold  3 nodes ((-88.0)/44.0) -> -2.000000e0
  gen    160  fold  4 nodes tanh((1/log(Abs((-17.0))))) -> 3.389946e-1 ×26
  gen     20  fold  4 nodes cos((x_0 - x_0)) -> 1.000000e0 ×5
```

**One that was not** — the verdict is the SEARCH's answer, not the stream's
state, so a finished fit that missed says so in capitals:

```
LAW UNFOUND  (the fit ended with the bar unmet)
FINISHED  feynman_III_15_27.tsv  seed 7014  gen 371  60/60 s
  best hff 0.000000e0   1-R2 train 3.551e-9   val 1.153e-9   log10 p —
```

Note `best hff 0.000000` beside a non-zero `1-R²`: the f32 angle has saturated,
which is why the viewer prints both and never reports the angle alone.

### Keys

| key | |
|---|---|
| `m` | the model pane: the discovered expression, indented on its own brackets |
| `↑` `↓` | walk the **hall of fame** — every model the fit bettered itself with; the pane redraws for the selected one |
| `Tab` | protected form · plain form · the engine's raw `Math` |
| `h` `l` | scroll the printed model · `w` gives back the raw single line · `y` writes it to a file |
| `Tab` (main view) | focus an island · `Enter` a cohort's detail · `/` filter · `g` global cohorts |
| `Space` | pause the redraw — records keep arriving, and the fit is never touched |

The model pane matters more than it sounds: a recovered model is one line of
several hundred characters, and wrapped it is a wall while scrolled sideways it
is a slot. Indenting on its own brackets is what makes the *shape* readable —
which factor multiplies which sum, how deep the nesting goes. Terms pack up to
the pane width rather than taking a line each.

### Watching a cascade of many fits

A sweep runs one fit per law, a minute each. With a file per law the viewer is
pointed at a stream that dies sixty seconds later, so a cascade sets
`HFF_TELEMETRY_APPEND=1` and every law writes its own `run_start` into **one**
file. The viewer follows by byte offset and reads a fresh `run_start` as a new
run, so it rolls from law to law on its own — one path, live from the first law
to the last.

```bash
HFF_TELEMETRY_APPEND=1 EVOLVE_TELEMETRY_FILE=$PWD/logs/RUN/stream.jsonl \
  ./target/release/examples/evolve_fit data.tsv
```

Truncating stays the default: a single fit owning its file is right for a single
fit, and a sweep silently appending to an earlier sweep's stream would be worse
than either.

## Python API

| Function | Purpose |
|---|---|
| `denoise(expr, rows, tolerance=, k_variants=, positive_vars=, nonzero_vars=)` | Shrink a `Math` string, R²-gated against `rows`. |
| `denoise_karva(head, tail, …)` | Same, as a GEP mutation operator on karva head/tail token lists. |
| `snap_karva(head, tail, …)` | Up-flip: replace fitted numbers with named lattice constants. |
| `concretize_karva(head, tail)` | Down-flip: replace named constants with their numeric values. |
| `equals(a, b)` / `proves_equal(a, b, …)` | Decide equivalence by equality saturation. |
| `to_math(expr)` / `from_math(s)` | SymPy ⇄ `Math` s-expression bridge. |
| `physics_mutate_karva(…)` | Physics-prior structural mutations (inverse-square, etc.). |
| `eclass_extract_hff_instrumented(…)` | Tournament extraction with per-variant metrics. |
| `bf_simplify(source)` / `bf_parse` / `bf_unparse` | Brainfuck program simplification. |
| `master_constants()` / `master_lattice()` / `master_pset()` | The constant lattice and primitive set. |

## Why an e-graph instead of SymPy

SymPy rewrites in place, over the complex domain, with simplification paths that
can blow up on adversarial input and native code that installs its own signal
handlers. On a hot evolutionary loop that is both slow and unreliable. An
e-graph represents *all* equivalent forms at once, applies a **bounded**
ruleset (saturation is capped), extracts by an explicit cost
model, and is fully deterministic — the same input always yields the same
output. Equivalence is *proved* by saturation, not sampled.

## Design guarantees

- **Bounded.** Every saturation runs a capped rule schedule; no unbounded work.
- **Behaviour-preserving.** Rewrites are either exact (algebraic identities) or
  data-gated (kept only if R² does not drop) — a rewrite cannot silently change
  what an expression computes.
- **Deterministic.** No wall-clock, no randomness outside an explicit `rng_seed`.
- **Crash-safe at the boundary.** PyO3 entry points return `Err`, never abort:
  deeply nested / oversized inputs are rejected rather than overflowing the
  stack.

## Status

The core is complete and in regular use inside the
[HFF](https://github.com/Gamakon/HFF) symbolic-regression engine. Denoising,
the snap/concretize pair, the physics-prior mutations, the SymPy bridge and
the Brainfuck simplifier are all implemented and tested. The rewrite ruleset
is deliberately small and grows slowly: a rule is added only when it is either
provably sound or guarded by the data check. The consumer guide is
[`docs/USAGE.md`](docs/USAGE.md), with per-feature notes elsewhere in `docs/`.

## Citation

fuller is the rewrite substrate of the [HFF](https://github.com/Gamakon/HFF)
symbolic-regression engine. If you use it in published research, please cite
the GECCO 2026 poster:

```bibtex
@inproceedings{morgan2026hff,
  author    = {Andrew James Morgan},
  title     = {Hyperspherical Fitness Functions for Many-Objective Optimization},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation
               Conference Companion (GECCO Companion '26)},
  series    = {GECCO Companion '26},
  year      = {2026},
  month     = jul,
  location  = {San Jose, Costa Rica},
  publisher = {ACM},
  address   = {New York, NY, USA},
  isbn      = {979-8-4007-2488-6/2026/07},
  doi       = {10.1145/3795101.3805445}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

---

Built by [Gamakon](https://gamakon.ai). Support, integration help, and
collaboration enquiries are welcome — reach out via [gamakon.ai](https://gamakon.ai).
