# fuller — egglog-based bidirectional AST hub

## Mission

Build a Rust crate, exposed via PyO3, that uses **egglog** as a substrate for
symbolic expression rewriting. The crate is consumed by sibling project
`hff/` (Hyperspherical Fitness Functions for symbolic regression) but is
designed to be domain-agnostic: GEP karva chromosomes today, SQL trees,
sympy expressions, and LLM-emitted strings tomorrow — all become inputs
to one e-graph.

The strategic prize: today five subsystems (HFF library, GEP engine, snap
rewriters, rule-discovery loop, planned LLM-emits-karva work) each have
their own AST handling, their own glue code, their own representation.
egglog collapses them into one declarative substrate where karva mutations,
sympy rewrites, snap rules, dimensional analysis, and HFF-as-cost-function
all become rules and queries over one e-graph.

Phase 1 (this brief): prove the bidirectional round-trip end-to-end with
one usable feature — a **denoise mutation operator** that takes a GEP
chromosome, projects it into an e-graph, rewrites away noise, projects
back as a karva chromosome. This is the skateboard. It must work on real
chromosomes from `hff/` and produce simpler chromosomes that the GA can
inject into its population.

Phase 2 (separate, later brief): port mathematical identities from
SymPy's simplification modules into egglog rules. The skateboard's
ruleset grows over time without changing the consumer-side API.

## Status of `hff/` (what's already proven there)

The sibling project `/Users/andrewmorgan/Dev/kaito/hff/` has working code
demonstrating the round-trip is feasible — built today (2026-05-25) in
sympy, but sympy has proven slow, fragile, and exhausting to maintain.
We are explicitly **abandoning sympy** for this work. Read these files
in `hff/notebooks/` for the patterns and the lessons:

- `_gene_decompose.py` — Celko nested-set decomposition of a GEP gene
  head; FK placeholders for parent-context preservation. Worth reading
  closely; the algorithm transfers directly to egglog.
- `_sympy_to_karva.py` — the "visit" function: AST → karva token list,
  including BFS serialisation. Will be ported but should not depend on
  sympy in the new crate.
- `hff_sr_engine.py` (search for `_extract_best`) — the consumer-side
  call chain we replaced multiple times today, hitting sympy bug after
  sympy bug. The new crate is the durable replacement.
- `_chromosome_ensemble.py` — generated equivalent forms; the JSONL log
  at `/tmp/equivalent_forms.jsonl` (if present) contains real worked
  examples of chromosome → simplified expression transitions.

**Do not import any of this into fuller.** Read for patterns and
counter-examples, then build clean.

## Out of scope

- No GA, no fitness evaluation, no HFF math. Those live in `hff/`.
- No sympy dependency. (We are deliberately leaving it behind.)
- No SQL frontend in Phase 1 (mentioned in mission for future direction).
- No LLM integration in Phase 1.
- No snap-library port in Phase 1 (Phase 2 task).
- No automatic rule mining from SymPy in Phase 1 (Phase 2 task).

## Phase 1 deliverable — denoise mutation operator

End-to-end demonstration that proves the architecture works on one
real example: a GEP karva chromosome enters, an equivalent but
smaller karva chromosome exits, and the rewrite happened via egglog
saturation + extraction.

### The skateboard

```
karva chromosome  →  egglog terms  →  saturate(denoise rules)
                                              ↓
                                       extract_smallest_with_data_parity
                                              ↓
                                       egglog term  →  karva chromosome
```

The denoised chromosome:
- Decodes to a valid GEP gene structure (head + tail, GEP arity rule
  satisfied, tail padded with random terminals where necessary)
- Evaluates numerically to within `tolerance` (default 1e-3 relative)
  of the original on the supplied training data
- Has strictly fewer head tokens than the original (or is returned
  unchanged if no denoising opportunity exists)

### Karva conventions you must encode

A GEP chromosome consists of `n_genes` genes, each a flat token list.
Each gene splits into:

- **head**: length `h`, may contain functions (arity ≥ 1) or terminals
  (arity 0). Functions decoded level-order with BFS consume terminals
  from the tail.
- **tail**: length `t = h * (max_arity - 1) + 1`. Terminals only — never
  functions. This is a hard GEP invariant; the converter must enforce it.

A "live" expression is determined by walking the head left-to-right
and consuming child slots as function nodes are seen. Tokens beyond
the live region (in either head or tail) are **neutral region** —
they don't appear in the decoded AST but get carried along during
mutation/crossover.

After denoise, when a smaller head is emitted, the tail must be
re-padded with random terminals from the pset to satisfy
`t = h * (max_arity - 1) + 1`. The non-coding tail rule guarantees
every random padding produces a valid GEP gene — no compile failures.

### Denoise rules (initial set, ~5-10)

Rules to encode in egglog for the skateboard:

1. **Constant folding** — subtrees with no free symbols evaluate to a
   number, replace with that number. (`cos(G) → 0.999...` etc.)
2. **Multiplicative identity** — `(x * 1) → x`, `(1 * x) → x`.
3. **Additive identity** — `(x + 0) → x`, `(0 + x) → x`.
4. **Multiplicative zero** — `(x * 0) → 0`, `(0 * x) → 0`.
5. **Same-op nest collapse** — `Abs(Abs(x)) → Abs(x)`, `(-(-x)) → x`,
   `sqrt(x**2) → Abs(x)`.
6. **Drop data-irrelevant terms** — for each top-level term in `Add`
   or factor in `Mul`, evaluate the expression with that term removed
   on training data. If `R² loss < tolerance`, accept the smaller form.
   (This is the data-aware rule; egglog can extract multiple equivalent
   forms and the harness scores them externally — egglog cost functions
   are pure structural.)

Rules 1-5 are pure algebraic and run inside saturation. Rule 6 is
implemented as a post-saturation extraction loop: extract the smallest
N candidates, score each on data, return the smallest that preserves
R² to within tolerance.

### Consumer API (the Python signatures `hff/` will call)

```python
# In Python, after `pip install -e .` of this crate:
from fuller import denoise, karva_to_terms, terms_to_karva

def denoise(
    head: list[Any],          # list of geppy Function/Terminal tokens
    tail: list[Any],          # list of geppy Terminal tokens
    pset: PsetSpec,           # see PsetSpec below — pure data, no geppy import
    train_X: np.ndarray,      # (n_rows, n_vars)
    train_y: np.ndarray,      # (n_rows,)
    tolerance: float = 1e-3,  # max relative R² loss
    rng_seed: int = 0,        # tail re-padding determinism
) -> tuple[list, list]:
    """Return (new_head, new_tail). Returns the original tokens unchanged
    if no denoising opportunity exists or if all candidates fail tolerance.
    Never raises on un-encodable expressions — returns original.
    """

class PsetSpec:
    """Pure-data description of the pset, no geppy dependency.
    Constructed once per fit by hff/ and passed in on every denoise call.
    """
    variables: list[str]                  # ["x0", "x1", ...]
    functions: list[FunctionSpec]         # name + arity + semantic_id
    rnc_values: list[float]               # numeric constants in Dc array

class FunctionSpec:
    name: str          # e.g. "add", "mul", "sin", "_diff_sq", "protected_sqrt"
    arity: int
    semantic_id: str   # one of {"add", "sub", "mul", "div", "neg", "sin", "cos",
                       #         "log", "exp", "sqrt", "abs", "tanh", "pow2",
                       #         "pow3", "inv", "diff_sq"} — what the function
                       # ACTUALLY computes, regardless of geppy name
```

The `semantic_id` is critical: the same operator can have different
geppy names across psets (`protected_sqrt` vs `math.sqrt`). fuller
sees only the semantic ID; the consumer is responsible for the name
mapping when constructing `PsetSpec`.

### Tail-padding contract

When the denoised head is shorter than the input, the new tail must
be padded with random terminals to `len(new_head) * (max_arity - 1) + 1`.

- Source of randomness: `random.Random(rng_seed)` — caller controls seed
  for determinism / reproducibility.
- Pool: `pset.variables + [str(v) for v in pset.rnc_values if v is not None]`
- The padded tail must be deterministic given `(new_head, rng_seed, pset)`.

### Non-goals for the skateboard

- No trigonometric simplification (sin² + cos² = 1, etc.). Phase 2.
- No power simplification (x^a * x^b = x^(a+b)). Phase 2.
- No logarithm rules. Phase 2.
- No matrix expressions, no integration, no limits.
- No "guess what simpler chromosome would have evolved" — strictly
  algebra-driven rewriting.

## Architecture

The crate uses **maturin** as the build backend so `hff/` (and any other
consumer) can install via:

```
pip install -e /Users/andrewmorgan/Dev/kaito/fuller
```

That command must (a) build the Rust extension via maturin, (b) install
the `python/fuller/` shim package, and (c) leave a working
`from fuller import denoise, karva_to_terms, terms_to_karva` import
in the target Python environment.

```
fuller/
├── Cargo.toml
├── pyproject.toml            ← maturin backend, PyO3 deps
├── BRIEF.md                  ← this file
├── README.md                 ← short summary, install instructions
├── src/
│   ├── lib.rs                ← PyO3 entry + module surface
│   ├── pset.rs               ← PsetSpec / FunctionSpec structs
│   ├── karva.rs              ← karva token list ↔ egglog term
│   ├── ruleset/
│   │   ├── mod.rs            ← ruleset registry
│   │   ├── identities.rs     ← Rules 1-5 above (pure algebra)
│   │   └── data_aware.rs     ← Rule 6 (extraction-time loop)
│   ├── extract.rs            ← extract-many-and-rank
│   └── eval.rs               ← lambdify-equivalent: evaluate egglog term
│                              on numpy data (real-domain only, no complex)
├── python/
│   └── fuller/
│       ├── __init__.py       ← re-exports the public Rust functions
│       └── _typing.py        ← Python-side dataclass mirrors of PsetSpec
├── examples/
│   ├── 01_roundtrip.py       ← karva → terms → karva, parity check
│   ├── 02_denoise_demo.py    ← real chromosome denoising, side-by-side
│   └── 03_synthetic_noise.py ← hand-built "noisy" chromosomes (x+0, x*1
│                              wallpaper) demonstrate rule firing
├── tests/
│   ├── test_roundtrip.rs     ← 100 random head+tail combos, parity
│   ├── test_rules.rs         ← each rule fires on its target pattern
│   ├── test_denoise.rs       ← end-to-end on synthetic + real cases
│   └── test_pyo3.py          ← Python-side smoke
└── reports/
    └── phase1_report.md      ← acceptance report (template below)
```

## Phase ordering

Do not interleave. Finish one phase, ship a commit, then start the next.

### Phase 1.0 — Calibration

Before touching karva, prove you can drive egglog from Rust end-to-end:

1. Read the egglog Rust crate's README + tutorial. Note the API
   version pinned. Record in `reports/environment.md`.
2. Build a trivial ruleset (boolean algebra: identity, double negation,
   De Morgan ×2, absorption — 5 rules). 20 hand-written test cases.
3. Round-trip Rust → egglog → extract → Rust on the 20 cases. All
   pass. Write `examples/00_calibration.rs`.
4. PyO3-expose ONE function and call it from Python. Confirm the
   FFI boundary works on this machine (macOS arm64, Python 3.12).

**Stop and report if calibration fails.** The egglog API has drifted
from what this brief assumes; the fix is upstream.

### Phase 1.1 — karva ↔ terms converter

Pure structural; no rules involved.

1. Define `PsetSpec` and `FunctionSpec` (`src/pset.rs`).
2. `karva_to_terms(head, tail, pset) → egglog term`. Walks head BFS
   per the GEP rule; functions become egglog constructors; terminals
   become typed leaves; RNC indices become Number nodes.
3. `terms_to_karva(term, pset, rng_seed) → (head, tail)`. Inverse:
   BFS the term, emit head tokens until last function, emit tail
   terminals, re-pad to GEP rule with rng_seed.
4. `tests/test_roundtrip.rs`: generate 100 random valid karva chromosomes
   for several psets; round-trip; assert byte-equal head + tail (modulo
   tail padding randomness).

### Phase 1.2 — Rules 1-5 (pure algebra)

1. Define the algebraic ruleset (`src/ruleset/identities.rs`).
2. Each rule: a written math identity in a comment, then egglog syntax.
3. Per-rule unit test: build a term containing the LHS pattern,
   saturate with only that rule, extract, assert the RHS appears.
4. All-rules-together unit test: build a synthetic chromosome
   `add(mul(x, 1), mul(0, y))` → `x` after saturation + extract.

### Phase 1.3 — Real-valued evaluator (no sympy, no complex)

1. `src/eval.rs`: given an egglog term and `(X, y)` arrays, evaluate
   the term row-by-row in pure Rust. Use `ndarray` or `rayon`-friendly
   numpy interop. Return `(predictions, n_failures, mean_err)`.
2. Operators evaluate in the real domain only: `sqrt(negative) → NaN`,
   `log(non-positive) → NaN`, `div_by_zero → NaN`. Caller decides what
   to do with NaN; the evaluator does not "protect" via Abs wrapping.
3. Test: evaluate known terms on known data, compare against hand-computed.

### Phase 1.4 — Rule 6 (data-aware extraction)

1. `src/extract.rs`: after saturation, extract the smallest K candidates
   from the e-graph by structural cost (egglog provides this).
2. For each candidate: evaluate on `(train_X, train_y)`, compute R²
   loss vs the original input expression.
3. Return the smallest candidate whose loss < `tolerance`. If none
   qualify, return the original.

### Phase 1.5 — Public API + Python wrapping

1. PyO3-export `denoise`, `karva_to_terms`, `terms_to_karva`.
2. `python/fuller/__init__.py` re-exports.
3. `python/fuller/_typing.py` defines the Python-side `PsetSpec`
   dataclass — pure dataclass, no geppy import.
4. `examples/02_denoise_demo.py`: build a known noisy chromosome,
   call `denoise(...)`, print before / after side-by-side.

### Phase 1.6 — Acceptance & report

1. Run `examples/03_synthetic_noise.py` on 20 hand-built wallpaper
   chromosomes. Every one must shrink AND preserve R²>0.99 vs original.
2. Run `examples/02_denoise_demo.py` on 5 chromosomes drawn from the
   hff equivalent-forms log (`/tmp/equivalent_forms.jsonl` if present;
   else regenerate via hff). Report shrinkage and R² loss per case.
3. Write `reports/phase1_report.md` — template below.

## Hard constraints

- **Rust edition 2021. egglog Rust crate, pinned version, recorded in `reports/environment.md`.**
- **No sympy. Anywhere. Not in tests, not in examples, not in docs.**
  We are explicitly leaving sympy behind. If a test needs to compute
  an expected value, do it by hand or in pure Rust/numpy. (`numpy`
  is allowed in examples for evaluation only.)
- **PyO3 0.22+** for Python bindings. Match `hff/`'s setup pattern.
- **Saturation budget per call**: 1s wall clock, 10,000 e-graph nodes.
  Hard cap. If a rule causes saturation to exceed this, reject the rule
  with the failure logged.
- **No bare commutativity or associativity rules.** egglog handles
  these via e-class merging; encoding them as rewrites is the most
  common beginner mistake and causes blowup.
- **Pattern variables on the RHS must appear on the LHS.** Otherwise
  saturation diverges. Reject at compile time.
- **Conditional rewrites require explicit guards** (e.g. `x/x → 1`
  needs `when (!= x 0)`).
- **Determinism**: same input + same rng_seed = bit-identical output.
  Tests assert this.

## What to do if egglog Rust API has changed

If you find that egglog's Rust API differs significantly from this brief's
assumptions:
1. Document the actual API in `reports/environment.md`.
2. Adjust the implementation to match, but keep the public Python signatures
   unchanged.
3. If a critical feature is missing (e.g. extraction by external cost
   function), STOP and report. Do not invent a workaround.

## Reporting template (`reports/phase1_report.md`)

```
# fuller Phase 1 acceptance report

## Environment
- Rust:     <version>
- egglog:   <crate version>
- PyO3:     <version>
- Platform: <macOS / Linux>, Python <version>

## Calibration
- Boolean algebra ruleset:     PASS / FAIL
- PyO3 FFI smoke:              PASS / FAIL

## Round-trip parity
- 100 random chromosomes, byte-equal after karva → terms → karva:
  - Pass: <N>/100
  - Fail cases logged at: <path>

## Rule firing (pure algebra)
- Rule 1 (constant folding):     <PASS / FAIL>
- Rule 2 (mul identity):         <PASS / FAIL>
- Rule 3 (add identity):         <PASS / FAIL>
- Rule 4 (mul zero):             <PASS / FAIL>
- Rule 5 (same-op nest):         <PASS / FAIL>

## Data-aware denoise
- 20 synthetic noise cases:
  - Shrink rate:  <N>/20
  - R² preserved (>0.99): <N>/20
  - Median tokens removed: <count>
- 5 real chromosomes from hff equivalent-forms log:
  - <chrom_id>: orig=<N>tok new=<N>tok R²_loss=<f>
  ...

## Timing
- Mean denoise wall time per chromosome: <ms>
- P95 saturation node count:             <N>

## Open issues / next phase recommendations
- ...
```

## What you do not do

- Do not import sympy. We are leaving it.
- Do not depend on geppy. The `PsetSpec` is the boundary.
- Do not commit `examples/03_synthetic_noise.py` outputs as test
  fixtures — regenerate per run.
- Do not push. Leave commits local for the user to review.
- Do not modify `hff/`. The two projects are siblings, not parents.

## Git workflow

- Initialise as a fresh git repo (`git init`) in `/Users/andrewmorgan/Dev/kaito/fuller/`.
- Commit as you go, one commit per phase (1.0, 1.1, etc.), descriptive
  messages following conventional-commits style.
- Do NOT push.
- Final tree should be clean (`git status` empty) when phase 1 ships.

## Acceptance criteria (overall)

Phase 1 is done when:
- Calibration passes.
- Round-trip parity ≥ 99/100 random chromosomes.
- All 5 pure-algebra rules fire on their target patterns.
- 20/20 synthetic noise cases shrink with R²>0.99.
- ≥3/5 real chromosomes from hff demonstrate non-trivial shrinkage.
- `reports/phase1_report.md` written and committed.

Quality is judged on the report plus the artefacts. No prose updates
required during the work — proceed silently and report at the end.

## Phase 2 preview (not in scope for this brief)

After Phase 1 ships, Phase 2 will port mathematical identities from
SymPy's simplification modules (`powsimp.py`, `radsimp.py`,
`trigsimp.py`, `ratsimp.py`) into the fuller ruleset. The user
has a separate brief for that work. The Phase 1 architecture must
not block it — specifically, the ruleset registry in
`src/ruleset/mod.rs` should support adding new modules without
touching `src/lib.rs` or the public API.
