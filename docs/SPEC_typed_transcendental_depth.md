# Typed transcendental depth — the symbol table and the design

## Context

**Objective.** Recover SRBench's 133 ground-truth equations exactly. A model with
test R² 1.000000 and the wrong shape scores zero.

**Approach.** GEP with Karva K-expressions on the GPU, selected by HFF, with
`fuller` rewriting genes in place as a variation operator.

**The unique feature this builds on.** nucleotable's master symbol table carries
a *typed, many-hot arity signature* per row, and a kingdom is a query over it.
Multityped GEP is what the schema was designed for. This spec is its first real
use: types that make an illegal gene **unrepresentable** rather than merely
penalised.

**Status.** Specification. Nothing built. The measurement it rests on is done.

**Why.** The engine wastes most of its head space on shapes no law has. The fold
operator was built to detect and collapse them after the fact and costs 23% of
a run's pace. Typing removes the shapes instead of cleaning them up.

## The measurement this rests on

All 133 SRBench true models parsed from `metadata.yaml`; 128 parse to sympy.

| | count |
|---|---|
| **directly nested transcendental pairs** — `f(g(...))` | **0** |
| laws at max transcendental depth 0 | 78 |
| depth 1 | 47 |
| depth 2 | 3 |
| depth ≥ 3 | **0** |

Function totals across all true laws:

```
sqrt 26 · cos 21 · sin 21 · exp 10 · arcsin 2 · tanh 1 · arccos 1
log 0 · tan 0 · Abs 0
```

`log`, `tan` and `abs` **never appear in a true law**, and our reported models
are full of them. That is a separate finding and not what this spec acts on.

The three depth-2 laws reach depth 2 *through arithmetic*, not by direct
nesting — which is why the design below must propagate depth through `+ − × ÷`
rather than resetting it there.

## The types

Extend `geneframe::Ty` with two variants. Existing rows are untouched, which is
the property `Ty`'s docstring already claims.

| type | meaning |
|---|---|
| `F` | a plain float: a variable, a constant, or arithmetic over plain floats |
| `T1` | the output of one transcendental applied to plain floats |
| `T2` | the output of a transcendental applied to something already `T1` |

There is no `T3`. **That absence is the depth rule** — not a check, not a
penalty, an arity signature that does not exist.

## The symbol table

`master_table()` gains rows. Same `semantic_id`, several signatures — exactly
what the schema anticipates: *"a kingdom may give the same semantic id several
aliases."*

### Terminals — unchanged

```
var         in {}        out {F:1}
const       in {}        out {F:1}
rnc  "?"    in {}        out {F:1}
named  pi   in {}        out {F:1}
```

### Arithmetic — ABSORBING

Accepts any mix, yields the highest depth it saw. This is what carries depth
through `+ − × ÷` so that `tanh(x + exp(y))` is correctly `T2`.

```
add   in {F:2}            out {F:1}
add   in {F:1,  T1:1}     out {T1:1}
add   in {T1:2}           out {T1:1}
add   in {F:1,  T2:1}     out {T2:1}
add   in {T1:1, T2:1}     out {T2:1}
add   in {T2:2}           out {T2:1}
```

and the same six for `sub`, `mul`, `div`, `pow`, `protected_div`.

Unary arithmetic — `neg`, `inv`, `pow2`, `pow3`, `protected_inv` — is
depth-preserving:

```
neg   in {F:1}   out {F:1}
neg   in {T1:1}  out {T1:1}
neg   in {T2:1}  out {T2:1}
```

### Transcendentals — DEPTH-RAISING, and they stop at two

```
tanh  in {F:1}   out {T1:1}
tanh  in {T1:1}  out {T2:1}
                                  <- no {T2:1} row. The ceiling.
```

and the same pair for `sin cos tan log exp sqrt abs asin acos` and every
`protected_*` transcendental.

A transcendental has **no row accepting `T2`**, so a third level cannot be
constructed. `tanh(exp(cos(log(x))))` has no legal signature at its third
function.

### What this admits and forbids

| expression | depth | legal |
|---|---|---|
| `m*v**2/2` | 0 | yes |
| `sqrt(1 - v**2/c**2)` | 1 | yes |
| `m_0/sqrt(1 - v**2/c**2)` | 1 | yes |
| `exp(-(theta/sqrt(2))**2)` | 2 | **yes** — the depth-2 laws survive |
| `tanh(exp(cos(log(x))))` | 4 | **no** |
| the measured 26-node blob | 5+ | **no** |

**All 128 parsed laws remain reachable.** Nothing on record is excluded.

## Where it lands in the engine

The change is confined to **what a head slot may draw**. Karva's guarantee is
untouched: the tail holds terminals only, so every head arrangement still
decodes.

> **CORRECTED ON IMPLEMENTATION — point 1 below is wrong as written.**
> `decode_gene` runs on the HOST, in `Engine::evaluate`, long after
> `init_main` has filled the row, and `init_main` fills every position from a
> counter-based draw with no arity walk in it at all. There is no existing walk
> for the sampler to read a demanded type out of.
>
> What is true is the useful half. A position's children are the next unclaimed
> positions, so **a position's parent always sits at a lower index** and its
> budget is known before the position is reached. That makes the walk *state
> carried through the loop the sampler already runs* — a `next_child` pointer
> and one array — rather than a second pass. Implemented as
> `evolve::DepthBudget`, spent top-down, and proved to be the same predicate
> `engine::t_depth` computes bottom-up.
>
> Typing is therefore a **budget spent top-down**, not out-type matching. The
> two agree because both compute the largest transcendental count on any
> root-to-leaf path.

1. **`decode_gene` already walks arity.** The parent's `inputs` says what each
   child slot demands, and that walk happens today — there is no second pass.

2. **`init` and `mut_point` draw by demanded type.** They currently draw
   uniformly from `sample_functions` / `sample_terminals`. Typed, they draw from
   the subset whose `out` matches the slot's demand. `SymbolCodes` gains a
   per-output-type index list; the kernel gains one `u32` per symbol and one
   comparison per draw.

3. **Crossover and transposition** move token spans between genes. A span legal
   in its source can be illegal in its destination. Two options, and the second
   is preferred:
   - reject the move — costs diversity, and silently
   - let it land and let `decode_gene` fail it, as an over-long ORF already
     fails; the row scores `PI` and dies in the next tournament
   The second needs no new mechanism and is the engine's existing behaviour for
   an unfit gene.

4. **`write_back`** must refuse a graft whose types do not close, alongside its
   existing `head_oversize` / `not_closed` refusals, and **count it**.

## Cost

| | |
|---|---|
| kernel | one `u32` per symbol row, one comparison per draw |
| `MAX_NODES` / `HEAP_ARITY` | untouched — arity is still ≤ 2 |
| `geneframe.rs` | already expresses this; currently unused by the engine |
| the fold's 23% pace tax | largely unnecessary once shapes cannot be built |

## Risks, and how each is answered

**The three depth-2 laws.** A strict ceiling at `T1` would make them
unreachable. The `T2` level exists precisely for them. If measurement shows the
ceiling still bites, the fix is a `T3` row set, not a redesign.

**A depth-2 law we have not parsed.** Five of 133 did not parse to sympy. They
could be deeper. Before building, parse them by hand and confirm.

**The search may need illegal shapes as stepping stones.** A blob at depth 4
might be a waypoint to a depth-1 law even though no law is depth 4. This is the
real risk and it is not answerable from the true-model distribution — only from
an A/B. Hence the knob.

**Type errors from crossover reduce effective population.** Measurable: count
rows that fail to decode, which the engine already reports as `oversized`.

> **CORRECTED: it does not.** `oversized` counts a gene that DECODED and then
> had more than `MAX_NODES` nodes. A gene refused on type decodes perfectly
> well and is only too deep — a different failure, and counting them together
> would have made the A/B unreadable. It is counted separately as
> `FitResult::typed_refused`, and `write_back` counts its own refusals as
> `types_do_not_close`.
>
> **And a refused gene is not the same as a dead row.** The spec says "the row
> scores `PI` and dies in the next tournament", which is only true when EVERY
> gene of the row is illegal. The engine runs `n_genes = 3` with
> `gene_subsets` on, so a row with one refused gene scores on its other two and
> the illegal gene rides along — never scored, never removed, still mutating
> and crossing into its neighbours. That is why a typed run's depth histogram
> still shows entries past the ceiling: they are, by construction, the unscored
> refused genes. `oversized` has always behaved this way, so the typed path
> does match the spec's stated preference; it is the spec's picture of what
> that preference does that is wrong.

## The knob

`Config::typed_depth: Option<u32>` — `None` is the engine as it is, `Some(2)`
is this spec. Default `None` until an A/B says otherwise. Every claim above is
about the true-model distribution, not about search behaviour, and those are
different things.

## The tiny test case

Before any engine work, a study that needs no kernel:

1. Take the 128 parsed true models. Assign every node a type under these rules.
   **Assert every law types cleanly.** If one does not, the spec is wrong and
   this is where it is cheapest to find out.
2. Take the reported models from today's runs — the 26-node blob, the 400-char
   `tanh(exp(...))` chains. **Assert they do not type.**
3. Count what fraction of a real population would fail to type, from a recorded
   gene dump. That number is how much of the search space this removes, and it
   is the one number that says whether the idea is worth the build.

Steps 1 and 2 are a script over data already on disk.
