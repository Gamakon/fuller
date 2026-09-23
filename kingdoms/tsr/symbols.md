# TSR — the symbol table

The human-readable statement of what `geneframe::typed_depth_table()` builds.
The table is the design; the Rust is its loader, and
`the_typed_kingdom_leaves_the_untyped_rows_alone` holds the two together.

Kingdom name: `Symbolic Regression (typed depth)` (`geneframe::TYPED_SR`).

## The type ladder

| type | depth | meaning |
|---|---|---|
| `F` | 0 | a plain float: a variable, a constant, or arithmetic over plain floats |
| `T1` | 1 | the output of one transcendental over plain floats |
| `T2` | 2 | the output of a transcendental over something already `T1` |

**There is no `T3`.** `Ty::at_depth(3)` is `None`, and that absence is the
depth rule — not a check, not a penalty, a rung that does not exist.

## The rows

Same 26 `Math` semantic ids as the `symbolic-regression` kingdom, several
signatures apiece. Multiple rows per `semantic_id` is what the schema
anticipates: *"a kingdom may give the same semantic id several aliases."*

### Terminals — one row, `out {F}`

A variable or a constant is depth 0. In the engine these are `Symbol::Input`,
`Symbol::Rnc` and the withheld `Symbol::Named`; none of them raises depth.

### Arithmetic — ABSORBING

Any mix in, the **highest depth seen** out. This is what carries depth through
`+ − × ÷`, and it is why the three depth-2 SRBench laws — which reach depth 2
*through arithmetic*, never by direct nesting — stay legal.

```
add   in {F:2}            out {F:1}
add   in {F:1,  T1:1}     out {T1:1}
add   in {T1:2}           out {T1:1}
add   in {F:1,  T2:1}     out {T2:1}
add   in {T1:1, T2:1}     out {T2:1}
add   in {T2:2}           out {T2:1}
```

Six rows each for the binary ops: `add sub mul div pow protected_div`.

Unary arithmetic is depth-**preserving**, three rows each — `neg inv pow2 pow3
protected_inv`:

```
neg   in {F:1}   out {F:1}
neg   in {T1:1}  out {T1:1}
neg   in {T2:1}  out {T2:1}
```

### Transcendentals — DEPTH-RAISING, and exactly two rows

```
tanh  in {F:1}   out {T1:1}
tanh  in {T1:1}  out {T2:1}
                              <- NO {T2:1} row. The ceiling.
```

Two rows, not three, is the whole design. The fifteen that raise
(`geneframe::DEPTH_RAISING`):

```
sin  cos  tan  log  exp  sqrt  abs  tanh  asin  acos
protected_sqrt  protected_log  protected_exp  protected_asin  protected_acos
```

`pow2` and `pow3` are whole powers and do **not** raise. Raw `pow` is absorbing
here: `SymbolTable::wide` carries no raw `Pow`, so the sampler never has to
decide, and `t_depth` counts a `Pow` only when its exponent is not whole —
a property of the node, not of the op.

## How the engine spends it

The engine does not match output types slot by slot. It spends a **depth
budget top-down** (`evolve::DepthBudget`), which is the same predicate: both
compute the largest transcendental count on any root-to-leaf path, which is
exactly what `engine::t_depth` measures bottom-up.
`the_typed_table_is_the_predicate_t_depth_already_computes` is the test that
holds them together, and it asserts the spec's own admitted and forbidden
expressions.

Per symbol the engine carries one `u32` — `SymbolCodes::depth_cost`, 1 for a
raising function and 0 for everything else — and one extra draw list,
`sample_flat`, the functions costing 0. A slot with budget draws from all the
functions; a slot without draws from `sample_flat`. That is the spec's "one
`u32` per symbol row, one comparison per draw".

A **compound** costs what its expansion costs: `SqrtSum` is `sqrt|a + b|`, one
root on the path, so it spends one level like the `sqrt` it expands to.

## What is typed, and what is left to the decode

| operator | typed? |
|---|---|
| `init` | **yes** — the budget is spent as the head is filled |
| point mutation | **yes** — same budget, same walk |
| inversion, IS, RIS, gene transposition, crossover | **no, on purpose** |

The span-moving operators move a subtree whole, and a span legal in its source
can be illegal in its destination. The spec prefers letting that land and
letting the decode fail it over refusing the move, because refusing costs
diversity silently. `Engine::evaluate` refuses the gene and counts it as
`typed_refused`, apart from `oversized`; `write_back` refuses a graft that
would break the ceiling as `types_do_not_close`.

**A refused gene is not a dead row.** At `n_genes = 3` with `gene_subsets` on,
a row with one refused gene scores on its other two, so the illegal gene rides
along unscored. That is why a TSR run's depth histogram still shows entries
past the ceiling — those entries *are* the refused genes.
