# TSR — Typed Symbolic Regression

**Andrew Morgan's name for it**, and the name carries the claim:

> "It's the only symbolic regression to properly handle the distance needed
> between the functions."

Everyone writes SR for symbolic regression. This is **TSR**.

## What is different

Ordinary SR gives every function the same type. A float goes in, a float comes
out, so `exp` will accept `tanh`'s output as readily as it accepts `x`, and the
search is free to build `tanh(exp(cos(log(x))))`. Nothing in the representation
says that is a worse shape than `m*v**2/2` — only the fitness does, after the
fact, by which point the head space is already spent.

TSR limits the **distance** between functions. The depth a term sits at (how
many transcendentals are stacked on a path below it) and its Horton-Strahler
stream order are **properties** of the one data type `F`, derived bottom-up by
the gene frame's Flow rows, and a `must` Guard row says where a shape is too
deep or too wide. Where a guard fails, the decoder renders the node as its
first input: `sin(sin(sin x))` becomes `sin(sin x)`.

Illegal shapes stop being penalised and become **inexpressible**: they stay
representable in the gene (GEP's silent material) and never reach the program.

## The rows (the kingdom `tsr.<variant>` in `kingdoms/master/`, the depth variant as `sr`, written by `gen_tables.py`)

| kind of row | what TSR puts there |
|---|---|
| DataType | `F`, carrying `depth`, `order`, `fam`, `arm`, `ties`; its constant `?` is a leaf (depth 0, order 1) |
| Argument | the 18 operators of phylu's `SymbolTable::wide` (`F` in and out), the input terminals `x0..x9`, the constant `?` |
| Flow (up) | `add`: `depth := max(in0.depth, in1.depth)`; a raising unary (`sin cos tan tanh psqrt plog pexp abs pasin pacos`): `depth := in0.depth + 1`; a level-keeping unary (`neg pow2 pow3 pinv`): `depth := in0.depth`; stream order over flattened sum and product chains through `fam`, `arm`, `ties` |
| Guard (`must`, repair `{in0}`) | `depth`: `in0.depth <= 1` on the ten raising unaries; `order`: `out.order <= 3` on the four binaries; `windows`: `in0.order <= 2` on `sin cos tan tanh pow2 pow3 pinv` |
| Entry | `tsr`, output `F` |

Every variant has the same Argument and Flow rows; only the Guard rows differ:

| variant | guards | arm |
|---|---|---|
| `single` | none | A |
| `depth` | the depth guard: **the symbolic regression kingdom** `sr` (plan 7.7, K1; `master::select("sr")`) | E |
| `order` | the order ceiling | B |
| `order_depth` | both | C |
| `order_depth_windows` | both and the windows | F |

The gene is one type, 104 words at head 34 (tail 35, Dc 35), no Dv lane: the
inputs are terminals, not variables. A law of n columns keeps the first n input
terminals (`TsrRows::load`).

## Why depth two, and not one or three

Measured across all 133 SRBench ground-truth models (128 parse):

| | count |
|---|---|
| **directly nested transcendental pairs** | **0** |
| laws at depth 0 | 78 |
| depth 1 | 47 |
| depth 2 | 3 |
| depth 3 or more | **0** |

A limit at depth 1 would exclude three real laws. A limit at 3 would admit
shapes no law uses. **Two is where the data puts it.** The three depth-2 laws
get there *through arithmetic*, never by direct nesting — which is why
arithmetic takes the greatest input depth rather than resetting it.

## What was deleted (2026-11-05, plan 7.7 task K1)

The first build carried the depth limit as **types**: a ladder `F`, `T1`, `T2`
(`geneframe::Ty`, `typed_depth_table`, `EVOLVE_TYPED_DEPTH=2` in phylu), and
the TSR arms as **duals** `shape.level` (`types.tsv`, `functions.tsv`, a `ROOT`
twin of every function row, up to 34 types and 20,980 words a gene). Both are
gone: the depth rule is the `depth` Guard row, the order rule the `order` Guard
row, and a gene is 104 words whichever limits are on. Programs worth keeping
were kept as text (phylu's champion pool); no old gene is read. The history of
the typed build is in `docs/SPEC_typed_transcendental_depth.md` and
`docs/PLAN_tsr_typed_gep.md`.

## Where it runs

phylu decodes the same rows on the device (`typed_decode.wgsl`, the rule words
of `tsr_props.rs` compiled by `TsrRows::device_rules`) and on the host
(`tsr_props::program`, the one host decoder). `EVOLVE_SR_PROPS=<variant>`
chooses the variant; the law runner's default is `depth`.

## Tests

`src/tsr_kingdom/rows/tests.rs`: the Flow rows give phylu's `stream_order` and
`t_depth` on 46,279 exhaustive and 100,000 random trees; 10,000 random genes a
variant decode to legal programs and nothing illegal; the K1 smoke test (1,000
genes on `depth`: legal, and the host decode equals the device rules' decode);
`gen_tables.py --check` (the master rows of every variant equal a fresh
generation) is a test.
