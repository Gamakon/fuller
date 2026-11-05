# `futhark` — the Futhark kingdom

**What it evolves:** typed Futhark programs over multi-dimensional arrays, each problem's kingdom a view of one master
table, every gene a program Futhark accepts.

**The types are data types.** A gene's types are the data types of function arguments and nothing else (`f32`,
`[]f32`, `[][]f32`, `i64`, `size`, tuples, function types). An array's length and an index's bound are
**properties** that travel with the value; which variables a node may use is **scope**, the environment the decoder
builds; where a requirement fails, the rendered program carries an **adapter** (`fit`, `fitrows`, `fitcols`,
`clampix`). The design is `docs/DESIGN_futhark_types_and_properties.md`, the plan `docs/PLAN_futhark_kingdom.md`
section 7. (The earlier shape-class kingdom, master 5, whose types were classes and contexts such as `mat.n.m` and
`a.k:c`, was deleted at plan task D1; its reference programs survive as text in
`tests/fixtures/futhark_m6_references.txt`, its pool programs in `tests/fixtures/futhark_pool_texts/`.)

## The files

| path | what it holds |
|---|---|
| `master/6/` | the master: data types, properties, rows (Argument, Flow, Guard columns), repairs, formulas, preludes, e-graph rules, literal pools, the generated static cost table; format in `master/README.md` |
| `sig/<problem>/m6/view.tsv` | a problem's specification, written by `sniff.py` from the problem's Futhark source |
| `sig/<problem>/m6/tables.tsv` | the view's table of four kinds of row (Argument, Flow, Guard, Entry), written by `view.py` |
| `sig/<problem>/m6/egraph.*` | the view's e-graph: one sort a data type, one constructor a symbol, the rules |
| `sig/<problem>/pool/` | the language-model and benchmark pool of the problem (program text, `*.fut`, and the pool tables) |
| `sig/kahan/kahan.fut` | UltraDim's k-means constructs one by one (Kahan mean, argmin, cell update, kernel boundary), program text |
| `pixel/<shader>/` | the pixel family (below) |

The problems with a committed view are a7, histogram, kmeans, kmeans_i32, pathfinder, reduce and scan
(`SIG6_PROBLEMS` in `gen_tables.py`, `COMMITTED6` in `src/futhark_view/master6.rs`).

## The checks

`src/futhark_view/master6.rs` computes every view at run time and a test requires it to be `view.py`'s table byte for
byte; `src/futhark_view/egraph6.rs` does the same for the e-graph. The renderer (`src/futhark_table/kingdom6.rs`), the
interpreter (`src/futhark_sig/interp6.rs`), the static cost (`src/futhark_cost.rs`), the reader
(`src/futhark_reader.rs`) and the e-graph (`src/futhark_saturation/master6.rs`) read only the view's rows. The proofs:
the 15 reference programs render their stored texts byte for byte; 1,000 random genes a view render programs
`futhark check` accepts; 100 random genes a view compile on the c backend and equal the interpreter; the reader reads
every rendered program back to its tree, and the 842 pool texts with the same values; the e-graph's variants are
today's texts.

## Regenerating

```
python3 kingdoms/futhark/gen_tables.py            # check the committed artefacts (the default, --check)
python3 kingdoms/futhark/gen_tables.py --write    # regenerate them, then check
```

The tables are committed, so the loader's `include_str!` sees them. The default is to **check** them, because the
committed table is the trust boundary; `--write` regenerates into a staging directory, runs the witnesses there and
only then moves the files into place. A test (`the_committed_artefacts_are_what_the_generator_produces`) runs the
check in `cargo test`.

## The pixel family (`pixel/<shader>/`)

A second family from the same generator: a picture as a map over the pixels of
a scalar program, for the cuneus image shaders. The image row is
`map (\yi -> map (\xi -> let px = .. in let py = .. in <block>) (iota cols)) (iota rows)`
over a tile, and a pixel block is a chain of `let`s and bounded `loop`s over
`f32` ending in `[r, g, b]`. Every rule is a type:

| rule | the type that carries it |
|---|---|
| a variable is bound before it is read | `f32.s{k}`: an f32 in scope k; its terminals are `v0 .. v{k-1}` and the interface's inputs |
| `let` binds the next variable | `let@blk.s{k}` takes `f32.s{k}` and a block of scope k+1 |
| a loop binds its state and its counter | `loopS` binds `v{k} .. v{k+S-1}`, its body is `bodyS.d{d+1}.s{k+S+1}` (counter `v{k+S}`), the rest scope k+S |
| a loop is bounded | trip counts are `count` literals (at most 64), nesting depth at most 2 (`PIXEL_DEPTH`) |
| a loop body returns its state | `bodyS.*` ends in an S-tuple |

An instance is declared by its WGSL source: the uniform blocks' f32 fields
become the inputs (`interface.tsv`), and the source's float literals join the
generic constant pool. `volumepassage`: 300 types, 3,651 rows, scopes s0..s40,
loop arities 1, 3 and 5. The e-graph (one sort a dual, constants by their pool
index) holds IEEE identities that are bit-identical: commutativity of `+`, `*`,
`max`, `min`; `a - b = a + (-b)`; `-(-a) = a`; `|-a| = |a|`; `a * 2 = a + a`;
`a * 1 = a`; `a / 2 = a * 0.5`; `a / 4 = a * 0.25`.

The witness tests P1-P9 re-read the emitted text: the root, the declared duals,
the placeholders, every variable's scope, every let and loop binding, the loop
bound, a terminal for every dual, the literal pool against the shader's source,
and the e-graph mapping. The Rust side is `src/futhark_pixel.rs`
(loader, program, work bound and tiles, the hand-written reference, the
e-graph), and the examples `futhark_pixel_reference` and `futhark_pixel_evolve`.
