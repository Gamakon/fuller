# wgsl_true: the WGSL kingdom on the gene frame's rows

The WGSL kingdom is rows of the gene frame's fixed schema (`docs/DESIGN_geneframe.md` 3.3 and 8.4), held in the one set
of tables `kingdoms/master/` as the kingdom `wgsl` with one variant per interface:

| variant | interface |
|---|---|
| `wgsl.gram` | the Gram problem: `phi` in, `h` out, `cfg { d, m, r_rows, r_off }`; the kernel's epilogue stores `h` at the invocation's own cell |
| `wgsl.image` | a picture into the write-only texture `output` from the uniforms `time_data` and `params` (the cuneus volumepassage's bindings, fuller's tiled image dispatch); `store_px` writes the invocation's own pixel |

Nine data types (eight for the image: no buffer). Every legality rule that the old 305-type table carried in a type is
a property moved by Flow rows and enforced by `must` Guard rows with their repairs inline: `owned` and `local_owned`
(no two invocations write one element), `uniform` (barriers only in uniform control flow), `env.work` (every loop within
the work bound of 64 trips an invocation), `bound` and `len` (every index in range), `width` (vector widths agree).
Scope is context: `let` bindings, loop counters and the entry's builtins and uniforms are variables chosen by `var.T`'s
index word.

## The files

| File | What it is |
|---|---|
| `gen_rows.py` | the generator: writes every variant's rows into `kingdoms/master/` through `kingdoms/master/write_rows.py`; `--check` compares (a Rust test runs it) |
| `gram/shipped.wgsl`, `gram/tiled.wgsl`, `gram/blocked.wgsl` | the Gram problem's reference programs: the shipped kernel (the timing reference) and the two hand-written controls |

The Rust side: `src/wgsl_rows.rs` (load, decode, render, the device rules and their host twin, the share pass),
`src/wgsl/true_kingdom.rs` (`TrueKingdom::from_rows`, the bridge phylu's runners read). The runners are phylu's
`wgsl_true_fit` (Gram) and `wgsl_true_image_fit` (image).
