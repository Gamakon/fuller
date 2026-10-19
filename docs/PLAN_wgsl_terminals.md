# PLAN: a real terminal for every WGSL dual

## Context

**Objective.** Make the WGSL kingdom pass the multi-typed GEP kingdom check (`src/kingdom_contract.rs`): every type a gene root or a function argument can demand has at least one terminal row, so every tail can be filled with a symbol of its own type and no fallback exists anywhere.

**Approach.** Each of the 59 duals that has no terminal today gets a real, named terminal row in `kingdoms/wgsl/functions.tsv`, written as real WGSL text. Nothing is substituted at decode time. A terminal is an ordinary symbol; selection ranks it like any other.

**Unique feature.** A tail cell of type T holds a terminal of T, so a dual with no terminal is an unfillable tail. The check reports it as "our software has a bug: type X is demanded and has no terminal". It is the check doing its job.

**Status.** Proposal only, nothing added. Measured 2026-10-19 on `WgslKingdom::load()` (83 duals, 1143 symbols, 1105 functions, 38 terminals): 24 duals have a terminal, 59 do not, and that is the only defect the check finds (ordered signatures and one output per row are fine).

**Why.** Without these rows the WGSL kingdom cannot host a gene (CR section 9 item 3).

## What demands a terminal

The check demands one for every type some function takes as an argument, and for every gene root. My `wgsl_relations()` currently lists all 83 duals as roots, which is wider than real use. Counting by demand:

| Group | Duals | Demanded as an argument | Demanded only as a root |
|---|---|---|---|
| A. value duals with a literal domain | 22 | yes | |
| B. value duals with no literal domain | 32 | yes | |
| C. `grid.grid`, `block.block` | 2 | yes | |
| D. `array<T>`, `store.store`, `kernel.kernel` | 3 | no (consumed by 0 rows) | yes |

Total 59.

## A. Value duals with a literal domain (22): a `?_T` placeholder

Rule: a constant placeholder `?_T` with a Dc (design section 4), values in the typed `C_T` drawn inside the dual's `literal` domain from `types.tsv`. Row name `literal.<dual>`, so the kingdom's existing `literal.*` rule marks it a placeholder.

| Duals | Count |
|---|---|
| T010-T012 `vec2/3/4<i32>.int` | 3 |
| T018, T020 `vec2<u32>.index`, `vec4<u32>.index` | 2 |
| T022, T024 `vec2<u32>.count`, `vec4<u32>.count` | 2 |
| T030-T032 `vec2/3/4<u32>.bits` | 3 |
| T035-T037 `vec2/3/4<bool>.flag` | 3 |
| T061-T069 `mat2x2` ... `mat4x4<f32>.real` | 9 |

Emitted text is the dual's constructor over `C_T` values, for example `vec2<i32>(3, -1)`, `mat2x2<f32>(...)`.

## B. Value duals with no literal domain (32): needs a ruling

`types.tsv` gives these no `literal` domain on purpose: they are forms whose value should come from a computation (a hash, a quantised code, an opaque value). A `?_T` needs a domain to draw from, and a register is zero-initialised, which is a constant 0 under another name. Two options:

1. **Add a literal domain** to each (data change in `types.tsv`), then treat as group A. Domains are definable for most: `u32.code4` 0..=3, `code8` 0..=7, `code16` 0..=15, `code1024` 0..=1023, `bool.sign` {false, true}, `i32.q15_16` and its vectors any i32, `vec2<u32>.hash64` and `u32.hash32` any u32.
2. **A named register per dual**, like the existing `reg<f32>.real` rows (T076 on, `F217` "declared at the top of the kernel and initialised to its form's fallback"). That precedent exists but its initialiser is the thing the plan removes, so a register here would be `var r: T;` with WGSL's own zero value stated in the row, not a fallback table.

Recommendation: option 1 for the code, sign and q15_16 duals (their domains are exact), and ask Andrew for opaque and hash duals, where "no literal" may be the intent.

| Duals | Count |
|---|---|
| T013-T016 `i32.q15_16` and vec2/3/4 | 4 |
| T025-T028 `u32.code4/8/16/1024` | 4 |
| T033 `u32.hash32` | 1 |
| T038 `bool.sign` | 1 |
| T039-T058 `f32/f16/i32/u32/bool .opaque`, scalar and vec2/3/4 | 20 |
| T059 `vec2<i32>.q15_16w` | 1 |
| T060 `vec2<u32>.hash64` | 1 |

## C. Statement-level duals (2)

| Dual | Proposed terminal | Text |
|---|---|---|
| T074 `block.block` | `stmt.empty_L0` | `{}` |
| T073 `grid.grid` | `grid.unit` | `@workgroup_size(1, 1, 1)` |

`{}` is a real WGSL compound statement, and so is a legal block of zero statements. The renderer's current "fallback is 16 x 16 at 1 x 1" grid is a fallback and is not carried over: `grid.unit` is an ordinary row, selected or not. "L0" follows the lead's naming; I have not found what a level means in `kingdoms/wgsl`, so please confirm it.

## D. Root-only duals (3)

None of these is an argument of any row today.

| Dual | Role | Proposal |
|---|---|---|
| T071 `store.store` | the statement a head gene's root produces | `store.store` is a different dual from `block.block`, so it needs its own row `store.empty`, text `{}` |
| T072 `kernel.kernel` | the entry point, a gene's root | `kernel.empty`: `@compute @workgroup_size(1, 1, 1) fn main() {}` |
| T070 `array<T>.<form of T>` | a buffer of T, "one row per buffer at read-in" | no row exists in the fixed vocabulary: it is a per-kernel row (`load.buffer`) added by the reader. A named binding terminal belongs there. Proposal: do not make it a root; it is never demanded by a function |

Decision for the lead: real genes are rooted at the kernel dual (and `store`/`block` when a statement is evolved). If `wgsl_relations()` roots are narrowed to those, group D drops out of the check, and only A, B and C (56 duals) need rows. That is the smaller change and I recommend it. I will not narrow the roots without your word, since the CR says a gene's root is its first token's output.

## Order of work if approved

1. Narrow or keep the roots (decision above).
2. Group A rows and group C rows (24 duals), re-run the check, close the 59 to 32 or fewer.
3. Group B per ruling.
4. Group D per ruling.
5. Remove the covered-subset helper in `src/typed_gep_tests.rs` once the full kingdom passes, so the closure runs on all 83 duals.

Each step is a fuller commit with the check result, tail widths at h=32 and 100,000-gene closure in its message.
