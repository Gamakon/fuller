# Tracked kernels the safe reader does not read (T43, 2026-10-21)

`examples/wgsl_reader_census` reads every tracked `.wgsl` kernel (the corpora, gen0, the safe kernels, the seeds, the saved-genome
texts, the problems' references) and every gzipped one under `tests/fixtures/wgsl_texts` with `read_safe`. The test
`every_tracked_kernel_the_reader_refuses_is_listed` (src/wgsl/reader_census.rs) asserts that the refused set is EXACTLY the one
below: a kernel that starts to read, or one that stops, fails it until this file says so. The same kernel appears in several
directories (the 11 `library_ep` texts are copies of `safewgsl` replies), one row each.

## Expected refusals (48): not valid WGSL, or a kernel wrong for its own header

Classes: A a parse error (model prose, truncation, stray braces); B an identifier the kernel never declares; C a WGSL reserved word
used as a name; D naga rejects it; F1 the body computes one row a thread, the `// grid 16 8 4 8` header dispatches for eight.

| directory | kernel | class | why |
|---|---|---|---|
| tests/corpus/safewgsl | draft_matrices.wgsl | D | naga rejects it (negation of an unsigned value) |
| tests/corpus/safewgsl | reply_fitA2_g001_r10p1_fable.wgsl | B | names an identifier it never declares |
| tests/corpus/safewgsl | reply_fitA2_g001_r18p1_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitA2_g002_r10x_union_1625Z_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitA2_g002_r16p1_1536Z_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitA2_g002_r18p1_1536Z_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitA2_g002_r4p1_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitA2_g002_r5p1_codex.wgsl | F1 | its body computes one row a thread, its `// grid 16 8 4 8` header dispatches for eight: it computes the wrong cells under its own header |
| tests/corpus/safewgsl | reply_fitA2_g003_r2x_union_1702Z_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_fitrepro2_g001_r13p1_sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_g001_r2x_union_fable.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/safewgsl | reply_g001_r8x_union_fable.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0010.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0348.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0541.wgsl | B | names an identifier it never declares |
| tests/corpus/model_replies | k0548.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0565.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0568.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0649.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0658.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0706.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0737.wgsl | F1 | its body computes one row a thread, its `// grid 16 8 4 8` header dispatches for eight: it computes the wrong cells under its own header |
| tests/corpus/model_replies | k0805.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0821.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k0828.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k1145.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k1149.wgsl | B | names an identifier it never declares |
| tests/corpus/model_replies | k1179.wgsl | B | names an identifier it never declares |
| tests/corpus/model_replies | k1220.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k1252.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k1387.wgsl | B | names an identifier it never declares |
| tests/corpus/model_replies | k1644.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/corpus/model_replies | k1719.wgsl | C | uses a WGSL reserved word as a name |
| tests/corpus/model_replies | k1723.wgsl | C | uses a WGSL reserved word as a name |
| tests/corpus/model_replies | k1725.wgsl | C | uses a WGSL reserved word as a name |
| tests/corpus/model_replies | k1726.wgsl | C | uses a WGSL reserved word as a name |
| tests/corpus/model_replies | k1748.wgsl | C | uses a WGSL reserved word as a name |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1536Z__g001_r10p1.fable.wgsl | B | names an identifier it never declares |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1536Z__g001_r18p1.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1536Z__g002_r16p1.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1536Z__g002_r18p1.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1625Z__g002_r10x_union.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1625Z__g002_r4p1.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1702Z__g003_r2x_union.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_A2_stopped_1729Z__g002_r5p1.codex.wgsl | F1 | its body computes one row a thread, its `// grid 16 8 4 8` header dispatches for eight: it computes the wrong cells under its own header |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_repro2__g001_r13p1.sonnet.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_smoke6__g001_r2x_union.fable.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |
| tests/fixtures/wgsl_genomes/library_ep/texts | fit_smoke6__g001_r8x_union.fable.wgsl | A | not WGSL: a parse error (model prose, truncation, stray braces) |

## Open reader gaps (10): valid WGSL the any-WGSL gate does not read yet

Classes: E a barrier under a condition (needs a uniform-conditional rung); F3 a row held in a variable assigned in a branch
(ownership of a select of two row forms); K a loop step or nesting depth outside the kingdom; J a struct built from its members.
Closed in this task: F (lidx axes, two axes in one array), I (vec4<u32> lane stores), H (a loop entered at a uniform expression),
G (a stale let held in a register).

| directory | kernel | class | why |
|---|---|---|---|
| tests/corpus/model_replies | k0245.wgsl | F3 | a row that is a variable assigned in a branch (reader gap: ownership of a select of two row forms) |
| tests/corpus/model_replies | k1777.wgsl | J | a struct built from members (reader gap: struct constructor and member stores) |
| tests/corpus/model_replies | k1788.wgsl | K | a loop step or nesting depth outside the kingdom (reader gap) |
| tests/corpus/cuneus | radix_sort.wgsl | K | a loop step or nesting depth outside the kingdom (reader gap) |
| tests/fixtures/wgsl_texts/v24_run1_raw | seed28.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |
| tests/fixtures/wgsl_texts/v24_run1_raw | seed29.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |
| tests/fixtures/wgsl_texts/v24_run1_raw | seed31.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |
| tests/fixtures/wgsl_texts/v24_run1_seeds | seed28.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |
| tests/fixtures/wgsl_texts/v24_run1_seeds | seed29.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |
| tests/fixtures/wgsl_texts/v24_run1_seeds | seed31.wgsl | E | a barrier under a condition (reader gap: a uniform-conditional rung) |

## Outside the kingdom (A0, 2026-10-21, 6): read before, refused now because the only spelling was an escape row

The rows `uflag.of_flag` and `uniform.of_checked` admitted any flag or index into a uniform context "when the checker proves
it"; they are deleted (`kingdoms/safewgsl/gen_tables.py`), so a test over a barrier that is not a comparison of workgroup-uniform
values, and a loop over a barrier whose bound is not a uniform value, have no spelling. The reader refuses them with the reason
"outside the kingdom: ... no row admits a flag into a uniform context". The same shape is the open gap E above
(seed28, seed29, seed31, a barrier under a condition); these six differ only in that the test is a register or a thread's value.
A uniform register, set by `workgroupUniformLoad`, is the type that would spell them (the D1 construction `Val<T, WgUniform>`).

| directory | kernel | class | why |
|---|---|---|---|
| tests/corpus/model_replies | k1482.wgsl | O | a barrier under a test of registers and the workgroup id (uniform only if the registers are): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
| tests/corpus/model_replies | k1797.wgsl | O | a barrier under a test of the thread's local index (not uniform, so illegal under its contract): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
| tests/fixtures/wgsl_texts/v24_run1_raw | seed06.wgsl | O | a barrier under `reg < 512u` (a register): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
| tests/fixtures/wgsl_texts/v24_run1_raw | seed21.wgsl | O | a barrier under `reg > 0u` (a register from a workgroupUniformLoad): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
| tests/fixtures/wgsl_texts/v24_run1_seeds | seed06.wgsl | O | a barrier under `reg < 512u` (a register): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
| tests/fixtures/wgsl_texts/v24_run1_seeds | seed21.wgsl | O | a barrier under `reg > 0u` (a register from a workgroupUniformLoad): no row admits the flag into a uniform context (`uflag.of_flag` deleted) |
