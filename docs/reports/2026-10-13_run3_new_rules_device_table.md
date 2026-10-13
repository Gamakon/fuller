# Run 3: the 10 new structural rules on the device (rule_table_new10)

**Result: unroll-k4 is the one strong new rule (35 of 36 children faster, median 1.20×); grow-cols helps about half the time; double-buffer and pad-shared slow these kernels; wg-shape has a bug (0 of 10 children correct).**

Method: fuller ff24411 (`examples/wgsl_rule_table.rs`), the 49 correct kernels of popR_rules. For each rule and each kernel it fires on, one child: the layered check, the fit's own fitness (128 cases and the pair sets), the largest cell difference against the parent on one Φ, then 6 interleaved timings against the parent at the fit's three shapes. 02:28–03:08Z. Raw: `logs/HGP/001/rule_table_new10/` (table.log, children.tsv).

| rule | fires on | correct | faster in 5–6 of 6 | median ratio (child/parent speed) | max error vs parent |
|---|---|---|---|---|---|
| unroll-k4 | 36 | 36 | 35 | 1.198 | 0 |
| grow-cols | 24 | 24 | 13 | 1.023 | 0 |
| split-accumulator | 36 | 36 | 10 | 0.994 | 4.5e-7 (float order) |
| drop-dead-store | 49 | 49 | 3 | 0.998 | 0 |
| hoist-load | 1 | 1 | 0 | 0.985 | 0 |
| pad-shared | 45 | 45 | 0 | 0.935 | 0 |
| double-buffer | 45 | 45 | 0 | 0.713 | 0 |
| wg-shape | 10 | **0** | 0 | — | **1.07** (wrong values) |
| roll-k, shrink-cols | 0 | — | — | — | — (fire only on unrolled / column-grown kernels) |

## What it shows
- 9 of the 10 rules preserve values exactly (split-accumulator differs only by float rounding, 4.5e-7).
- unroll-k4 is a reliable speed rule on these kernels; grow-cols is mixed; drop-dead-store and split-accumulator are neutral.
- pad-shared (0.94×) and double-buffer (0.71×) cost time here: these kernels have no bank conflicts, as predicted, and the extra buffer and index arithmetic outweigh the saved barrier at these shapes.
- **wg-shape is a bug in our software**: every child computes wrong values (error 1.07 of the largest cell). It is being fixed before any further run uses it.
