# Run 6e ledger analysis: why the typed WGSL run did not improve

**Conclusion first:** Within run 6e, no interface improved after generation 0. Every change in the best score happened in the seed evaluation at generation 0. Of the legal children, 92 to 100 % score exactly the do-nothing score (0.58569), and for interfaces 0, 3 and 5 most children are rejected by the legality layer, mostly for workgroup memory races.

The "improvements" in the request (0.19447 to 0.19551, 0.29538 to 0.28755, 0.33925 to 0.33771) are comparisons with run 6d, not with evolution inside run 6e. Those three starting values are the generation 0 bests of run 6d (`logs/V24/run6d_halfroot.log`). Run 6e's own generation 0 bests differ: interface 3 is worse (0.19551 against 0.19447), and interfaces 0 and 5 are better. The seeds did the work, not the operators.

## Source

- Ledger: `logs/V24/run6e_halfroot/typed_ledger.tsv`, 25,348 evaluated rows (19,529 feasible, 5,819 illegal), 201 generations (0 to 200).
- Population bests: the `# typed gen` lines in `logs/V24/run6e_halfroot.log`.
- The ledger has no parent column. Neutrality is therefore measured against the do-nothing score (0.58568555, shown as 0.58569), not against each child's parent.
- The ledger holds children only. An elite that is carried over is not logged again, so child bests can be worse than population bests. Table 4 gives both.

## Table 1: distinct fitness values among feasible children

| Interface | Evaluated | Feasible | Distinct feasible values | Top value (do-nothing 0.58569) | 2nd value (count) | 3rd value (count) |
|---|---|---|---|---|---|---|
| 0 | 4,022 | 1,421 | 58 | 96.0 % (1,364) | 0.33771 (1) | 0.63656 (1) |
| 1 | 3,218 | 3,196 | 3 | 99.9 % (3,194) | 0.63005 (1) | 0.58719 (1) |
| 2 | 3,218 | 2,955 | 1 | 100 % (2,955) | none | none |
| 3 | 1,610 | 208 | 4 | 98.6 % (205) | 0.19551 (1) | 0.58760 (1) |
| 4 | 2,414 | 2,381 | 4 | 99.9 % (2,378) | 0.65462 (1) | 0.65054 (1) |
| 5 | 1,208 | 193 | 14 | 93.3 % (180) | 0.28755 (1) | 0.59378 (1) |
| 6 | 1,208 | 1,195 | 2 | 99.9 % (1,194) | 0.62614 (1) | none |
| 7 | 4,826 | 4,591 | 4 | 99.9 % (4,587) | 0.58719 (2) | 0.67084 (1) |
| 8 | 1,208 | 992 | 1 | 100 % (992) | none | none |
| 9 | 1,208 | 1,207 | 2 | 99.9 % (1,206) | 0.58568734 (1) | none |
| 10 | 1,208 | 1,190 | 2 | 99.6 % (1,185) | 0.58719 (5) | none |

## Table 2: legal share and top illegal reasons

Reasons are the first snake_case token in the `detail` column of rows with `stage = legality`.

| Interface | Legal share | Illegal rows | 1st reason (count) | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|---|
| 0 | 35.3 % | 2,601 | workgroup_memory_race (1,376) | arena_index_not_in_bounds (313) | subgroup_operation_in_nonuniform_control (254) | store_index_not_affine (196) | kernel_does_not_render (193) |
| 1 | 99.3 % | 22 | kernel_does_not_render (13) | store_index_not_affine (3) | workgroup_memory_race (2) | unknown_row (2) | kernel_does_not_parse (1) |
| 2 | 91.8 % | 263 | arena_index_not_in_bounds (83) | kernel_does_not_parse (58) | kernel_does_not_render (40) | workgroup_memory_race (38) | subgroup_operation_in_nonuniform_control (26) |
| 3 | 12.9 % | 1,402 | workgroup_memory_race (667) | arena_index_not_in_bounds (460) | subgroup_operation_in_nonuniform_control (133) | kernel_does_not_render (37) | store_cell_wraps (35) |
| 4 | 98.6 % | 33 | kernel_does_not_parse (14) | workgroup_memory_race (8) | kernel_does_not_render (6) | unknown_row (2) | feature_missing (1) |
| 5 | 16.0 % | 1,015 | workgroup_memory_race (462) | store_index_not_affine (123) | write_sets_overlap (95) | kernel_does_not_render (88) | subgroup_operation_in_nonuniform_control (72) |
| 6 | 98.9 % | 13 | kernel_does_not_parse (9) | unknown_row (2) | workgroup_memory_race (2) | none | none |
| 7 | 95.1 % | 235 | arena_index_not_in_bounds (88) | workgroup_memory_race (72) | kernel_does_not_render (30) | subgroup_operation_in_nonuniform_control (21) | kernel_does_not_parse (10) |
| 8 | 82.1 % | 216 | arena_index_not_in_bounds (54) | workgroup_memory_race (43) | subgroup_operation_in_nonuniform_control (43) | kernel_does_not_render (39) | kernel_does_not_parse (22) |
| 9 | 99.9 % | 1 | kernel_does_not_parse (1) | none | none | none | none |
| 10 | 98.5 % | 18 | kernel_does_not_render (6) | workgroup_memory_race (4) | kernel_does_not_parse (4) | store_index_not_affine (3) | write_sets_overlap (1) |

Legality stage for interfaces 0, 3 and 5 is layer 7 (write sets and ordering) in 2,505, 1,352 and 944 rows respectively.

## Table 3: the eight interfaces that never left 0.58569

| Interface | Rows | Rows exactly at 0.58569 | Feasible children from gen 1 on at 0.58569 | Any feasible value below 0.58569? |
|---|---|---|---|---|
| 1 | 3,218 | 99.3 % | 100.0 % | No |
| 2 | 3,218 | 91.8 % | 100.0 % | No |
| 4 | 2,414 | 98.5 % | 100.0 % | No |
| 6 | 1,208 | 98.8 % | 100.0 % | No |
| 7 | 4,826 | 95.0 % | 100.0 % | No |
| 8 | 1,208 | 82.1 % | 100.0 % | No |
| 9 | 1,208 | 99.8 % | 100.0 % | No |
| 10 | 1,208 | 98.1 % | 99.6 % | No |

"Rows" counts all evaluated rows, including illegal ones. Illegal rows have no fitness in the ordinary sense; they sit in the infeasible band (above 1.5). No row on these eight interfaces ever scored below 0.58569.

## Table 4: interfaces 0, 3 and 5

### Generation 0 rows that are neither 0.58569 nor illegal

| Interface | Gen 0 rows | Illegal | Non-do-nothing feasible rows (row: fitness) |
|---|---|---|---|
| 0 | 22 | 5 | row 1: 0.33771; row 2: 0.63656; row 4: 0.63227 |
| 3 | 10 | 2 | row 0: 0.19551 |
| 5 | 8 | 1 | row 0: 0.28755 |

### Best fitness by generation

Population best, from the `# typed gen` lines of the log:

| Interface | Gen 0 | Gen 10 | Gen 50 | Gen 100 | Gen 200 |
|---|---|---|---|---|---|
| 0 | 0.33771 | 0.33771 | 0.33771 | 0.33771 | 0.33771 |
| 3 | 0.19551 | 0.19551 | 0.19551 | 0.19551 | 0.19551 |
| 5 | 0.28755 | 0.28755 | 0.28755 | 0.28755 | 0.28755 |

Best child in the ledger at each generation, with feasible children logged in that generation:

| Interface | Gen 10 | Gen 50 | Gen 100 | Gen 200 |
|---|---|---|---|---|
| 0 | 0.58587 (1 feasible child) | 0.58569 (2) | 0.58569 (4) | 0.58569 (4) |
| 3 | none feasible | 0.58569 (2) | 0.58569 (2) | 0.58569 (1) |
| 5 | none feasible | 0.58569 (2) | none feasible | none feasible |

Interface 0 at gen 0 has 17 feasible rows, interface 3 has 8, interface 5 has 7. From gen 1 on, feasible children are rare: 0 to 4 per logged generation.

### Size growth

The log's median node count per interface jumps once evolution starts:

| Interface | Gen 0 median nodes | Gen 1 median nodes | Gen 10 median nodes |
|---|---|---|---|
| 0 | 19 | 12 | 872 |
| 3 | 7 | 681 | 706 |
| 5 | 12 | 695 | 691 |

## 5. Most likely reasons evolution does not improve

The numbers point to four causes, in order of weight. First, almost every legal child scores exactly the do-nothing score, so the search gets no signal: between 92 and 100 % of legal children on the eight stuck interfaces, and 98.6 % on interface 3, sit at 0.58569. Second, legality is the wall on interfaces 0, 3 and 5, where only 35 %, 13 % and 16 % of children are legal. Workgroup memory races (layer 7) cause most of the rejections, so the few legal parents that remain have little variety to offer. Third, programs grow from a median of 7 to 19 nodes at generation 0 to 680 to 870 nodes within one or ten generations. Such large programs are mostly illegal or neutral, and the operators do not shrink them. Fourth, no child anywhere in the run beat its generation 0 seed. The best score on interfaces 0, 3 and 5 is set by the seed at row 0 or row 1, so the 0.19551, 0.28755 and 0.33771 values measure seed admission, not evolution. Two caveats apply: the ledger has no parent column, so neutrality is measured against the do-nothing score, and the ledger holds children only, so the child bests in Table 4 are lower bounds on what was tried.

## Not done

- No git push. The request asked for one, but `fuller/CLAUDE.md` says "Do not push", so the commit stays local.
- No code, GPU or cargo runs. The analysis used `python3 -I` on the ledger and the log.
