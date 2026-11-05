# Adversarial review: plan section 8, Typed Symbolic Regression (TSR) on the new data model

Reviewed: `docs/PLAN_futhark_kingdom.md` section 8 at fuller 5c28c5e2 (the advisor's review applied), against phylu
d73ca669 (`src/evolve/engine.rs`, `typed_decode.wgsl`, `typed_decode.rs`, `tsr_typed.rs`, `cand.wgsl`,
`evaluations.rs`, `examples/evolve_fit.rs`, `scripts/law_runner.py`, `scripts/sr_old_new.py`, `scripts/ks_one.py`,
`experiments/order133.txt`, `experiments/tsr_arms.json`, `logs/SR_OLD_NEW` run1 to run3, every `result.json` and a
`card.json`) and fuller `src/geneframe.rs`, `src/tsr_kingdom.rs`, `docs/PLAN_tsr_typed_gep.md`. Read only; nothing
built or run. 2026-11-05.

## Verdict

**Do not build as planned. The design of the properties is sound, but the engine would score one tree and confirm,
report and graft another, so the new arms could not be credited with a law found through a substitution.** The device
decoder is the only place the plan puts the substitution; every host path of the engine (the stop bar's f64 confirm,
the reported model, the beam, snap, fold, typed snap) decodes the raw gene with a decoder the plan does not touch.
Second, the depth limit leaks through the wrappers in the new arms. Third, the problem list does not follow its own
rule, and the deciding seeds are the seeds that chose the problems. Fourth, the "every feature fires" rule cannot pass
(share is zero by construction) and the finisher would take hours, not minutes. Each has a fix below; with them the
plan is about 10 agent days, not 8.6.

What checks out: the hardest ten and their 15 of 100; the six lost laws; the 22-minute figure for the engine of
2026-10-20 without the features; 104 words a gene; the 11 generations and the zero feature counts.

## Findings, most serious first

### 1. The host decodes the raw gene; the device would score the substituted one (critical)

Where: plan 8.3 ("The host twin (`typed_decode.rs`) ... must give the same node array"), 8.5, 8.6 item 3; phylu
`engine.rs:7232-7237` (`decode_host`), `tsr_typed.rs:47-49`.

For a single-layout kingdom, `decode_host` calls the flat `decode_gene`; only a `TsrTyped` kingdom goes through
`codec.gpu_nodes`, and `TsrTyped::new` refuses the single layout ("the single kingdom is sized by the head"). So the
plan's "props variant on the single layout" in `tsr_typed.rs` does not reach `decode_host`, and `typed_decode.rs`'s
`decode_reference` is used only by tests (`tsr_typed.rs:745`). These engine paths read the raw tree:

| path | where | what goes wrong under substitution |
|---|---|---|
| the stop bar's f64 confirm | `confirm_over`, `engine.rs:6291-6294` (`nodes_of_gene`, then `t_depth(n) <= c`) | with `EVOLVE_TYPED_DEPTH=2` the host refuses every gene the device substituted for depth, so that row can never pass the bar; with it unset, the host scores the raw tree, a different value, so the bar fails or passes on a tree that was never selected |
| the reported model | `math_of`, `engine.rs:9154` (hof, `model_plain`, the finisher's input) | prints a tree the device never scored |
| snap | `engine.rs:6398`, `6588-6598` (`node_to_gene_position`) | node index to gene position assumes one node a position, no dropped arms: grafts land in the wrong place or in silent material |
| fold, typed snap | `engine.rs:7066`, `7388` | the device's (substituted) nodes are paired with the raw `codec.nodes` tree, so positions disagree |
| beam | `beam`, via `confirm_over` | as the confirm |

Fix: one host decoder with the substitution (extend `decode_reference`), used by `decode_host` for `props`, that
also returns each node's gene position. Add tests that run the engine's own paths: for 20,000 random genes with at least
one substitution, `decode_host` equals the device's array word for word; `confirm` on the best row re-scores within
f32 of the device's fitness; `math_of` parses back to the device's array; every graft position from
`node_to_gene_position` is an expressed position. Put `engine.rs` (`decode_host`, `nodes_of_gene`, `confirm_over`,
`math_of`, `node_to_gene_position`) in 8.5 and in P2.

### 2. The depth limit leaks through the wrappers, and the arms do not get the same wrappers (critical)

Where: plan 8.7 requirement (5); phylu `cand.wgsl:66`, `typed_decode.wgsl:762`, `experiments/tsr_arms.json`.

`LogAbs` and `SqrtAbs` are raising wrappers. `cand.wgsl` refuses them over a depth-2 body only when the typer's `typed`
flag is on, and that flag is `Config::typed_depth`. The plan does not say what `EVOLVE_TYPED_DEPTH` is in the two new
arms. If it is unset (as arms C, E, F are), a new arm may wrap a depth-2 body in a log: a model of depth 3, which the
limit forbids, and more candidates than arm A has. If it is `2`, the device's type ladder (`typed_decode.wgsl:762`)
fails the gene before any substitution runs, and finding 1's confirm filter applies. Fix: take the typer's wrapper
check from the props table (the wrapper is a raising unary at the root), not from `typed_depth`; write each arm's full
environment in `tsr_arms.json`; extend test 2 to the wrapped candidate.

### 3. The unsolved ten do not follow the stated rule (high)

Where: plan 8.7 "Ten it has not solved". Rule: 0 of 10 in both arms of run2 and run3; the lost laws I_15_10,
II_13_34, I_12_11; the Gaussians I_6_2, I_6_2b; then by position in `order133.txt`.

The 33 laws at 0 in both arms of run2 sit at positions 51 to 100 of `order133.txt`. The first five by position, after
the five named, are strogatz_lv2 (51), feynman_II_21_32 (53), feynman_I_8_14 (56), feynman_II_11_28 (58),
feynman_II_11_27 (62). The plan has feynman_test_9 (81) and feynman_I_30_3 (82) instead of lv2 and II_21_32, and lists
the three it shares out of order. Dropping III_4_32 (88) and II_35_21 (80) was right; keeping test_9 and I_30_3 was
not. **The correct ten:** feynman_I_15_10, feynman_II_13_34, feynman_I_12_11, feynman_I_6_2, feynman_I_6_2b,
strogatz_lv2, feynman_II_21_32, feynman_I_8_14, feynman_II_11_28, feynman_II_11_27. (I_6_2b is 0 of 10 in both arms
of run3, as the rule needs.)

Verified as written: the hardest ten are exactly the laws at 1 or 2 of 10 in run2's new arm, 15 of 100; run1's lowest
is 4 of 10 (II_38_14) and run3's only solved law is feynman_test_5 at 4, so neither changes them. The six lost laws
(`PLAN_tsr_typed_gep.md` line 60) are II_13_23, I_10_7, II_13_34, I_15_10, I_12_11, I_6_2a: five are in the 20 and the
sixth is the eleventh law, as the plan says.

### 4. The deciding seeds are the seeds that chose the problems (high)

Where: plan 8.7 requirement (5), DEV_SEEDS 7001 to 7010.

The hardest ten were chosen because the 2026-10-20 binary passed 1 or 2 of exactly these seeds, the unsolved ten
because it passed none. Arm A is that configuration. Where the engine is deterministic for a seed, arm A replays its
lucky seeds on the hardest ten and its unlucky ones on the unsolved ten, while the new arms draw fresh luck: on the
hardest ten the comparison leans towards A, on the unsolved ten towards the new arms. Fix: choose on 7001 to 7010 (as
done), decide on 7011 to 7020.

### 5. "Every feature fires" cannot pass, and the finisher costs hours (high)

Where: plan 8.7 requirement (3); phylu `evaluations.rs:17-30`, `law_runner.py:118`, `408-412`, `ks_one.py`,
`examples/evolve_fit.rs`, `logs/SR_OLD_NEW/run2/new/feynman_I_10_7/seed7001/card.json`.

- **share** is zero inside every fit by construction ("Share (not in the live loop) ... are zero inside a fit"; the
  population Fold that would charge it is not built). The rule "a source at 0 in either arm is our software has a bug"
  fails every run. Fix: remove share from the rule and say why.
- **post_search** (the finisher) is filled only by `--check ks`, which runs `ks_one.py` per seed: a kitchen-sink run
  with a 90 s timeout, up to 30 SRBench verdicts with a 40 s timeout each, 780 s a law. Over 630 seed-runs that is
  hours, against a 50-minute guard. In every recorded checked run, post_search was above 0 in 6 of 174 seed records.
  Fix: `--check ks-gate` (gate passers only) inside the run, or the finisher as a separate timed phase after it; state
  which count decides (gate pass or "KS FOUND").
- **beam** has no switch: `evolve_fit` reads `EVOLVE_SNAP_EVERY` and `EVOLVE_FOLD_EVERY` but nothing for the beam.
  P3 must add `EVOLVE_BEAM_EVERY`; the plan names fold, snap and typed snap only.
- **tower**: the baseline ran with `tower: false` (the card). "Keep the tower on" changes the fitness (an extra HFF
  objective), which by the house rule is a new named fitness scheme, and puts a second change into every arm. Fix: tower
  off, as recorded.

### 6. The beam at every 3 generations eats about a quarter of the budget (high)

A generation is 89,820 counted evaluations (about 1,996 rows times 45 candidates). The beam scores 2,000 mutants a beat,
45 candidates each, about 90,000 evaluations, charged to the cap. Beats at generations 3, 6 and 9 take about 270,000 of
1,000,000, so about 8 main generations instead of 11. On the hardest ten, 12 of the 15 new-arm passes came at generation
4 or later (5 to 11 for most). The cost is equal in every arm, but it lowers the very counts the decision rests on.
Fix: record main generations per arm in the table; the calibration (R1) runs with the beam on and reports it.

### 7. A decision rule with little power and a tie that favours the new arm (high)

Where: plan 8.7, "New TSR is kept if n10 ≥ n01".

The hardest ten gave 15 passes an arm; the unsolved ten gave 0. Discordant pairs will be about 10 to 25, so an exact
McNemar test reaches p < 0.05 only at about 3 to 1 or wider. "n10 ≥ n01" keeps the new TSR on a tie, so with no true
effect it keeps it more than half the time. Fix: state in the plan the smallest split the run can detect; call the
result a win only at McNemar p < 0.05, a loss only at p < 0.05 the other way, and otherwise "no measurable
difference".

### 8. Making the bar a parameter touches five places, and the second count is not needed (medium)

Where: `law_runner.py` lines 62 and 67-80 (`FIT_ENV` is built once, at import, from the constant), 196 (gate note),
219-220 (the card check compares with the constant), 353 (refuses any other bar from the environment), 427 (the
result's `stop`); `sr_old_new.py` 15 and 102-104 (passes only a fixed list of `PHYLU_` names).

If P3 changes the child environment only, the check at 219 marks every seed failed. If it changes the constant only,
every other caller (`scripts/runpod/sweep_ab1.sh`, `scripts/wgsl_override_sweep.sh`, the PROD1 scripts) changes bar
silently. Fix: one `--bar` / `PHYLU_BAR`, read once and used in all five places; `sr_old_new.py` passes it through;
`--aggregate` refuses to put arms with different `stop.gate_one_minus_r2` in one table.

The 1e-6 count beside it is not worth an engine change: in `logs/SR_OLD_NEW`, 15 of 15 new-arm passes on the hardest
ten, and 945 of 970 passes over all runs, already had train and validation 1 − R² ≤ 1e-10 when they stopped (the
largest on the hardest ten, 1.6e-8, is one old-arm seed of II_10_9). The engine also has no field for it: stopcheck
lines are written only on a pass or a 10 % better validation error (`engine.rs` near 9835). Fix: drop the second
count; report the final train and validation 1 − R² per seed, which `result.json` already holds.

### 9. The time: 22 minutes holds only without the features (medium)

Measured from `result.json`: about 1.1 s of engine time a seed for 1,078,020 evaluations (about 1 million a second),
about 2 s of wall a seed, about 20 s a law and arm. The 20 laws plus I_6_2a sum to 437 s of new-arm wall on the
2026-10-20 engine; three arms make 22 minutes. Not in that figure: typed snap at every 3 instead of every 10 (about
250 ms a beat, `PLAN_tsr_typed_gep.md` finding (e); about 10 more minutes), fold's saturation each beat (unmeasured),
and the finisher (finding 5). Fix: R1's calibration runs one law with every switch the deciding run will use, finisher
included, and projects from that.

### 10. Section 7 shares the phylu decoder (medium)

"Touches no Futhark file" is true of fuller. In phylu, section 7's P1 (fb69e6cc, the Dv lane) has already changed
`typed_decode.wgsl` and `typed_decode.rs`, and section 7's P3 is in flight on the same engine (d73ca669). Section 8 names
its own task "P1" for the same files. Fix: rename section 8's tasks (TP1, TP2, TP3); TP1 starts only when no section 7
phylu task has uncommitted work in `typed_decode.*`, `typed_write_back.rs` or `engine.rs`; "the old arm's decode
unchanged byte for byte" compares with today's binary, not 2026-10-20's.

### 11. Task sizes (medium)

P2 at 1.5 days already held fold, constant snap, typed snap, denoise, the e-graph forms and share; finding 1 adds the
engine's host decode, the position map, the confirm, the report and the beam. Three days is honest. The wrapper flag
(finding 2), the beam switch, the bar in five places and the problem list add about half a day to P3. Total about 10
agent days; critical path about 8.

### 12. Small things (low)

- Arms run in file order on each law (`sr_old_new.py:347-349`), not alternating as its docstring says (line 47); put
  the order in the plan and rotate it per law.
- `paired()` compares every arm with one reference (`sr_old_new.py:200`, `241-244`); old against depth-only and
  depth-only against full need two aggregates or a pairs argument (P3).
- `SymbolTable::wide()` is phylu's (`engine.rs:193`), not fuller's.
- The device stores a leaf's ties as 0 (`typed_decode.wgsl:727`), the plan as 1. Harmless (a leaf's ties is never read,
  its family is none), but test 3 must not compare that word on leaves.
- The 1,270 seed-runs in requirement (3) are the new arm only; `logs/SR_OLD_NEW` holds 2,560.

## What the walk, variation and the other features see

| stage | sees | right? |
|---|---|---|
| walk, `typed_vary`, init | the raw gene (genotype) | yes: GEP varies the gene; dropped arms are silent material |
| decode, evaluate, score, HFF, dedup hash | the substituted tree | yes, once built |
| typer's wrapper check (`cand.wgsl`) | the tower, gated by `typed_depth` | no: finding 2 |
| stop bar confirm, beam, reported model, hof, finisher | the raw tree (`decode_gene`) | no: finding 1 |
| snap, fold, typed snap, the e-graph forms | the device's nodes with the raw tree's positions | no: finding 1 |
| share | nothing (not in the live loop) | not applicable: finding 5 |
