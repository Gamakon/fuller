# Evolving GPU programs: results of 2026-10-22 to 23

**Part A** is the instance: can we evolve a faster UltraDim?
**Part B** is the class: have we built an environment where complex GPU pipelines can be learned?

## Summary

**Part A, the instance.** No evolved ingest routine beat today's UltraDim index. Under a memory budget of 268 bytes a record, the best evolved kernel was 59–107× faster at ingest and 146–267× faster at search. But its recall at 250,000 records was 0.62 falling to 0.38, against today's 0.98 falling to 0.97. Without the budget, a model-written GPU scan reached recall 0.999 with ingest in milliseconds. That was a single, inadequately tested measurement, and it is not a result.

**Part B, the class.** The environment finds improvements that exist: planted slowdowns were removed in both kingdoms tested. Most of the time went on finding and closing faults in measurement and fitness. Seven separate ways a candidate could score well without being better were found, each only after a run. Evaluation throughput is about 45 generations an hour at 250k, and that limits learning.

---

## 1. Setup

**Hardware and software**
- One Apple M3 Max laptop. Metal through wgpu/naga 30.
- One GPU process at a time.
- Every dispatch passes a work guard before it is submitted: at most 1.7 × 10⁸ node executions a submit, the WindowServer safety limit measured on this machine.
- Code: fuller (kingdom, harness, optimisers) and phylu (evolution engine). Commit ranges are given per experiment below.

**Representation.** True multi-typed gene expression programming:
- A kernel is one Karva gene with a head and tail per type.
- The symbol table's types carry every legality rule: ownership of writes, uniform control for barriers, bounded loops, in-range indexes.
- So every gene expresses a legal WGSL kernel, with no repair or rejection.
- Legality is checked by naga on random genes (1,000 of 1,000 per table version) and by device runs.

**Fitness (both problems).** HFF TrueNorth from the hff library, normalisation off, every measurement its own input, nothing averaged. The result is an angle from 0 (perfect) to π/2 ≈ 1.571 (worst).

Since 2026-10-22 22:00Z the fitness is two-level:

> θ_quality = HFF(quality inputs), θ_speed = HFF(speed inputs), fitness = HFF(θ_quality / (π/2), θ_speed / (π/2)).

This was adopted because a single flat HFF with 45 speed inputs and 24 quality inputs let wrong-but-fast kernels win (section 4.2). Each speed input is r/(1+r), where r is the candidate's time divided by the reference time, so 0.5 means equal.

**Variation (final configuration)**
- **Marriage:** two-parent reproduction, share 1.0 of non-elite rows, with two distinct parents (different rows and different expressed trees). It uses homologous one- and two-point crossover and typed subtree crossover via the per-type write-back.
- **Offspring mutation:** probability 0.2.
- **Selection:** tournament of 3, 2 elites.
- **SR's in-loop optimisers** on the top 50 % each generation (`--optimisers sr`): cleanse (dead writes), exact e-graph, picture- or result-changing e-graph, hoist out of loops, fewer loop trips, snap. Fold is off since run 6 (section 4.6).
- **Pump:** every 50 generations, injecting the hall of fame and unused seeds.

**Controls run before any real experiment**
- A/A timing: the shipped kernel against itself gave 0.9991 [0.9982, 1.0000]; our encoding against itself gave 1.000.
- Pre-flight: every operator must produce children with 0 defects.
- Planted-win positive control (section 3).

---

## 2. Part A: UltraDim ingest (the instance)

### 2.1 Problem
- Four seeds' 2,048-D sparse keys stream in five microbatches of 50,000 records, 250,000 in total.
- After each microbatch, 2,500 oracle queries search the index.
- Goal: recall@10 ≥ 0.996 on first ingestion, faster than today.
- Data: ARM card records, 718k-word + geohash space. Exact truth comes from UltraDim's brute force on the CPU, used only as the test reference.
- Problem `fuller/problems/v24_ingest250k` (fuller 05bad0a3). Index memory is fixed at 32 MiB a seed, which is 268 bytes a record at 250k. The run-4 scan needs about 512 bytes a record, so it cannot simply scan.

### 2.2 Baselines (measured 2026-10-22 22:40Z, 3 runs; recall was identical across runs)

| stage | records | today recall@10 | today settle (s, cumulative) | today search (s, 2,500 queries) | GPU scan, 32 MiB: recall | GPU scan, 128 MiB: recall | scan ingest (ms) | scan search (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 50,000 | 0.9795 | 55 | 13.7 | 0.384 | 0.9969 | 10 | 674–761 |
| 2 | 100,000 | 0.9812 | 175 | 15.2 | 0.259 | 0.9997 | 21 | 1,520–1,598 |
| 3 | 150,000 | 0.9799 | 305 | 13.9 | 0.002 | 0.9994 | 32–35 | 2,526–3,105 |
| 4 | 200,000 | 0.9745 | 437 | 25.1 | 0.002 | 0.9995 | 44–49 | 3,599–4,946 |
| 5 | 250,000 | 0.9744 | 504 | 16.8 | 0.001 | 0.9994 | 54–62 | 5,370–5,430 |

**Limits of this table:**
- One dataset only.
- Today's search is 2,500 separate calls summed, wall time; the scan's is GPU time. They are not like for like.
- The 128 MiB scan was widened by hand and not evolved or reviewed.
- No updates, deletes or concurrent load were tested.

### 2.3 Evolution: method and parameters (run 6, the final one)

| parameter | value |
|---|---|
| seeds | the readable V24 kernels: 50 of 67 read exactly into the true kingdom by 06:46Z; run 6 used the set read at 06:39Z, plus the run-4 and run-5 champions |
| population | 18 |
| fitness case | 250k cut to stages 1–2 (50,000 and 100,000 records), first 250 queries |
| time budget | 20 s of GPU wall a candidate; unfinished stages score recall 0 and the worst speed |
| verification | the hall of fame's best measured on all 2,500 queries and 5 stages every 10 generations |
| quality inputs | per stage and per query half (one half held out): miss rate, misses against today's, broken entries |
| speed inputs | per stage: ingest GPU ms against today's settle, search GPU ms against today's search, each r/(1+r) |
| program size cap | 1,000 functions |
| duration | 06:39–07:55Z, 43 generations, about 45 generations an hour |

Runs 1–5 (02:16–06:39Z) were stopped for faults. Each is listed with its cause in section 4.

### 2.4 Results

**New bests on the fitness case**
- HFF went from 0.57211 (generation 0) to 0.57071 (generation 23).
- θ_quality went from 0.88634 to 0.88423.
- θ_speed stayed about 0.0147, already near its best, because the GPU kernels are about 100× faster than today's index.

**On the whole 250k case**

| stage | today recall | generation-1 best | generation-23 champion | today ingest (s) | evolved ingest (s) | today search (s) | evolved search (s) |
|---|---|---|---|---|---|---|---|
| 1 | 0.9795 | 0.6236 | 0.2550 | 55.1 | 0.94 | 13.7 | 0.093 |
| 2 | 0.9812 | 0.5535 | 0.2258 | 175.0 | 1.88 | 15.2 | 0.094 |
| 3 | 0.9799 | 0.4791 | 0.1779 | 304.7 | 2.82 | 13.9 | 0.094 |
| 4 | 0.9745 | 0.4199 | 0.1258 | 437.1 | 3.75 | 25.1 | 0.094 |
| 5 | 0.9744 | 0.3809 | 0.0878 | 503.7 | 4.69 | 16.8 | 0.094 |

The generation-23 champion is worse at full size. It replaced `cfg.n_queries` with the literal 1024: on the 250 scored queries nothing changed, but on 2,500 queries those numbered 1,024 and above get no candidates. That is overfitting to the fitness case.

### 2.5 Instance findings
1. **Speed is not where today's UltraDim loses; recall within a memory budget is.** Every GPU design is 59–267× faster than today's index. Under 268 bytes a record none keeps recall above 0.62.
2. **The seeds do not scale.** They were written for 25,000 records. At 250k each loses recall after the first microbatch, or has a workgroup over the work cap (5 of 16).
3. **Today's index falls short of the goal on this data,** at recall 0.974–0.981, below 0.996.
4. **The memory budget per record is a product decision, not a technical one.** 268 B a record defeats every design tried, while 512 B lets a plain scan reach 0.999 up to 250k.

---

## 3. Calibration problem: the cuneus shader volumepassage

This is used to test the class on a problem with an exact answer: the identical picture, drawn faster.

| experiment | parameters | result |
|---|---|---|
| encoding parity | 64×64, 41 pairs × 5 rounds | our generated code against shipped: 1.052× before the template fixes, 0.983× after (fuller 37f19dbc; the loop form cost 4.3 %, the batch offset 2.9 %) |
| planted control | 3 planted slowdowns, pop 16, 250 generations, 10.8 min | 1.742× → 0.962× shipped, picture identical, all three plants removed (re-timed: 0.984 [0.983, 0.985]) |
| real run (ST13) | pop 16, 580 generations, 18 min, two-level HFF, 45 timing slots | best HFF 0.45846 → 0.45476 by generation 3, then flat; speed 0.976–0.979× shipped, which is the encoding's starting advantage; 0 picture difference |
| earlier runs (old mutation only, LEAF, HIGH) | pop 10–32, 55–3,468 generations | no learning, or gains that were measurement artefacts (section 4) |

**Conclusion for the class:** the machinery removes waste where waste exists. The shipped volumepassage has no speed-up reachable by the current operators.

**Ingest planted control:** the slowdown planted in seed27 (a repeated loop) went from 4 to 2 to 1 repetitions by generation 26, 31 minutes in.

---

## 4. Part B: what we learned about the environment (the class)

### 4.1 Measurement faults dominate
Seven ways a candidate scored well without being better were each found only after a run:

| # | fault | effect | fix |
|---|---|---|---|
| 1 | timing kernels split into 16 submits | rewarded workgroup shapes that only help the split (8×8 → 2×16: 2 % faster split, 0 % whole) | time at a single-submit size, with the same submit plan both sides |
| 2 | noisy per-pair timing | HFF rewarded steady timings, not faster ones | fenced passes in one command buffer; 45 timing slots accumulated per kernel text (spread 0.012 → 0.010; a 2 %-faster kernel now ranks first 10 of 10) |
| 3 | NaN pixels counted as 0 error | kernels writing NaN, after Metal deleted the real work, scored as exact and 14× faster | NaN counts as the worst error |
| 4 | input counts unbalanced (45 speed, 24 quality) | wrong pictures outscored correct ones | two-level HFF (in the re-score, the 11 correct kernels rank 1–11) |
| 5 | a log-scale pool input | an empty search outranked recall 0.98 | input rescaled |
| 6 | a small screen not representative | a fixed 32k-row mask never filled at 2,500 records: screen recall 0.995, full recall 0.0005 | fitness on real-size data |
| 7 | a fixed subset of 250 queries | the champion hard-coded 1024 queries: full recall 0.38 → 0.09 | open: score all queries, or a fresh random set each generation |

### 4.2 Positive controls work
- Planted wins were recovered in both kingdoms.
- A/A tests showed no order bias.
- These controls separated "the search cannot find it" from "it is not there".

### 4.3 The starting point must equal the reference
- Our generated code was 5 % slower than the shipped shader, from template choices.
- Evolution then spent its effort recovering our own overhead.
- Parity is now checked before every run.

### 4.4 Variation
- **Mutation alone gave no learning at any strength:**
  - LOW: children overlapped their parent by 0.5;
  - LEAF: overlap 0.998, diversity collapsed;
  - HIGH: overlap 0.09, mean HFF 1.14.
- **Gains came from two-parent marriage and the in-loop optimisers.**
  - Volumepassage, ST12: cleanse improved 203 of 398 of its children; hoist 48 of 100.
  - The planted ingest kernel's main plant was removed by subtree crossover.

### 4.5 Legality by types holds, with known gaps
- **50 of 67 ingest kernels read exactly; 14 are named outside the kingdom with evidence** (races in their own text); 3 are gaps.
- **Every table change was checked on 1,000–10,000 random modules through naga.**
- **Bugs found by this process and fixed:**
  - an epilogue that stored nothing;
  - a pool clamp that raced;
  - a constant index naga rejected.
- **Open:**
  - **Over the work cap:** 4.6 % of children have one workgroup over the per-submit work cap. The work is real. Building the limit into the types would rule out today's reference (54× the cap by the type bound). The proposed per-kernel fuel counter is written in the spec, not built.

### 4.6 Operators that harm at convergence
- Fold (ADF) froze whole entries into 3-node trees. Best HFF did not move from generation 10 to 60, so fold is off.
- The checkpoint does not carry fold's definitions, so a folded run cannot resume.

### 4.7 Throughput is the binding constraint
- A full 250k evaluation costs 90–230 s per kernel.
- Metal compiles each new kernel in about 150 ms.
- With the 20 s budget the run reaches about 45 generations an hour, so a night gives about 200 generations at population 18.

### 4.8 Process
- Every one of runs 1–5 was stopped for a fault the run itself revealed.
- What gave the most information per GPU hour:
  - a 3-generation pre-flight;
  - a planted control;
  - A/A timing;
  - a verification on held-out full-size data at a fixed interval.

---

## 5. Next steps

### Instance (UltraDim)
1. Set the memory budget per record as a product requirement. Today's index uses full keys outside the GPU arenas.
2. Score all 2,500 queries, or a fresh random subset each generation.
3. Seeds written for the 250k and 1M scale (the 1M set is prepared, fuller bc8ec492; 6–7 h to build).
4. Re-measure today's search as GPU time where possible, so the speed inputs are like for like.

### Class (the environment)
1. A validation checklist that must pass before any run:
   - A/A;
   - a planted win;
   - a known-bad control (a wrong answer must score worst);
   - a held-out distribution check.
2. Parallel evaluation of the population across GPUs to raise generations an hour.
3. The table-as-grammar parser (plan UI5), so a run's hall of fame can seed the next.
4. The per-kernel fuel counter (spec written).

## 6. Where everything is
- **Reports:**
  - `fuller/docs/reports/2026-10-23_UI_ingest_overnight.md` (with chart, diffs and the champion's text);
  - `fuller/docs/reports/2026-10-22_ST9_volumepassage_champion/` (README + ADDENDUM).
- **Plan:** `fuller/docs/PLAN_true_gep_wgsl.md` (rows ST11–ST13, UI1–UI5).
- **Logs:** `/private/tmp/claude-501/ui4/long*.log`, `/private/tmp/claude-501/st1*/`.
- **Final commits:** fuller ef0b681d, phylu 76a2157f.
