# PLAN: deciding the open options by experiment (2026-10-13)

## Context

**Objective.** Decide each open option of the WGSL evolution work by measurement, not by judgement: how a correct kernel is first found, whether symmetric-half is worth a kingdom change, whether population-wide section ids make Karva crossover useful, and whether untrusted kernels need a resettable GPU worker.

**Approach.** One experiment per option, each with a stated hypothesis, the measurement that decides it, the expectation written down before the run, several seeds where chance matters, and paired timings wherever speed is the claim. Every run passes the common-sense check (what each mechanism feeds in) and builds from committed code into its own binary path.

**What is already measured (2026-10-12/13, docs/reports/).** Rules + selection speed up correct kernels 1.19× and 1.25× (paired). Six random-start runs, up to 180 generations, found no correct kernel; closeness improved only in cells-written and input-sensitivity, correlation with ΦᵀΦ stayed ≈ 0. One seed per configuration so far.

**Why now.** The kingdom's random kernels now meet every bar (input-reading stores, varying guards, under the cost bound, integer division guarded, compile on Metal 1,000/1,000), so a failure to find correctness can be blamed on the search, not on broken genes.

**Constraint.** The GPU is shared with faxl work; experiments run in windows Andrew grants, one GPU process at a time.

## E1. How is a correct kernel first found? (start: random, sketch, Sonnet)

- **Question.** Which start reaches a feasible (correct) kernel, and how fast?
- **Arms.** (a) random genes (the current sampler); (b) sketch: the outline the problem fixes — thread owns cell (i, j), a loop over the m items, an accumulator stored into h[cell] — with every other subtree a random hole, mutation restricted to the holes; (c) fresh Sonnet kernels (40, then no models); (d) control: (b) with the outline also mutable.
- **Measure.** Generations to the first feasible kernel (censored at the cap); best closeness per generation; the correlation column of the best; correct kernels per 1,000 evaluations; after the first feasible kernel, the paired speed-up the rule beam reaches in 20 more generations.
- **Seeds.** 5 per arm (a, b, d), 2 for (c) (model cost). Same operators, population 80, pump every 20, cap 200 generations.
- **Expectation, before the run.** (a) 0 of 5 reach correctness in 200 generations (as runs 1–6). (b) at least 3 of 5 do, within 50 generations, because the remaining search is the indexing of two Φ reads. (c) most Sonnet kernels correct at generation 0 (earlier runs ~50 %). (d) between (a) and (b): mutation breaks the outline.
- **Decision rule.** Use the arm with the most feasible discoveries per GPU hour; if (b) beats (a) by ≥ 3 of 5, the sketch becomes the standard start for a new problem, and the paper reports (a) as the negative control.
- **GPU cost.** ~40 s a generation: (a), (b), (d) 15 runs × ≤ 200 gens ≈ 33 h worst case, much less if correctness comes early; run (b) first.

## E2. Symmetric-half: is it worth a kingdom change?

- **Question.** H = ΦᵀΦ is symmetric; computing only cells with i ≤ j and writing each mirror should approach 2×. Does it, on our kernels, and does it survive the band layout?
- **Build.** A kingdom change first, measured on its own: an ownership dual that lets a thread own a cell and its mirror (checker layer 2 extended to prove the pair disjoint across threads), the fitness's banded write extended to mirrored bands. Then the rule symmetric-half.
- **Measure.** On the 49 correct kernels of popR_rules: children correct (128 cases + pairs), paired speed-up (6 interleaved rounds, the fit's shapes), and a check that the 1,000 random genes stay legal and race-free under the new dual.
- **Expectation.** 1.6–1.9× median on the tiled kernels (the diagonal tiles and the extra mirrored store cost something); 100 % correct; 1,000/1,000 random genes legal.
- **Decision rule.** Keep the kingdom change if the median paired speed-up is ≥ 1.3× and no race appears; otherwise revert it.

## E3. Population-wide section ids: does Karva crossover become useful?

- **Question.** With stable section ids (the population-DAG design: a reference names a definition, never a slot), do Karva crossover children stop picking up fallbacks and stop being copies, and does that help a run?
- **Measure, offline (CPU).** The existing wgsl_cx_measure table on popSF_g58c and popR24_t: children with no fallback node, children identical to a parent, children passing check.rs — before (today: 45–71 % clean, 43–88 % copies) and after.
- **Measure, in a run.** E1's winning arm, 3 seeds with --karva-cross and 3 without: feasible discoveries, best closeness, and offspring feasible by origin.
- **Expectation.** Offline: ≥ 95 % children clean, ≤ 30 % copies. In a run: no worse than typed crossover alone; a gain only if crossover children are feasible at a rate comparable to rewrites (68 % not broken).
- **Decision rule.** Keep stable ids if the offline gate passes and the run arms are not worse; otherwise park the design as future work in the paper.

## E4. Do untrusted kernels need a resettable GPU worker?

- **Question.** How often does an evolved kernel hang the GPU now that integer division is guarded and runaways are detected, and what does a hang cost on the Mac against a Linux/Vulkan worker?
- **Measure.** Runaways per 1,000 evaluations across E1–E3 (the new runaway stage counts them); for each, whether the GPU drained by itself. On a RunPod Linux/Vulkan pod: the same random kernels (including r62 unguarded) — does the driver's watchdog recover the GPU, how long, and ms a generation against the laptop.
- **Expectation.** On the Mac after f17ccb9: runaways rare (< 1 per 10,000) but each one needs a reboot. On Linux/Vulkan: the watchdog recovers in seconds without a reboot.
- **Decision rule.** Move evolution runs to a resettable worker if the measured rate times the cost of a reboot exceeds the cost of the pod, or if any runaway occurs that the Mac does not drain.
- **Also:** confirm the r62 bisection (logs/HGP/001/r62_bisect: variants a, c, d, e; b stays off on the Mac).

## E5. The paper's open questions

Most are facts to look up, not opinions; each answered by citing the run or commit that settles it (outline.md lists them).

## Order
1. r62 confirmation (5 min GPU) — the safety fix must be proven before more runs.
2. E1 arm (b) sketch, seed 1 — the most likely to change the plan.
3. E1 remaining arms and seeds.
4. E2 build (CPU), then its device table.
5. E3 offline gate (CPU), then its run arms.
6. E4 counts accumulate throughout; the pod comparison when Andrew approves the spend.

---

## Revision 1, after Codex's adversarial review (logs/HGP/001/codex_review/plan_experiments_review.md)

Codex's findings are accepted; the designs above are superseded by these.

- **Budgets, not generations.** Every search arm runs to a fixed **16,000 scored offspring** per seed (80 × 200), the same wall-time watchdog, the same operator mix. Generations are not equal work: feasibility changes the operator path and slow candidates stop early (gram.rs:1250-1301).
- **Statistics.** Success rates with Wilson 95 % intervals; evaluations-to-success as a restricted mean; paired common seeds across arms. Decision thresholds are on confidence bounds, not point values.

**E0 (new, first). Timing calibration.** 20 ABBA repeats of shipped against shipped, to measure the timing noise every later speed claim is judged against.

**E1a. Sonnet as an initializer, audited alone.** 40 independently generated kernels, prompt and context locked, no evolution: the correct fraction with its 95 % interval. It is an answer-informed start, so it is reported apart from the search arms, not compared with them.

**E1b. Random vs sketch vs mutable sketch.** 12 paired seeds per arm, 16,000 offspring each. The sketch supplies the algorithm's outline, so it measures completion of a given outline, not unguided discovery; it is reported as such. Adopt the sketch only if its success rate's lower 95 % bound exceeds random's upper bound. Cost: about 2 GPU hours a seed, 72 hours for all 36 — a 3-seed pilot per arm first, to check the arms differ at all before the full budget is spent.

**E2. Symmetric-half.** First a CPU ownership proof for the new two-cell dual and an adversarial stress over 1,000 generated kernels on held-out shapes and seeds (legality of the old invariant does not prove the new one race-free). Correctness on held-out shapes, not the fitness's own 128 cases. Speed: 15 structurally distinct representatives of the 49 correct kernels, 20 randomized ABBA rounds each, cluster bootstrap. Keep only if the lower 95 % bound of the geometric-mean speed-up is ≥ 1.3, all held-out tests pass and no race appears.

**E3. Stable section ids.** The CPU gate first: 10 populations × 1,000 offspring per operator, bootstrap intervals on clean and copy rates. Then 12 paired seeds per arm at 16,000 offspring, crossover rates identical. Non-inferior if the one-sided 95 % lower bound on the success-rate difference is above −0.10; useful if above +0.15.

**E4. Runaways.** Prevalence and recovery measured separately: 30,000 i.i.d. random legal kernels (bounds an observed-zero rate below 10⁻⁴ at 95 %); a fault suite (r62 unguarded plus pathological bounded loops), 20 trials each on the Mac and on a Linux/Vulkan pod, recording detection, drain or reset, and time lost. Decide by expected loss (rate × recovery time × run length) against the pod's cost. The Mac fault trials that are expected to hang run only on the pod.

**Revised order.** r62 confirmation → E0 → E3's CPU gate → E1a → E1b pilot (3 seeds × 3 arms) → E1b full if the pilot shows a difference → E2 build only if still worthwhile → E4's i.i.d. sample accumulated alongside.

---

## Revision 2: fitted to a 5-GPU-hour budget, 1 hour flex (Andrew, 2026-10-13)

Cost basis, measured: a generation of 80 takes about 40 s, so one scored offspring is about 0.5 GPU s; a 4,000-offspring seed (50 generations × 80) is about 33 min. Everything that can run on the CPU does, and costs nothing from the budget.

| # | Experiment | GPU | What is cut from revision 1, and why |
|---|---|---|---|
| 1 | r62 confirmation (bisect a, c, d, e) | 5 min | — |
| 2 | E0 timing calibration, 20 ABBA shipped/shipped | 5 min | — |
| 3 | E1a Sonnet audit: 40 kernels, correctness only, no evolution | 5 min | the model calls cost money, not GPU |
| 4 | E1b sketch arm: 4 seeds × 4,000 offspring | 132 min | 12 seeds × 16,000 → 4 × 4,000; the mutable-sketch control dropped |
| 5 | E1b random arm: 2 paired seeds × 4,000 offspring | 66 min | 12 → 2 new seeds; runs 1, 4, 5, 6 (≥ 4,000 offspring each, 0 successes) reported beside them as unpaired history |
| 6 | E2 symmetric-half speed: 15 representatives × 20 ABBA rounds | 15 min | runs only if its CPU build and ownership proof are done in time |
| 7 | E4 prevalence: 5,000 i.i.d. random kernels, correctness only | 10 min | 30,000 → 5,000 (bounds a zero rate below 6 × 10⁻⁴); every E1b evaluation also counts its runaways |
| | **Total** | **238 min** | |
| | Flex, up to 60 min | | first: extend any sketch seed that is improving when its budget ends; then a 5th sketch seed |

CPU only, no GPU cost: E3's crossover gate (10 populations × 1,000 offspring per operator); E2's build, ownership proof and held-out correctness; the paper's open questions. E3's run arms and E4's pod fault suite are out of this budget: the CPU gate decides whether stable section ids are worth building, and the pod suite is a separate spend.

**What this budget can and cannot decide.**
- E1b with 4 against 2 (+4 historical) seeds is a pilot. If the sketch succeeds in ≥ 3 of 4 seeds and random in 0 of 6, the Wilson 95 % intervals separate (sketch ≥ 30 %, random ≤ 39 % — they overlap slightly, so it is strong evidence, not proof); 4 of 4 against 0 of 6 separates them cleanly. A null result (both 0) is decisive in the other direction: the sketch does not help at this budget.
- E2 decides on the lower 95 % bound of the speed-up, as in revision 1; 15 kernels × 20 rounds is enough for that.
- E4 can bound the runaway rate below 6 × 10⁻⁴, not 10⁻⁴.

**Order.** 1 → 2 → 3 → 4 (seed 1 first; if it fails all 4,000 offspring and its closeness is no better than random's, run seed 2 before committing the rest) → 5 → 7 → 6 if ready → flex.

**Stop rules.** Any runaway or bug stops the experiment it occurs in (the dump is kept) and the budget clock stops with it; a GPU probe slower than 1.5 s a shape stops all GPU work.
