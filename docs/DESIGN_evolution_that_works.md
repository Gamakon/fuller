# DESIGN: evolution that works, to win the Verifiable Quantum Circuit Compilation prize

## Context

**Objective.** Win the Google Quantum AI *Verifiable Quantum Circuit Compilation* competition on Kaggle ($150,000; entry deadline 13 January 2027, final 20 January 2027). We aim to win it by **solving genetic programming**: making evolution a workable way to find globally optimal programs, by blending every neural and symbolic method we have.

**The challenge is not building a kingdom.** We can map kickmix circuits to a multi-typed GEP kingdom, as we did for Futhark, WGSL, regex and symbolic regression. The challenge is the same one the Futhark study exposed: evolution that finds the best program, not just a legal one.

**Approach.**
- Every method we have is used and tested, each on the job it is provably good at:
  - language models for big jumps;
  - typed GEP for recombination and fitness-guided local search;
  - the e-graph for cost reductions that are correct by construction;
  - exact search for proving small windows optimal;
  - the fold and the library for promoting the building blocks that correct programs share.
- Diversity is built into selection, so no single program takes over the population.

**What is unique.**
- Every gene is a legal circuit by its types.
- The fitness is exact verification, by fuzzing on the GPU, plus the competition's own cost.
- Every mutation method we have, used and unused, is measured on its own before it joins the search.

**Status (2026-10-30).**
- Design only.
- The Futhark test bed has shown the search learns: on the fixed fitness the HFF falls every generation, and evolution beat random search in 4 of 4 runs. It has not yet produced a correct program faster than a published one.
- Evidence: `docs/reports/2026-10-29_futhark_reruns.md`, `docs/PLAN_futhark_fitness_fixes.md`.

**Why.** The verifier makes this competition a clean test of the question: correctness is checked exactly, the cost is a count, and there is no timing noise. If evolution can win here, it works.

## 1. The competition

Sources: the Google open-source blog post of 21 October 2026, arXiv 2610.23832, and kickmix 1.0.0 (Qualtran). The exact scoring rules must be read from the Kaggle page itself before any submission.

- **Three tasks**, each with a reference circuit and a verifier:

| task | the circuit's job |
|---|---|
| invsqrt (fusion) | apply a phase proportional to 1/√x, within a stated error |
| dqi (optimisation) | decode a Reed-Solomon code in superposition |
| dfthcsos (chemistry) | lookup-table circuits for FeMoCo, plus a Hamiltonian factorisation |

- **Score, as reported by a community repository:** the geometric mean of each task's cost reduction against its baseline. Cost comes from qubit and Toffoli counts.
- **Circuits are kickmix circuits:** classical reversible gates, phase gates, and measurement-based uncomputation.
  - Each input bitstring maps to one output bitstring and one phase, so a circuit is checked by simulating random inputs.
  - The inputs are seeded by a hash of the submission, so there is no hidden test set.

## 2. What the Futhark work taught us

| finding | evidence | consequence for this design |
|---|---|---|
| Nothing learns until the fitness rewards getting closer | the study: evolution no better than random; the slope check failed (Spearman −0.08); after the fix, the HFF fell in every run | the fitness passes a slope check before any run (section 4) |
| Random starts are dense at dead points | 44–49 % of random programs output all zeros | start from the baseline circuits and from model-written alternatives, never from random alone |
| Point mutation does most of the work | point mutation: 4.4–7.9 % of children better than their parent; pump and fold 0 % (exact rewrites cannot change quality) | exact rewrites belong on the cost axis, after correctness |
| One good program takes over | Fable's exact seed held the best place for 18 generations, and nothing better appeared | diversity is enforced by selection (section 5) |
| A correct start makes improvement the task | Fable's programs were exact at generation 0 | every task starts from its correct baseline: the search is for cheaper, never for correct from nothing |
| The goal metric must be measured from generation 0 | no improvement run measured speed | every exact circuit's cost is reported every generation |

## 3. The kickmix kingdom

- **Types carry legality:** a register's width; a qubit's state (clean scratch, in use, must be returned clean); phase. So every gene is a circuit that restores its inputs and returns every scratch qubit to zero, by construction where the types allow it, and by the verifier otherwise.
- **Windows:** the full circuits have thousands of Toffolis over hundreds of qubits. A gene is a window of the baseline circuit (an adder, a comparison, a lookup step) with the rest fixed. Windows that improve are spliced back, and the whole circuit is verified.
- **GPU fitness:**
  - kickmix's bit-parallel simulation (Toffoli = AND then XOR; CNOT = XOR; phase = XOR into a phase bit) is ported to a GPU kernel;
  - a whole population of windows is scored on thousands of inputs in one dispatch;
  - the official kickmix verifier is the reference in tests, and checks every submission.
- **Fitness:** the two-level HFF (the hff library's TrueNorth, every measurement its own input), gated by correctness (section J of the Futhark plan).
  - Quality inputs: wrong output bits, phase errors, scratch qubits not returned to zero, inputs not restored, per input set.
  - Cost inputs: Toffoli count and qubit count, against the baseline window.
  - An inexact window always ranks below every exact one.

## 4. The fitness gate

Before any run, the slope check of the Futhark plan (section A):
- every one-edit neighbour of the baseline window, against random windows;
- it passes only if the neighbours rank above the random windows and the fitness falls as a window gets closer to correct.

## 5. Selection that keeps diversity

- **One slot per distinct behaviour** (output hash) and per algorithm family, so a champion cannot fill the population.
- **Islands by origin:** the baseline and its mutants; model-written alternatives; random canvas genes. Migration between them is measured.
- **Age layers** (ALPS, reformulated): newcomers compete with their own age group first.
- **Parent pairs never duplicates:** different outputs, not only different texts.

## 6. THE MUTATION METHODS: ALL OF THEM ARE TESTED

**This is the core of the design. We test every mutation method we have, the ones already in use and every one not yet used.** Each is first measured alone, then added to the baseline alone, then in combination. The search uses whatever the measurements show works. No method is left out because it is untried.

### Methods in use, and what they measured on Futhark (2 seeds, fixed fitness)

| method | what it does | unseen output | better than parent |
|---|---|---|---|
| point mutation | 1 to 4 edited cells, only cells the walk reads | 39–60 % | 4.4–7.9 % |
| crossover + mutation | swap a range inside one type's frame, then point-mutate | 44–59 % | 1.5–3.2 % |
| marriage | typed subtree swap between parents with different outputs | 29–48 % | 3.1–6.3 % |
| crossover alone | the range swap with no mutation | 15–22 % | 1.7–5.7 % |
| pump | e-graph variants of the best program | 0 % | 0 % |
| saturated fold | share-equivalence rewrite into shared definitions | 0 % | 0 % |
| ring walk | shift one type's window along a ring of real program pieces | 0.1–1.7 % | 1.5–2.7 % |

### Methods not yet used, all to be tested

| method | what it does | its job |
|---|---|---|
| **transposition** (IS, RIS) | copy a short run of a type's head (one starting with a function, for RIS) to another place in the same head | structural change that keeps working parts |
| **gene transposition** | move a whole gene of a multigenic chromosome to the front | reorders sub-programs |
| **inversion** | reverse a run inside one type's head | structural change |
| **sorting** | reorder the cells within a head (the ordering to be settled with Andrew) | structural change |
| **model edits** (Sonnet, then Opus, then Fable when the best stops improving) | a model is given an elite program, its scores and what is wrong with it, and asked for a variant; the reply is read back into a gene | big jumps: a different algorithm or structure |
| **model-proposed alternatives** | a model writes very different programs for the same task, with the problem statement changed when they are too alike | diverse starting points and new islands |
| **model-proposed rewrite rules** | a model proposes a rewrite rule; the verifier checks it before the e-graph uses it | widens the e-graph's reach |
| **the replacement dictionary** | every improvement found becomes a rewrite rule, kept and grown across runs | learning across runs |
| **e-graph extraction on cost** | equality saturation of an exact window, then extraction of its cheapest form | cost reduction, correct by construction |
| **window superoptimisation** | exact search (enumeration or a solver) over windows of 5 to 10 gates | provably optimal windows |
| **library components** | the building blocks that correct programs share, promoted by the fold to shared definitions and drawn as subtrees | reuse of discoveries |
| **component-drawn subtree mutation** | a subtree replaced by a popular component of its type | builds from proven parts |
| **cleanse, snap, hoist, let-share** (from symbolic regression) | simplify, snap constants, lift shared subexpressions | tidy and compress exact programs |
| **ring canvas growth** | a terminal turned function reads real program pieces from the canvas | growth into real material |

### How each method is tested

1. **Measured alone first:** 700 children on each task, with the share giving an unseen output, the share better than their parent, the share with the same output as their parent, and the share that fails verification.
2. **Then added alone to the baseline** (point mutation, crossover and marriage): the same seeds, budget and binary, with only that method added, at a stated share of the children.
3. **Reported every generation:** best and median HFF (is it falling?), and the Toffoli and qubit counts of the best exact window against the baseline.
4. **Kept if it beats the baseline** at equal numbers of circuits scored: a lower best HFF in at least 2 of 3 seeds, or a cheaper exact window.
5. **Then combinations** of the methods that were kept. Their shares are adapted during the run by each method's measured success rate.

## 7. Optimality: what we can claim

- Genetic programming cannot guarantee a global optimum.
- We can claim two things:
  - optimal within the e-graph's rules, by equality saturation;
  - provably optimal per window, by exact search.
- The global search widens both over time, as the dictionary of rules grows and new windows are proven.

## 8. Order of work

| # | step | done when |
|---|---|---|
| 1 | Read the Kaggle rules: scoring formula, tool rules, data terms | the rules are quoted in this doc |
| 2 | Fix the Futhark timing paths, and run the histogram improvement as a test bed for sections 5 and 6 | all three evaluation paths agree; speed reported every generation |
| 3 | The kickmix kingdom: types, windows, splice-back | baseline windows read in and verified bit for bit |
| 4 | The GPU fitness: the bit-parallel simulator on the GPU, the official verifier as reference | the same verdicts as kickmix on every test circuit |
| 5 | The slope check on invsqrt windows | passes |
| 6 | Measure every mutation method of section 6 alone, on invsqrt | one table, every method |
| 7 | Runs: each method added alone, then the kept ones combined | the table of section 6, step 4 |
| 8 | dqi, then dfthcsos | as invsqrt |
| 9 | Submissions, each checked by the official verifier | a leaderboard score |

## 9. Expectations, stated now

| measurement | expected |
|---|---|
| the slope check on invsqrt windows | passes, because the verifier gives graded bit-error counts |
| model edits, children better than their parent | higher than point mutation's 4–8 % |
| e-graph extraction on exact windows | Toffoli reductions on some windows, with none on others |
| a cheaper exact invsqrt circuit than the baseline | likely, by combining windows |
| winning the prize | possible, not expected: the field includes the circuit experts the tasks come from |

## 10. Risks

- The big wins may need algorithmic insight, such as a new decoder or a new 1/√x method. That is the job of the model edits and the model-written alternatives, and it is why they are tested.
- The chemistry task is half numerical (the factorisation), which a circuit kingdom does not cover.
- The competition's rules on AI tools must be read first.
