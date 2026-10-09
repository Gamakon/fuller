# PLAN: hallucino-genetic programming (HGP) on the Gram kernel

## Context

**Objective.** Test Andrew's hypothesis (2026-10-09) that a language
model's hallucinated programs, selected and refined by evolution, produce
a correct WGSL kernel faster than the shipped one: for Blue World's
weighted Gram `H = ΦᵀΦ` (`docs/PLAN_wgsl_gram_superopt.md`), a population
of kernels written by Sonnet from the task's contract, graded on a ladder
(parses, validates, compiles, correct, fast), varied by rewrite rules, by
model edits and by crossover, and reported as rung counts, correctness
rates per operator and the paired speed-up on held-out shapes.

**Approach.** The model proposes; it never judges. Every candidate climbs
the ladder: naga parses the text, naga validates it, Metal compiles it,
the probes and the pair criterion decide correctness, the paired timing
across shapes decides speed. Selection is by highest rung first, and the
HFF angle over the performance columns within the top rung. Invalid
candidates are kept with their diagnostics and offered one repair by the
model; a repaired kernel re-enters at the bottom of the ladder. Every
prompt, reply, measurement and parentage is on disk.

**Unique feature.** Hallucination is the diversity source, not a fault:
the model assembles whole pipelines the random typed sampler never lines
up (today's measurement: 440 of 440 random typed kernels valid, none
correct), and the ladder plus the HFF supply the judgement the model
lacks. The kingdom's own machinery (reader, rewrite rules, renderer,
probes, pairs, timing) is the middle of the loop unchanged.

**Status.** Plan, 2026-10-09 evening, nothing built. Supersedes the
typed-sampler search of `PLAN_wgsl_gram_superopt.md` §3 as the way to a
population of correct kernels; §1 (harness, criterion, pairs) and the
probes (`src/wgsl/probes.rs`) stand and are what this plan grades with.

**Why now.** Three runs of the typed-tree search today learned nothing:
no fitness improved over 31 generations, because a pipeline's output
fitness has no slope between all-right and one-step-wrong (an index off
by one gives an entirely wrong table). The reviewer's note and Andrew's
proposal agree: get plausible whole programs from the model, let
evolution select among correct ones.

## 1. The ladder (progressive fitness)

| rung | test | what a candidate earns | measured as |
|---|---|---|---|
| 0 | generated | admission | text on disk, prompt id, strategy |
| 1 | `naga::front::wgsl::parse_str` | syntactic | pass, or the first error line |
| 2 | naga validation + Metal pipeline | executable | pass, or the first error |
| 3 | 128 admission probes (`probes.rs`, families A–E sampled), then the 1,024 and the fixed pairs for the top of the population | correct | probes passed, pair criterion ratio |
| 4 | paired timing against shipped on 3 held-out shapes, 3 rounds, random order | fast | geometric mean of the speed-up, its worst shape (the robustness column) |

Selection: the highest rung first; within rung 4 the HFF angle over the
performance columns (speed-up on each shape, worst-shape speed-up, spread
over rounds), all normalised against the shipped kernel; rung 3 within
itself by probes passed. A correct champion is never displaced by a
candidate below rung 4. Probes passed is the gradient for repair, not
for performance.

## 2. The model calls

`claude -p --model sonnet --output-format json` (Claude Code's
non-interactive mode; verified available on this machine). No sampling
temperature is exposed, so diversity comes from the prompt: a strategy
drawn per call from {thread per cell (the baseline), register block 2×2,
register block 4×4, workgroup tile of items in shared memory, vector
loads of four columns, symmetric half with mirrored writes, split
accumulators, unrolled item loop, two of the above combined, "something
unlike the shipped kernel"}, plus a random seed phrase. The prompt
carries the contract verbatim: the three bindings and `Cfg`, item-major
Φ, the band semantics (`r_rows`, `r_off`, read-add-write into `h`), the
dispatch rule (`ceil(d / (x·bx)) × ceil(r_rows / (y·by))`, so the kernel
states its workgroup size and cells per thread in a fixed header comment
the harness reads), and the shipped kernel as the reference text. The
reply must be the kernel only; the harness strips fences.

Model edits (mutation B): the kernel text, its rung and measurements, one
instruction drawn from {reduce memory traffic, change the accumulation
strategy, tile differently, vectorise, remove work on the symmetric
half, fix this error: <diagnostic>}; the reply re-enters at rung 1.

Crossover (mutation C): two parents' texts with their measurements and
the instruction to combine the named strategy of each; re-enters at
rung 1. Typed subtree crossover in the kingdom's tree form is used when
both parents read through the reader; the model-mediated form otherwise.

Rewrite rules (mutation A): the reader cuts the kernel at its stores and
conditions, `mutate.rs` proposes equal-by-rule variants of the arithmetic
(the e-graph family), the folded rebuild writes the text; re-enters at
rung 1 (naga has refused reorderings before; the probes catch the rest).

Each call's prompt and reply are written to `logs/HGP/<run>/calls/<id>.{prompt,reply}.txt`
with its parentage; nothing is ever paraphrased in the ledger.

## 3. The loop

Population 32. Generation 0: 32 fresh calls. Each later generation: the
rung-4 elites kept (up to 8), and 24 offspring by operator probabilities
A 0.35, B 0.35, C 0.15, fresh 0.10, repair 0.05 (one repair per invalid
candidate at most), recorded per offspring so the rates of §4 are
measured per operator. 500 offspring evaluations in all (about 16
generations), three seeds of the run if time allows.

## 4. Measurements and expectations (before any run)

| measurement | expected |
|---|---|
| of 32 generation-0 kernels: parse; validate and compile | 28–32; 24–30 |
| pass the 128 admission probes; pass all 1,024 and the pair criterion | 10–20; 5–15 |
| correct and faster than shipped on the held-out shapes, generation 0 | 1–5 |
| offspring that are correct, by operator: rules, model edit, crossover, fresh, repair | 90 %, 40 %, 20 %, 25 %, 30 % |
| best geometric-mean speed-up after 500 evaluations | 2–6× (a tiled register-blocked Gram is standard) |
| worst-shape speed-up of the champion | within 0.7 of its mean |
| model calls per correct kernel; GPU seconds per correct kernel | 3–6; 60–200 |
| wall clock of the experiment | 4–6 h at 20 s a call and the GPU at today's reduced speed |

## 5. Constraints and safety

One GPU process; every candidate under the static cost bound of
`wgsl_gram_evolve` and the watchdog (a build over 50× the shipped build
exits the process naming the kernel, the runner restarts the loop from
the ledger); the GPU must first be back at its measured speed (shipped
build 2.35 s at the bench size, measured 15:43; 8.4 s at 19:14 after the
afternoon's hang), which needs a restart of the machine. Generated
kernels are untrusted text: they are compiled and dispatched only through
the harness, never written into any crate.

## 6. Tasks

Every task follows `~/dev/minkymorgan/qdrant/docs/Rules.md` (no stubs,
real runs only, unit tests, factual commits, timestamps, warnings fixed,
a bad result named). The wider project, for every task: the HGP
hypothesis tested on the Gram kernel of the Blue World stack, with
fuller's WGSL kingdom (harness, probes, reader, rules, renderer) as the
grading and variation machinery. Tools: Claude Code CLI (`claude -p`),
naga, wgpu/Metal, the fuller examples.

1. **Read docs/Rules.md, then: the caller and the ladder.** Purpose: one
   generation measured. Architecture: `examples/hgp_gram.rs` (fuller,
   `gpu,wgsl`): the contract prompt with strategies, the CLI call with
   JSON output, the stripping of the reply, rungs 1–3 with the 128
   admission probes, rung 4 timing on three held-out shapes, a ledger
   row per candidate, prompts and replies on disk. Tests: the header
   comment parses; a fenced reply strips; the shipped text climbs to
   rung 4; a reply with a syntax error stops at rung 1 with the line
   named. Done when generation 0's rung counts are in the README of
   this plan beside the expectations of §4.
2. **Read docs/Rules.md, then: the three mutations and the loop.**
   Purpose: the evolutionary claim. Architecture: §2's operators A, B,
   C plus fresh and repair with the probabilities of §3, parentage in
   the ledger, elites by rung then HFF, 500 evaluations, three seeds.
   Done when the per-operator correctness rates and the champion's
   speed-up table are in the README.
3. **Read docs/Rules.md, then: the champion in Blue World.** Purpose:
   the production number. Architecture: `PLAN_wgsl_gram_superopt.md`
   task 4 unchanged (the switch in `generator.rs`, the smoke test, three
   interleaved production builds). Done when the production table with
   intervals is in both READMEs.
4. **Read docs/Rules.md, then: the report.** The table of §4 against
   measurements, 200 words at most, and the answer to the research
   question: did model diversity plus selection beat either alone (the
   model's generation-0 best, and the rule-only search of today).
