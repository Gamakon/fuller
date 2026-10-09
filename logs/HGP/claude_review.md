# Adversarial review: docs/PLAN_hgp_gram.md (1d9bccd)

Reviewer: Claude (read-only). Read in full: the plan, src/wgsl/{gram,probes,render,mutate,reader,scaffold}.rs, src/population_dag.rs, examples/wgsl_gram_evolve.rs, docs/PLAN_wgsl_gram_superopt.md, kingdoms/wgsl/gram/shipped.wgsl; spot-read chromosome.rs:260-300, oracle.rs error scopes, wgpu 0.20 `ErrorFilter`, `claude --help`.
Plan lines are cited as `plan:N`.

## 1. What cannot work as written

1.1 **No static cost bound exists for model text.** `render::cost` takes a `render::Tree` and keys on renderer row names (`loop.items`, `loop.tile`, `load.*`; render.rs:759-803). Model kernels are text; the reader yields `reader::Node` roots (`index.mul`, `load.buffer.phi.#`) with loops only as statement paths (reader.rs:147-161). No reader-to-`render::Tree` bridge exists (the only `render::Tree` use outside render.rs is gram.rs:24). plan:37 and plan:97-98 ("the cost bound … reused") are false for every model-written candidate.

1.2 **Mutation A cannot change anything that makes the Gram faster.** `mutate.rs` rewrites only f32 `arith.*`/`trig.*` regions (mutate.rs:42-59, `mapped` at 93). u32 index arithmetic reads as `index.add`/`index.mul` (reader.rs:1114 test) and is opaque. The rebuild walks the ORIGINAL function's statements (scaffold.rs:4-7, 487-637) and clones the module, so workgroup size, loops, locals, workgroup arrays and memory layout are fixed (scaffold.rs:104). In the shipped kernel the only region with `min_ops = 2` is `acc + phi[a]*phi[b]`; its Algebra variants are commutations and reassociations. None of the strategies of plan:61 is reachable by A. The "strong editor given a starting program" premise (plan:11-12, plan:66) is not what the code does, yet A gets 0.35 of the offspring budget (plan:68).

1.3 **A's pipeline adds an unrelated failure mode.** plan:68 goes through `chromosome_with_roots` then `rebuild_folded` (scaffold.rs:73), which folds shared subtrees, places `let def_*` definitions and can refuse ("before its definition was emitted", scaffold.rs:905). A single-region rewrite needs only `rebuild(kernel, roots)` (scaffold.rs:58). The `--variants` share subsets (plan:68) are a CLI flag of `examples/wgsl_read_kernel.rs:43-51, 218`, not a library call; `unfold_slot` is homeotic.rs:251. CSE subsets are speed-neutral after the Metal compiler's own CSE.

1.4 **Crossover in tree form between two model kernels is unsound.** `chromosome_with_roots` requires the same root count as the scaffold (chromosome.rs:278-280), and roots name per-kernel locals and lets by arena index (`local.acc@3`, `let.c@17`; reader.rs:696, 826, 870). Worse, `Builder::local` documents "the index selects; the name is checked" but never checks the name (scaffold.rs:777-782). A root from kernel B rebuilt in kernel A binds A's local #3 silently. If the types match, the result validates and computes something else. plan:63 must be text-only (model crossover) until names are checked and alpha-renamed.

1.5 **"Symmetric half" is infeasible under the banded driver.** The accumulator is band-local `r_rows × d` (gram.rs:517-520; shipped.wgsl:3, 16-17). A mirrored write `(c, r)` with `c` outside `[r_off, r_off + r_rows)` has no buffer. At bench shapes row_bands = 1 (superopt §1a, line 111-113), so such kernels pass families A/B/C/E and the timing shapes. They fail only the 25 family-D probes with two row bands (probes.rs:92, `(n/25)%4 == 3`) and the d = 8,192 extrapolation pairs, which have two row bands. Production d = 8,192 has two row bands (superopt line 113). The strategy (plan:61) and the instruction "drop the symmetric half's work" (plan:62) steer the model into a trap. Either remove them or put the band constraint in the contract. At most the diagonal block of each band can be halved.

1.6 **Rung 2 names a private function.** `Gram::compile_with` is private (gram.rs:458); plan:49 names it. The usable path is `compile_rendered` with a hand-built `Rendered { text, grid, helpers, masked }` (render.rs:166-176, gram.rs:453). `Gram::compile` sets `bx = by = 1` (gram.rs:449). Nothing checks the `// grid x y bx by` header (plan:59) against naga's `@workgroup_size`. Take x and y from naga and only bx and by from the header, and refuse a mismatch at rung 2.

1.7 **The reused timing code breaks the house rule and is ambiguous about its clock.** `timing()` is `#[allow(dead_code)]` (wgsl_gram_evolve.rs:300), which CLAUDE.md and Rules.md forbid. It returns `gpu_ms` (312, 553-557). The `Build` docs call `gpu_ms` "the search's time" (gram.rs:412-416). The test comment says it "reads 0 for short passes on Apple silicon, which is why the search uses compute_ms" (gram.rs:715-717). `timing()` errors on `ts <= 0` (315). Rung 4 must name its clock and show the clock is non-zero at the rung-4 shapes.

1.8 **Mutation A on tiled kernels is untested.** Reading is tested: workgroup memory and barriers appear only in a read test (reader.rs:1284-1312). No scaffold test rebuilds a kernel with `var<workgroup>` and `workgroupBarrier()` (scaffold.rs:1117-1385). `Statement::Barrier` is copied (scaffold.rs:535). `Atomic`, `WorkGroupUniformLoad` and subgroup statements are read as "unread" (reader.rs:600-613) but refused by rebuild (scaffold.rs:631). plan:36's "so model-written tiled kernels read" conflates reading with round-tripping.

1.9 **The model call reads the judge.** The call as specified (plan:39, 59) runs `claude -p` in the repo with default tools and auto-discovered CLAUDE.md. `claude --help` shows `--bare` (skips CLAUDE.md discovery, memory and hooks) and `--tools`/`--disallowedTools`. Without them the proposer can `Read` src/wgsl/probes.rs, the ledgers and previous champions. That contradicts "the model proposes and never judges" (plan:25). `--bare` accepts only `ANTHROPIC_API_KEY` or `apiKeyHelper` and never reads OAuth. The flags are not "verified" until the auth path is.

1.10 **The plan's founding premise contradicts the repo's own test.** plan:8-9 and commit 84a7c2b say the shipped kernel with its multiply broken "fails all" probes. The test asserts `fails > 100 && fails < ps.len()` (probes.rs:177): it fails some and passes others. So the probes do have a slope between right and wrong. That weakens the stated reason the tree search "learned nothing". See 4.1 for a more likely cause.

1.11 **Arithmetic.** (500 − 32) / 24 = 19.5 generations, not "~16" (plan:95). plan:96 does not say what fills the elite slots when fewer than 8 kernels reach rung 4.

## 2. Unmeasured or unfalsifiable

2.1 **The research question has no compute-matched control.** "Model diversity plus selection beats the model alone (generation 0)" (plan:19-20) compares 500 evaluations with 32. Any gain is confounded with 15× more sampling. The fair arm is best-of-N fresh calls, with N equal to the model calls the evolution spends, through the same ladder and timing. No margin for "beat" is stated.

2.2 **The "rule-only editor alone" arm is already decided by the code.** It has no seed population, budget or operator set (plan:20). Applied to the shipped kernel it can only commute `a*b + acc` (1.2), so it reads about 1.0× by construction. It is not an informative comparator.

2.3 **No timing null.** The earlier example timed the shipped tree against the shipped text every generation (commit 7c299f4). The plan drops that control. Without shipped-vs-shipped rounds in the same session, "faster than shipped" has no false-positive rate.

2.4 **The GPU state is not gated.** plan:99-100 records 2.35 s vs 8.4 s for the same build, a 3.6× drift. No acceptance test is defined, for example the shipped build within 10 % of 2.35 s before and after each rung-4 batch. Wall clock and every rung-4 number depend on it.

2.5 **Determinism and reproducibility.** There is no temperature control (plan:40), and "three seeds" (plan:118) do not seed the model. A rerun is reproducible only by replaying saved replies. The restart-from-ledger design (plan:97) must replay texts, not re-call. The number of distinct kernels among 32 calls driven by 10 strategy phrases is unmeasured. Report distinct normalised texts.

2.6 **"Components that matter" is uncontrolled.** "Which components correct kernels carry that incorrect ones lack" (plan:80-81) compares about 10-20 correct kernels with the rest, confounded by strategy and model style. It will always find something. It needs a test against a permutation null, or a causal check: transplant the component and re-run the probes.

2.7 **Task 2b's bar is zero.** Its baseline, the tree search of `wgsl_gram_evolve` from random trees, produced no correct kernel. Any non-zero "correct kernels per 100 evaluations" passes. 5.4 and 5.5 show it cannot be built as written.

2.8 **A probe kind does not test what it claims.** `Kind::Cancelling` claims off-diagonal sums cancel (gram.rs:98). Rows r and −r give 2rrᵀ (gram.rs:224-234), not zero. It does not test exact cancellation.

## 3. Unjustified expectations

3.1 **Rules at 90 % correct (plan:109) is the wrong measurement.** The rewrites are semantics-preserving up to rounding, so near 100 % correct is expected. The missing row is "offspring with a distinct paired time, by operator". For A the expected count is about 0 (1.2).

3.2 **1-5 correct and faster at generation 0 (plan:108) has no cited prior.** It is the quantity being measured. Tiled strategies are the fast ones, and they are the most likely to fail naga's uniformity analysis: `workgroupBarrier()` after an early `return` guard is the common LLM pattern. Expect validate/compile rates to vary by strategy, and report them per strategy.

3.3 **2-6× after 500 evaluations (plan:110) is not derived.** The superopt plan expected 4-12× for the best grid kernel (superopt:310), and that grid was never built (superopt:46). The shipped kernel runs at 117 G/s multiply-adds (superopt:111), far below M3 Max f32 peak, so headroom exists. But the fixed driver forbids split-K: z = 1, one pass per item band, no f32 atomics (gram.rs:530-540). It also forbids cross-band symmetry (1.5). Without a measured positive control (7, step 0) no speed-up range is justified.

3.4 **4-6 h wall clock (plan:113) is optimistic.** 500 admissions × 12.5 s (128 probes at the measured 100 s per 1,024) ≈ 1.7 h. About 20 generations × 8 full evaluations × (100 s probes + pairs) ≈ 5 h, unless memoised across generations, which the plan does not state. Add rung-4 timing: 3 shapes × 3 rounds × 2 kernels × 2-8 s ≈ 36-144 s per candidate. Add about 330 serial model calls at tens of seconds each. That is 7-10 h at the healthy GPU rate and more at the 8.4 s rate.

## 4. Where a wrong or noise-favoured kernel wins

4.1 **The HFF angle is magnitude-blind.** `angle` (wgsl_gram_evolve.rs:197-204) is `acos(Σ(1−x) / (√n ‖1−x‖))`. For any constant column vector `x = (c, …, c)` with c < 1 it returns 0, the best value. Within rung 4 (plan:53-54), a kernel uniformly 1.1× faster on every shape ties with one uniformly 5× faster, and both beat a kernel 5× on two shapes and 2× on the third. The same property applies to the probe fitness: a kernel whose probe columns are all equal scores 0, like a correct kernel. Computed with the example's exact `columns` and `angle` formulas (scratchpad, not the repo):

| kernel (1,024 probe columns) | angle |
|---|---|
| all correct | 0.000 |
| every probe wrong by the same δ = 1/1024, column 0.332 (roughly an all-zero kernel on family A) | 0.000 |
| 90 % correct, the rest at column 0.332 | 0.103 |
| 50 % correct, the rest at column 0.332 | 0.197 |

The fitness ranks a uniformly wrong kernel with the correct one and above a half-correct one. It punishes progress, which is a more likely cause of "learned nothing" than the cliff of plan:5-9. Real wrong kernels are only roughly uniform across probes, so measure the angle of the logged population before relying on this. Rank rung 4 by a magnitude (the geometric-mean speed-up) and use the angle only as a tie-break. Or put a magnitude column in that is not normalised away.

4.2 **Accuracy left the fitness at rung 4.** The only accuracy control is `gate` (wgsl_gram_evolve.rs:262-264): `max_ratio ≤ max(2·shipped_max, 1)`. Inside the gate, a kernel at 1.9× shipped's error and 3× speed beats one at 0.5× error and 2.8× speed. The superopt plan's six quality columns (superopt:154-161) were dropped. Restore at least max ratio over shipped on train and extrapolation.

4.3 **Timed builds are not checked.** Rung 4 (plan:51) times builds on shapes but does not say the H from those builds is checked against a reference. A kernel that special-cases `cfg.d` or `cfg.m` at the timing shapes, which models sometimes do, would be fast and wrong where it is timed. Check every timed build's H, or time only shapes whose reference exists, and require bit-identical H across repeat builds (4.9 covers races).

4.4 **The held-out shapes are selected on.** plan:51 times rung 4 on "3 held-out shapes" and selects on them. plan:19 then reports the champion's speed-up "on held-out shapes". Those shapes are training shapes. The report needs shapes and fresh timing rounds never used in selection.

4.5 **Winner's curse.** Three rounds per shape (plan:51) against a measured run-to-run spread of 12 % (superopt:105), with hundreds of rung-4 candidates. The maximum of noisy medians is biased upward by several σ. Re-time the champion in a fresh session with 5+ rounds and the shipped-vs-shipped null, and report that number, not the selection number. A spread column rewards consistency, and with 4.1 it can dominate speed.

4.6 **Probe shapes are too small for tiled kernels.** A/B/C/E use d = 32, m = 16 (probes.rs:45), and D has d ≤ 40, m ≤ 50 (probes.rs:89-90). A 16×16 workgroup with 4×4 register blocks covers 64×64 cells, so every probe runs in one workgroup per axis. Errors in `workgroup_id`-based offsets are invisible. An item tile of 64 never runs a full tile at m ≤ 50; only the remainder path runs. The pairs (d ≥ 512) catch these, but plan:50 runs pairs only for "the top 8". Rung 4 must require all 1,024 probes plus all training pairs. Add probes at d ≥ 130 and m ≥ 2 × the largest tile.

4.7 **The probe pass criterion is a mean over d² cells.** `distance` averages per-cell error over 1,024 cells, scaled by `max(|H_ref|, 1)` (probes.rs:123-135), with PASS = 1e-5 (probes.rs:139). A single cell can be wrong by up to about 1 % of the scale and pass. Example: probe A with v = 0.125 has one cell of 0.015625. A kernel writing half of it scores 7.6e-6 and passes. Pass should be a per-cell max, as the pair criterion is (gram.rs:350-406).

4.8 **Admission overfits.** A fixed 128-probe admission set (plan:50), plus repair prompts that quote the exact diagnostic (plan:64), overfits the 128 over about 20 generations. Within rung 3 the admitted kernels all score 128/128, so "top 8" is decided by the tie-break of index order (wgsl_gram_evolve.rs:451). Rotate the admission sample per generation from probes(1), and keep probes(2) for validation only.

4.9 **Races pass by chance.** A model kernel doing non-atomic read-add-write on the same `h` cell from several threads (a split-K attempt) gives schedule-dependent results that can pass a probe. Require bit-identical H over two builds.

## 5. What the share-equivalence scan needs that the reader and population_dag lack

5.1 **Cross-kernel alpha-renaming.** Reader leaf names embed source identifiers, arena indices and versions: `local.acc@3@v2`, `let.base@17`, `call.f@7` (reader.rs:127, 171, 190, 423, 878). `PopulationDag::intern` hash-conses by rendered text (population_dag.rs:96-99). `scaffold::canonical` renames by arena position (scaffold.rs:248-338), which also differs between kernels. Two kernels computing `t·d + c` will not share a node. This blocks task 2's report. Without it, "share rate" measures how often `(literal.index (Num 0.0))` and `load.uniform.cfg.d` repeat.

5.2 **Let expansion.** The reader cuts trees at `let` leaves (reader.rs:824-827). A model writing `let base = t * cfg.d; … phi[base + c]` yields `(index.add (Var "let.base@N") (Var "let.c@M"))`; the subtree `t·d + c` never exists. The scan needs lets inlined, kept apart from the sound let-bound form used for rebuild.

5.3 **u32 index AC-canonicalisation.** `c + t·d` and `t·d + c` differ as text. The e-graph rules cover only the Math f32 operators (mutate.rs:42-59); index algebra has no rules.

5.4 **Statement-level components are unrepresentable.** "A tile load into workgroup memory", "a register-blocked accumulation" and "a mirrored store" (plan:75-76) are loops, barriers and store groups. The reader's roots are per-statement expression trees with a path (reader.rs:147-161); the structure lives in the scaffold, which no tree carries. The superopt plan lists reading statements into `block`/`stmt`/`loop` nodes as the "next increment", not built (superopt:247-250).

5.5 **Kernel-level referrer sets.** `uses` counts every path to a node, across and within kernels (population_dag.rs:76, 140-146). A subtree repeated 16 times inside one unrolled kernel looks popular. "Correct carry, incorrect lack" needs a set of kernel ids per node. `report()` returns counts and a histogram, not node texts (population_dag.rs:263-273). `eval` cannot run on WGSL rows: there is no `eval::apply` for `index.add` (population_dag.rs:205). That is fine for the scan but rules out memo claims.

5.6 **Task 2b needs a vocabulary bridge that does not exist.** It needs components as "terminals of the typed sampler" (plan:82-84). `render::sample` draws kingdom rows plus role rows (render.rs:847-894) in the renderer's vocabulary: `load.phi`, fixed registers `f0…f3`/`i0…i3`, `wg_a`/`wg_b`. The reader's vocabulary is `load.buffer.phi.#`, `local.*`, `let.*`. No translator exists, and a component naming a model's local has no counterpart in the register-limited grammar.

## 6. Running untrusted generated kernels on the GPU

6.1 **The cost bound does not apply** to model text (1.1). Termination is unchecked. naga validates structure, not termination: `loop {}` with a wrong break, a `u32` counter compared with `<= 0xffffffffu`, a NaN-controlled loop and nested `cfg.m` loops all validate. The renderer could never emit these: `loop.items` is bounded by `cfg.m` and refuses nesting (render.rs:370-379), and `loop.tile` is bounded by a literal.

6.2 **The watchdog cannot preempt the GPU.** It calls `process::exit` (wgsl_gram_evolve.rs:229-239). The 2026-10-09 run hung the device for 22 minutes (render.rs:751-752). On this laptop the same GPU drives WindowServer, and a memory note records a hard reset from GPU starvation. The plan's only defence for model text is this watchdog.

6.3 **The watchdog is disarmed during compile and armed too loosely.** `score` compiles (wgsl_gram_evolve.rs:281) before it arms (285). A Metal compile that hangs on a pathological shader, such as a 4,096-way manual unroll, is unguarded. The arm is a fixed `deadline_ms(200)·10` = 100 s for a whole probe batch whose healthy cost is about 12 s for 128 probes. Arm per build at a few seconds for probe shapes.

6.4 **Device errors can panic the process.** `Gram::build` has no error scope (gram.rs:478-583). The layout has `min_binding_size: None` (gram.rs:466), so a model `Cfg` with a fifth field, or a fixed-size array binding larger than the buffer, is checked late at bind or dispatch. That error goes to wgpu's uncaptured-error handler, which panics by default; `Device::new` (oracle.rs:302) installs no handler. `compile_with` pushes only `ErrorFilter::Validation` (gram.rs:462). wgpu 0.20 also has `Internal` (verified in the crate source). Whether a Metal-compiler rejection of a naga-valid shader lands in Validation or Internal is unverified. Push both scopes and install `on_uncaptured_error` that returns an error instead of panicking.

6.5 **What is covered.** Buffer reads and writes are bounds-checked by wgpu/naga's Metal policy. Workgroup memory is zero-initialised. Too-large workgroups and workgroup storage fail pipeline creation inside the scope. Barrier uniformity is checked by naga.

6.6 **What to add.**
- A naga-IR pre-dispatch pass that refuses any `Loop` whose exit condition does not compare an induction variable against `cfg.m`, `cfg.d`, `cfg.r_rows` or a literal ≤ 4,096, and refuses nested loops whose bounds multiply to more than `m × 64`.
- Run every candidate first on the smallest probe with a 1-2 s deadline.
- Evaluate candidates in a child process, so the parent logs and survives.
- Never evaluate while another GPU process runs, per the one-GPU-process rule.

## 7. The smallest experiment that would prove or kill the hypothesis

Build no evolution loop yet.

0. **Positive control and timing null (one session, after the restart).** Write one known-good tiled kernel by hand, or take one from a single model call: a 16×16 workgroup, 4×4 register block, item tile in workgroup memory. Put it and the shipped kernel through rungs 1-3 in full: 1,024 probes, 47 train pairs, extrapolation. Then time it paired, 5 rounds, at d = 2,048, m = 65,536, beside shipped-vs-shipped rounds. If a known tiled kernel is under about 1.3× under this driver on this machine, the plan's speed question has no headroom. Stop.
1. **Generation 0 alone.** Make 32 isolated calls with `--bare`, no tools, an empty working directory, and saved texts. Run rungs 1-2, then the full 1,024 probes and all training pairs on every validating kernel, not 128. Time each correct one as in step 0. Report counts per strategy and distinct texts. If zero kernels are correct and faster, one repair pass on all 32 is the only next step.
2. **Operator A on real inputs.** Run the shipped kernel and the best generation-0 kernel through `read`, `mutants_of_roots(2, 8, 4, true)`, `rebuild`, write. Count distinct WGSL texts and time three mutants paired against their parent. Expected: zero distinct speeds. That would remove A from the speed search.
3. **The scan's precondition.** Take two generation-0 kernels that both compute `t·d + c` and print the reader's subtree text for each. Unequal texts confirm section 5 before task 2 is built.
4. **Only if steps 0-1 pass, the comparison.** Run best-of-N fresh calls against the evolutionary loop at the same model-call and GPU budget, say N = 100 each. Re-time the champion on unseen shapes and fresh rounds against the null. The hypothesis holds only if the loop's re-timed champion beats best-of-N by more than the null's spread.
