# The user flow: tune a shipped WGSL kernel on this machine (Andrew, 2026-10-14)

## Context
**Objective.** A user hands us a working GPU program. We capture its WGSL kernel, evolve a version that computes the same results and runs faster **on this hardware**, and hand back that kernel as WGSL, ready to drop into the program.
**The success measure.** One number per problem: the evolved kernel's time against the shipped kernel's, in milliseconds, measured by paired interleaved timing on this machine at the problem's timing shapes, with the evolved kernel passing the problem's oracle. Success is a lower time with a confidence interval that excludes 1.0×.
**The guarantee.** The shipped kernel is always in the population, so the run's best can never be worse than the program as shipped: it is its own hall-of-fame winner until something beats it.

## The flow
| step | what | output | gate (measured, every run) |
|---|---|---|---|
| 1 | Capture the program's WGSL kernel | `problems/<p>/shipped.wgsl` | the text compiles on this device |
| 2 | Parse into the kingdom's AST, encode as Karva genes | the shipped genome | decodes to the same AST; **emitted again, computes the same values (bit for bit on the CPU executor) and runs as fast as the original on this device (paired, within timing noise)** |
| 3 | Produce neighbours: the population | the shipped genome, its rule and rewrite children, model-written variants (Sonnet, seeded with the problem's contract) | every member legal (7 checker layers) |
| 4 | Evolve on this machine | generations scored by the problem's oracle and timed here | the shipped genome stays in the population (elite) |
| 5 | The final Karva genome | the run's best | it passes the oracle on every case, held-out shapes included |
| 6 | Emit the final WGSL kernel | `problems/<p>/tuned.wgsl` | compiles; same interface (bindings) as the shipped kernel |
| 7 | Measure the speed-up on this hardware | the report: ms shipped vs tuned per timing shape, paired, with a 95 % interval | the claim is only what this measurement shows |

## Why step 2's speed gate matters
If the kingdom's own rendering of the shipped kernel runs slower than the shipped text, every individual starts handicapped and the original is not truly in the race. Measured 2026-10-14 on nluca's generated_step: the original 1.2 / 25.0 / 25.0 ms, the kingdom's rendering 6.1 / 37.1 / 28.7 ms. That is a bug in the emitter, and it blocks the flow until the rendering matches the original.

## One command
`wgsl_tune --problem problems/<p>` runs steps 1–7 and writes `problems/<p>/tuned.wgsl` and `problems/<p>/report.md`. Until it exists, the steps run by hand and the report is written from their logs.
