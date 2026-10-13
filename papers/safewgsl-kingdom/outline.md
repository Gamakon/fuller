# The safeWGSL kingdom paper: outline and open questions

Text: `papers/safewgsl-kingdom/draft.md` (2026-10-13). House style: `papers/geneframe-gecco.tex`.

## Outline

**Context (one page).** Objective: evolve WGSL compute kernels with GEP; correctness a constraint, speed the objective. Approach: a kingdom in the geneframe whose rows are the whole WGSL compute language, duals that make crashing kernels unrepresentable, a layered checker, typed variation. Unique feature: two-column typing, `slot.form`, applied to values and to structure. Status: 183 duals, 1,529 templates, 181/184 constructs, 1,000/1,000 random kernels compile and read back, 1,000/1,000 through the genome, 123/1,000 AST-identical, 16 rules, 12 rewrites, one rule-driven run that climbed. Why: a bad kernel stops the machine, not the individual.

1. **Introduction and contribution.** GEP and the geneframe; the three facts of this kingdom (effects, danger, drifting measurement); five contributions; vocabulary and the names of the parts.
2. **The kingdom.** Duals (value, provenance, structure forms; fallbacks); rows and signatures (classes, K = 4, generated tables, composites); coverage (181/184, the fills); the interface; the master symbol table and typed Karva genomes (head, tail, `name@T061`); the typed projection as a total decode; the homeotic tail and the population design.
3. **Safety by construction.** What legal means; unrepresentable by type (ownership, thread ids, masks, bounded loops, barriers, no wrap, adapter); the six-layer checker and its one-day ratchet; the u32 wrap decision and Codex's review; the bugs (two offset races, the GPU sweep's five, Metal's matrix negation and degrees, the unbounded loop, the 2^48 unfold, the missing grid header, the skipped barrier, every refusal a bug).
4. **Round trips.** Text-AST-text (the parser and emitter, the corpus); AST-text-AST (the completeness ratchet, the canonical form's nine rules); AST-genome-AST (typed tokens, 1,000/1,000, the empty-kernel draw bug).
5. **Variation.** Point mutation (16/92 and why); Karva positional crossover and the 2026-10-11 tail-slot problem; typed subtree crossover; knob mutation over first-class parameters; the sixteen structural rules; the twelve generic rewrites; the engine (islands, pump, cohorts, random immigrants).
6. **Fitness.** The Gram problem; the 128-case constraint and the reference pairs; paired timing, drift, five rounds; the HFF angle; the closeness band for infeasible individuals; the cost bound and the watchdog.
7. **Where it lives.** File table.
8. **Limitations and future work.** Search-space size (the random-start runs); the round trip not yet an identity; beam search and auto-tuning by the rules; symmetric-half as a dispatch-plan change; mined rewrites not wired; interfaces; timing noise; the population-wide homeotic rows.

**Appendix A, highlights.** A.1 fit_RA_rules (table per generation, the three-step lineage, operators by outcome). A.2 the two random-start runs (0 feasible, closeness angle trajectories, what they found). A.3 measurements of the representation. A.4 the model-seeded baseline.
**Appendix B.** Nine figures, twelve tables, each with its data source.

## Open questions for Andrew

1. **The quoted speed-ups (resolved 2026-10-13).** The paired re-timing is `logs/HGP/001/fit_RA_rules/retime_paired/` (six interleaved rounds, three shapes). The paper quotes the medians over the rounds, the fitness's own rule: g000_r7 8.88 / 8.44 / 8.30x, g004_r31 10.53 / 10.20 / 9.84x, faster in 6 of 6 rounds; the per-round table is in Appendix A.1. One check for Andrew: the team lead's summary gives 102.4 / 103.4 / 107.7 ms and 10.46 / 10.37 / 10.00x for r31, which match neither the medians nor the means of `gate.log`'s measured-ms column (medians 99 / 102 / 111 ms); the paper uses the file's values.
2. **Title and the kingdom's name.** "safeWGSL" is the name in the tables and the code; the earlier README calls the kingdom "WGSL, after the language". Is the paper's kingdom "safeWGSL", or is safeWGSL the second version of the WGSL kingdom? The paper uses safeWGSL throughout.
3. **Which results are the highlights.** The rule-driven run climbed from model-written seeds; the random-start runs found nothing. Should the paper lead with the climb, with the representation measurements (1,000/1,000 compile, read back, survive the genome), or with the bug record? The paper leads the status paragraph with the representation.
4. **Rules and correctness.** Codex's finding 3 asks for preconditions over an explicit effect, alias and barrier model; today each rule's preconditions are its own, tested on the corpus and the device. Is that the paper's claim ("every child through the checker, correctness by the fitness"), or should the paper promise the effect model?
5. **The GECCO vocabulary and the code's words.** The code names statement depths L0..L4 and C0..C4 (a word the paper may not use), test kernels in `coverage_probes.py`, and seed kernels by a word the paper may not use. The paper says "statement context" and "depth", "construct test kernels", and "model-written seed kernels". Confirm these are the words you want in print.
6. **Authorship and venue.** Single author as in the geneframe paper, Gamakon Ltd.? Same venue (GECCO, acmart sigconf)? The text is Markdown; the LaTeX conversion waits on this.
7. **The population design.** Should the paper describe the population-wide homeotic rows (designed, measured on the host, not built into the loop) as future work only, or as part of the design with its measurements from `PLAN_population_dag.md`?
8. **Codex as a named reviewer.** The paper names Codex's adversarial reviews (the plan, the overflow decision) because the commits do. Keep the name in print, or say "an adversarial review"?
9. **The mined rules.** 235 candidates, none wired. Include the top-ten table in the appendix, or hold the mining for its own paper once the clean-example method has run?
10. **The seventeenth rule.** Symmetric-half is excluded by ownership; the paper presents it as a dispatch-plan change. Is the dispatch plan in scope for this paper or the next?
