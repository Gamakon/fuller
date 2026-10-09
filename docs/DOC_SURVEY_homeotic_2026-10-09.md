# Document survey: what the homeotic-chromosome design changes (2026-10-09)

Review only, by a subagent that read every candidate document in fuller
and phylu in full (Rules.md: no sampling). The design under review is
`PLAN_population_dag.md` §1. Groups: (1) contradicts the design and
must change; (2) silent on it and should state it; (3) unaffected.

## (1) Must change

- fuller `docs/PLAN_population_dag.md`: self-contradictions on spreads
  vs paired CIs and on arms/laws between §2 and §4; dangling superopt
  numbering; href identity unstated; "open reading frame"/"tail" clash
  with Karva terms. FIXED 2026-10-09 (§2d, text).
- fuller `src/homeotic.rs` //! l.1-34: href<t> = per-chromosome slot,
  forward by slot, EMPTY_SLOT. New: href names a population homeotic
  row (stable id); acyclic by level; Fold adds/rewrites/re-sorts.
- fuller `docs/Geneframe_Homeotic_Genes_design.md` §7 (rev 3): per-
  chromosome slots, last-to-first evaluation (l.500-506), per-chromosome
  gate (l.542-546), tail 2-4 (l.581); l.561-565 says population sharing
  "is not what the homeotic tail is for": reversed. Add §8 rev 4.
- fuller `docs/Evolving_DAGs_with_GEP.md`, `_v2.md`, `_v3.md`: say
  FoldToDag replaced the homeotic design; fold never edits in place; v3
  "genotype layout does not change". New: homeotic design live, second
  row kind, Fold rewrites ordinary rows; v1 a superseded banner.
- fuller `docs/Saturated_Share_Equivalence_v1.md` §2 l.71-73, §5 steps
  4-9: second dispatch, chromosome gate, per-chromosome GeneRef
  write-back. Saturation only supplies candidate equivalences to Fold.
- fuller `docs/FRONTIER_EVOLUTION.md` §4.1 table, l.391-392, l.458-460:
  add "Population DAG: designed, not built"; nested definitions =
  homeotic rows by level; checkpoints carry homeotic rows; replacement
  dictionary = persisted tail; the 63 % figure.
- fuller `papers/geneframe-gecco.tex`: abstract l.57, contribution
  l.149 "across the genes of one chromosome"; Table 1 l.246 and l.722-754
  two-pass. Add population homeotic chromosomes as design; "level
  schedule"; limits l.881-889 and conclusion l.937; Fig. 1 l.212.
- fuller `docs/PLAN_wgsl_superopt.md`: §5 l.155 and task 7 l.234 non-
  overlapping spreads; §3 l.106 and task 1 l.186 point at a moved §7.
- fuller `kingdoms/wgsl/README.md` l.557 "superopt §8" → population_dag
  §2; stale l.3 status, l.570 vs l.575, l.300 vs l.184.
- phylu `README.md` §Homeotic genes l.48-87: per-chromosome tail, rate
  per individual, one last-to-first pass, per-chromosome gate. Rewrite.
- phylu `docs/PLAN_fold_to_dag_generef.md` corrections 3,5, step 4,
  step 6, l.550: GENEREF_k chromosome-local, K=8, shared_base. Banner.
- phylu `docs/PLAN_saturated_fold_cost_gating.md` §6, §7, no nesting:
  superseded; the 177-366 op-equivalent dispatch cost is why sharing
  goes population-wide.
- phylu `docs/GENE_EXPORT_FORMAT.md` l.32-57 ordinary genes only; l.92
  bit-identical reload breaks once genes carry hrefs. Add a homeotic
  array (id, genome, rnc, root dual, referrer count, use rank, level),
  export_version 2.
- phylu `docs/PLAN_arbitrary_arity.md` l.106-107: GeneRef arg0 "shared-
  slot index" → homeotic row id.
- phylu `src/evolve/fold_to_dag.rs` //!: chromosome-level, stale name,
  k = position, "NOT wired, no persistence".
- phylu `src/evolve/evaluations.rs` //! l.79-81: Fold-rewritten rows not
  rescored or charged.

## (2) Should add a paragraph

fuller `src/geneframe.rs` //! (two row kinds); `CLAUDE.md` (homeotic.rs,
src/wgsl, the two-row geneframe, the current programme; "In flight"
stale); `docs/SPLIT_fuller_phylu.md` and phylu's copy (name the share
Fold apart from constant fold; fuller owns arena, memo keys, level
schedule; checkpoints carry homeotic rows; the determinism test; phylu
copy §7j); `docs/PLAN_wgsl_lineage.md` l.227-232, l.369 (placement as
metadata on a homeotic row; memo key carries lineage); `docs/
PLAN_wgsl_kernel_reader.md` l.244; `docs/SPEC_fuller_gpu.md` §6a, §3;
`kingdoms/README.md` l.61-62; `kingdoms/tsr/README.md`; `nucleotable/docs/
DataModel.md` §3.6, `README_nucleotable.md`, `EvolutionSQL.md` §1/§3/§7
(row-kind column, root type, referrer count, data version, memo slot;
reconcile "append-only" with counts and re-sort); `docs/
PLAN_saturated_share_equivalence_v1.md`; phylu `CLAUDE.md` l.31-33;
`docs/TYPED_ENGINE.md` §1, §3; `docs/PLAN_engine_on_gpu.md` l.51-59,
step 4; phylu //! headers: checkpoint.rs, mod.rs, device.rs,
resident.rs, gene_export.rs, diversity.rs, wgsl_sample.rs.

## (3) Checked and unaffected

fuller: PLAN_fuller_gpu, BRAINSTORM_tower_detector, CR_*, DESIGN_tsr_vs_
sr_srbench_runpod, physics_prior_rules, PHYSICS_MUTATE_API, PLAN_runpod_
trial_ab_tsr, SPEC_typed_transcendental_depth, USAGE, symbolic-regression
README, tsr/symbols, wgsl TYPES and symbols, nucleotable GeneFrameDesign
and KingdomMatrix, Fuller_System.tex (bar the term clash). phylu:
FINDINGS_universal_fitter, Things_to_Improve, PLAN_assertion_ensemble,
EXPERIMENTS, PLAN_regex_*, PLAN_multityped_gep, PLAN_srbench_competition,
PLAN_typed_evolution, STUDY_near_misses, SPEC_typed_transcendental_depth,
the remaining 30 src/evolve headers.

## First five, in order

1. fuller `docs/PLAN_population_dag.md` (done 2026-10-09)
2. fuller `src/homeotic.rs` //!
3. fuller `docs/Geneframe_Homeotic_Genes_design.md` (rev 4, retract l.561-565)
4. phylu `README.md` §Homeotic genes
5. phylu `docs/GENE_EXPORT_FORMAT.md` (the homeotic array)

## Contradictions already present today

- Evolving_DAGs v1-v3 say FoldToDag replaced the homeotic design;
  Homeotic rev 3 builds on it and cites neither.
- Evaluation order: phylu README l.53-55 and GECCO l.742 say one/two
  ordered passes; PLAN_arbitrary_arity and kernel_reader say per-level
  dispatch (built 6fa2e4f8).
- Nesting: cost_gating says a definition cannot hold a GENEREF;
  arbitrary_arity says it can, built.
- phylu README l.84-87 says the tail is not built; wgsl_sample.rs
  tail_slots and fold_to_dag::definition_levels exist.
- Parsimony: FRONTIER l.233 "executed cost for parsimony" vs Homeotic
  l.70-74 "no parsimony".
- Arity limit: GECCO l.359, l.908 and wgsl/symbols.md l.7 say two;
  kernel_reader §2a delivered four.
- SPLIT copies diverge (phylu has §7j).
- kernel_reader internally: depth capped at 1 vs unbounded.
