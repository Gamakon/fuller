# Evolving DAG-Shaped GEP Chromosomes using Homeotic Gene References

Status: design, partly built. Revision 3 (2026-10-07) adds §7, the homeotic
tail: Andrew's chromosome layout in which the shared partials live in a tail of
genes behind the programs, and the share finder's job becomes filling that
tail. Revision 2 incorporated external reviewer feedback (named-symbol design
replaced by positional gene references; parsimony framing removed, replaced by
the actual selection mechanism: HFF tournament). §§1–6 are revision 2 as
written; where §7 supersedes them it says so.

## 1. Context — what problem this solves

fuller/phylu evolve GEP chromosomes: each individual is one or more *genes*,
each gene a karva-encoded AST (`src/karva.rs`), decoded independently and
evaluated independently on the GPU (`phylu/src/evolve/decode.wgsl`,
`fuller/src/gpu_eval.rs`). Multiple genes in one chromosome are combined only
**after** each is fully evaluated, by a *linker* (avg/mul/add over finished
per-gene predictions — `GeneLinker` in `fuller/src/chrom_score.rs:326`,
`linked()` in `phylu/src/evolve/score.wgsl:72`). No gene can see another
gene's internal structure or reuse its sub-computation; genes are siblings,
never a DAG.

Separately, fuller's rewrite engine is `egglog` — an e-graph, which is
internally a DAG of equivalence classes (congruence closure over shared
subterms). Every extraction path currently collapses that DAG back to a flat
tree before emitting a `Math` term (`extract.rs`'s `termdag.to_string(term)`),
discarding any cross-term sharing structure egglog already discovered for
free.

The motivating paper is Neumann & Moerkotte, *Generating Optimal DAG-Structured
Query Evaluation Plans* (CSRD 2009; copy + extracted text in this `docs/`
folder: `PAPER_neumann_moerkotte_dag_query_plans.{pdf,txt}`). Two ideas from
it map directly onto this design, and the paper is explicit that they are
**separate layers, not one**:

- **Share equivalence (≡S)**, §4.2: two subplans that compute the same output
  can be merged into one physical operator, referenced by all consumers via a
  rename. This is a *semantic* fact — it says folding is **legal**, nothing
  about whether it's worth it.
- **Cost-based dominance**, §4.4–4.5: whether a shared plan is actually
  *better* is decided separately, by comparing real costs
  (`d₁ dominates d₂ iff costs(d₁) < costs(d₂) ∧ S(d₁) ≤ S(d₂)`). Neumann's
  sharpest result (§4.4, Fig. 5) is that DAGs have **no optimal substructure**:
  a locally cheaper non-shared plan can lose to a locally more expensive plan
  that enables sharing and wins overall. Semantic equivalence alone never
  decides this — only cost comparison across full candidates does.

Separately, their evaluation section uses the **magic-set transformation**
(Mumick/Finkelstein/Pirahesh/Ramakrishnan 1990) — synthesizing a new, derived,
query-scoped relation purely so that sharing which was previously invisible to
the optimizer becomes checkable. Their Query 2 result (Fig. 9): DAG + magic
set cuts runtime 70% versus the tree plan, specifically because magic sets
*expose* sharing that plain DAG construction alone misses — they deliberately
choose a plan **shape** that makes sharing visible, even at the cost of local
optimality elsewhere.

**This document proposes the GEP analogue of both layers, kept separate, as
Neumann keeps them separate:**

- The **e-graph's job is only the first layer**: scan an individual's genes,
  find share-equivalence opportunities, and emit legal fold candidates. It
  never decides whether a fold is good.
- **Selection is the second layer, and it already exists**: every fold is
  emitted as an ordinary additional candidate individual, entering the same
  HFF (hyperspherical fitness function) tournament as every other candidate —
  scored on whatever dimensions a run declares (accuracy, validation
  generalization, evaluation cost, anything else), compared via
  incomplete-beta/CDF normalization so heterogeneous, unbounded-dimensional
  fitness still ranks coherently. A fold wins or loses exactly like any other
  mutation. No new selection logic is needed, and no parsimony/size penalty is
  introduced anywhere by this design — fuller/phylu do not use parsimony as an
  overfit guard (validated empirically: the regex kingdom's winning head size
  and the SRBench-beating runs both use large heads with no size penalty;
  overfit is caught by validation-set generalization, not size).

## 2. What exists today (verified against source, with citations)

### 2.1 The static symbol table — `src/geneframe.rs`

The **master symbol table** is static, loaded once, keyed by `semantic_id`:

```rust
// src/geneframe.rs:140
pub struct Symbol {
    pub kingdom: String,
    pub symbol: i64,          // >0 = function, <0 = terminal (nucleotable convention)
    pub symbol_name: String,
    pub alias: String,
    pub semantic_id: String,  // the Math op fuller rewrites on
    pub arity: Arity,
}

// src/geneframe.rs:153
pub struct SymbolTable {
    rows: Vec<Symbol>,
}
```

```rust
// src/geneframe.rs:115
pub struct Arity {
    pub inputs: BTreeMap<Ty, u32>,
    pub outputs: BTreeMap<Ty, u32>,
}
```

`Ty` (`geneframe.rs:31`) is an extensible enum of value types (`F`, `I`, `S`,
`B`, `A`, `L`, the transcendental-depth ladder `T1`/`T2`, plus per-kingdom
types for NLP/regex/etc). A **kingdom is a query**: `SymbolTable::kingdom(name)`
(`geneframe.rs:167`) filters `rows` by `kingdom == name` — there is no
separate kingdom object, just a filter predicate over one flat table.

This table is built once by `master_table()` (`geneframe.rs:204`) and
`typed_depth_table()` / `regex_table()` for the other kingdoms. Every row is
hand-authored at compile time. **This design adds a fixed, compile-time row
family to this same static table** (`REF_0..REF_{G-1}`, §3.1) — no dynamic,
per-run, or per-individual table is needed, which is the main simplification
over the original (reviewer-superseded) scratchpad proposal.

### 2.2 The karva encoder/decoder — `src/karva.rs`

```rust
// src/karva.rs:28
pub struct PsetSpec {
    pub variables: Vec<String>,
    pub functions: HashMap<String, FunctionSpec>,  // token_name -> spec
    pub rnc_values: Vec<f64>,
}

// src/karva.rs:18
pub struct FunctionSpec {
    pub semantic_id: String,
    pub arity: usize,
}

// src/karva.rs:41
pub enum Token {
    Func(String),
    Var(String),
    Num(f64),
}
```

Conversion is **bidirectional and confirmed**:

- Decode: `karva_to_terms(head: &[Token], tail: &[Token], pset: &PsetSpec) -> Result<String, String>` (`karva.rs:171`) — BFS/level-order GEP walk, renders a `Math` s-expression.
- Encode: `terms_to_karva(term: &str, pset: &PsetSpec, rng_seed: u64) -> Result<(Vec<Token>, Vec<Token>), String>` (`karva.rs:450`), built on `terms_to_karva_sized` (`karva.rs:476`) — BFS the parsed `Math` tree into level-order karva tokens, pads the tail per the GEP rule (`tail_len = head_len * (pset_max_arity - 1) + 1`).
- Round-trip is unit-tested (`karva.rs:669`, `inverse_round_trips_through_math`).

**Implication for this design**: a `REF_j` terminal is an ordinary
`Token::Var`-shaped entry — no change to either converter function. It is
addable to a kingdom's `PsetSpec.variables`/`functions` exactly like any other
terminal.

### 2.3 The `Math` datatype and its e-graph — `src/expr.rs`

```
(datatype Math
    (Num f64) (Var String)
    (Add Math Math) (Sub Math Math) (Mul Math Math) (Div Math Math)
    (Neg Math) (Sin Math) (Cos Math) (Log Math) (Exp Math) (Sqrt Math)
    (Abs Math) (Tanh Math) (Tan Math) (Asin Math) (Acos Math)
    (Pow2 Math) (Pow3 Math) (Pow Math Math) (Inv Math)
    (ProtectedSqrt Math) (ProtectedLog Math) (ProtectedExp Math)
    (ProtectedInv Math) (ProtectedDiv Math Math)
    (ProtectedAsin Math) (ProtectedAcos Math))
```
(`src/expr.rs:30`, `MATH_DATATYPE`)

`math_egraph()` (`expr.rs:163`) builds a fresh `egglog::EGraph` with this
datatype loaded, no rules. Saturation (`extract.rs`, not reproduced here) runs
bounded algebra+powers rules over whatever terms are asserted. **Nothing in
assertion or saturation is single-root-specific** — multiple independent
`Math` terms can be asserted into one `EGraph` and saturated together; egglog's
congruence closure will merge any two terms' subterms that become provably
equal, across root boundaries, for free. The only place this sharing is
currently thrown away is at extraction (`termdag.to_string(term)`, called once
per root).

**Two staged modes, per reviewer feedback (§5 below):**

1. **Syntactic sharing (cheap, first)**: pure hash-consing across a
   chromosome's genes' raw `Math` terms — no saturation at all. Measures how
   often literally identical subterms occur across genes.
2. **Semantic sharing (saturation, second)**: run the bounded algebra+powers
   schedule (never distribute/trig/rational co-saturated, per the existing
   non-confluence rule), then look for cross-root e-class collisions. This
   finds sharing syntactic hash-consing misses (e.g. `a*b` vs `b*a`), at
   e-graph cost. Measuring the *delta* between (1) and (2) tells us directly
   whether the e-graph is earning its keep for this purpose.

Within mode 2, take a **minimum subterm size** and prefer **maximal** shared
e-classes — otherwise trivial common patterns like `(Mul x x)` get folded
everywhere, which is noise, not a useful opportunity.

### 2.4 GPU opcode lowering — `src/gpu_eval.rs`

```rust
// src/gpu_eval.rs:93 (abridged)
#[repr(u32)]
pub enum Op {
    Var = 0, Num = 1, Add = 2, Sub = 3, Mul = 4, Div = 5, Neg = 6, Abs = 7,
    Sqrt = 8, Log = 9, Exp = 10, Sin = 11, Cos = 12, Tan = 13, Tanh = 14,
    Pow = 15, Pow2 = 16, Pow3 = 17, Inv = 18, ProtectedDiv = 19,
    ProtectedSqrt = 20, ProtectedLog = 21, ProtectedExp = 22, ProtectedInv = 23,
    Asin = 24, Acos = 25, ProtectedAsin = 26, ProtectedAcos = 27,
    // ... Regex kingdom opcodes 28-46, compile-kernel-only, never numerically evaluated
}
```

```rust
// src/gpu_eval.rs:280 — the real wire format, 16 bytes
#[repr(C)]
pub struct GpuNode {
    pub op: u32,
    pub arg0: u32,   // Var: column index. Otherwise: index of first child.
    pub arg1: u32,   // index of second child (binary ops)
    pub konst: f32,  // literal value for Num
}
```

**Verified: decode and eval are two separate GPU dispatches, and this matters
a lot for how a `REF_j` can actually resolve.**

- `decode.wgsl` (phylu) turns karva tokens into `GpuNode`s, "one invocation per
  gene," with a bottom-up typed derivation pass (Design C, total typed
  projection — `phylu/docs/PLAN_multityped_gep.md`). **`decode.wgsl` has no
  binding to the `preds` buffer at all** — confirmed by direct read of the
  file. A gene's own node array is slot-addressable (`arg0`/`arg1` index
  earlier nodes in *its own* array), but nothing another gene has computed
  exists yet at decode time.
- `preds` is written later, by a separate evaluator pass
  (`gpu_eval.rs`'s eval kernel, invoked via `eval_resident`/`eval_pass_range`
  in `score.rs`), and only then read by `score.wgsl` (lines 78, 99) for
  linking.
- `SymbolCodes`' `KIND_NAMED` kind (`decode.wgsl:47,333-339`) was considered as
  a candidate home for this, but is **confirmed to be a different thing**: it
  emits `OP_NUM` with `konst` = a value baked in at decode time, and
  `arg0 = name_index+1` purely for debugging identification. It holds a
  **constant**, not a reference to another gene's per-row computed output. It
  cannot be reused for `REF_j`.

**Consequence**: a `REF_j` terminal must resolve at **eval time**, not decode
time. This requires:

1. A new `gpu_eval::Op` variant (e.g. `Op::GeneRef`) — none of the current 27
   numeric ops read another gene's output.
2. Gene-ordered evaluation: gene *j*'s `preds` row must be written before gene
   *i* (*i > j*) reads it via `Op::GeneRef`. Today's eval kernel has no
   cross-gene ordering dependency; this is new scheduling, either as one
   dispatch per gene index in order, or one thread per `(chromosome, row)`
   looping over genes in index order.

This ordering requirement is exactly Neumann's §6.3 fix for his DAG
cost-calculation blowup: visit nodes in topological order (here: gene index
order) so nothing is revisited and nothing is read before it's written. Gene
index order *is* topological order, for free, given the `i` references only
`j < i` constraint below.

### 2.5 Cross-gene combination today — `GeneLinker` (value-level only)

```rust
// fuller/src/chrom_score.rs:326
pub struct GeneLinker {
    pub genes: u32,      // bitmask: which of the chromosome's genes are used
    pub linker: usize,   // index into ScoreSpec::linkers
}

pub fn gene_linker_combinations(n_genes: usize, n_linkers: usize, gene_subsets: bool)
    -> Result<Vec<GeneLinker>, String>   // chrom_score.rs:352
```

`linked()` (`phylu/src/evolve/score.wgsl:72`) reads each used gene's
**already-fully-computed** `preds[...]` and combines by linker (0=avg, 1=mul,
2=add), strictly after every gene has been independently decoded and
evaluated. No sub-expression is shared between genes today; `GeneLinker` fuses
finished numbers, not structure.

**Verified: excluding a hoisted/referenced gene from the top-level linker
needs no new code.** `gene_linker_combinations(.., gene_subsets: true)`
already enumerates every gene subset as a bitmask — "every combination that
excludes gene *k*" is already present in that enumeration.

### 2.6 The persistent replacement dictionary (sibling mechanism)

Referenced in memory (`replacement-dictionary-learning-across-runs.md`):
data-discovered `(pattern → replacement)` pairs accumulate as a persistent
symbolic mutation dictionary across runs, nursery-cohort gated. Source
location not yet resolved — flagged as a follow-up to confirm before
implementation, since §3.3's promotion path (a `REF`-based fold rediscovered
often enough across runs) is the natural point where this mechanism and the
replacement dictionary merge into one, rather than remaining siblings.

## 3. The proposal

### 3.1 Homeotic gene references, not a dynamic symbol table

Following the reviewer's correction: GEP already has a native mechanism for
cross-gene composition — Ferreira's **homeotic genes**, where a gene's
terminal alphabet can include references to other genes' outputs by index,
rather than only problem inputs. This replaces the original named/minted
"magic symbol" design entirely for the common case.

Add a fixed terminal family to the relevant kingdom(s):

```
REF_0, REF_1, ..., REF_{G-1}
```

with the rule: **gene *i* may reference only `REF_j` where `j < i`.**

This gives three things at once, for free:

- **Acyclicity by construction.** Topological order is just gene index order.
  No cycle-detection code is needed anywhere.
- **No dynamic table, no lifetime questions.** `REF_2` means "gene 2's output"
  in every individual that has a gene 2 — it is ordinary, static kingdom
  vocabulary (§2.1), exactly as permanent as any other terminal. Crossover and
  mutation keep it meaningful without any GC or population-scoped scratchpad.
  (The original design's §3.4 "symbol lifetime" question is dissolved, not
  answered — there is no per-discovery symbol whose lifetime needs deciding.)
- **Total semantics for free, via the existing typed decoder.** A `REF_j`
  token is well-typed only if gene *j* exists, is active in this chromosome,
  and produces the type the reference site demands. Design C's total typed
  projection (`phylu/docs/PLAN_multityped_gep.md`) already handles exactly
  this shape of question for every other symbol kind — resolve `REF_j`'s
  output type as "gene *j*'s output type" at decode time, and let a dangling
  or type-mismatched reference fall through the decoder's existing fallback
  path (counted as a masked codon, the same signal already used elsewhere for
  "this token didn't apply"). No new validity-checking mechanism is needed;
  this is the same mechanism, one more symbol kind.

### 3.2 The fold operator (fuller side) — proposes opportunities, decides nothing

Input: the set of `Math` terms for one individual's genes.

1. Assert every gene's term as an independent root into **one** `EGraph`
   built from `math_egraph()` (`expr.rs:163`).
2. **Mode 1 (cheap, first):** hash-cons the raw terms across roots — find
   identical subterms with no saturation. Record the hit rate.
3. **Mode 2 (second):** saturate with the bounded algebra+powers schedule,
   then find cross-root e-class collisions (minimum subterm size, maximal
   shared classes only, per §2.3). Record the *additional* hit rate over
   mode 1 — this is the ablation that tells us whether saturation earns its
   cost here.
4. For each confirmed cross-root sharing opportunity: hoist the shared
   subterm into an earlier gene slot *j* (a real gene, evaluated and, per
   §2.5, excludable from the top-level `GeneLinker` combination via the
   existing bitmask).
5. Rewrite the subterm's occurrences in every consuming gene *i > j* to the
   terminal `REF_j`.
6. Re-encode every touched gene via `terms_to_karva` — no change to the
   converter; `REF_j` is an ordinary token.

Per the "generate everything then rank, never choose" discipline: this
produces one *additional candidate* individual (the DAG-shaped fold)
alongside the unfolded original. **The e-graph's role ends at step 6.** It
has established that the fold is semantically legal (≡S). It has not, and
must not, decide whether the fold is good.

### 3.3 Selection: the existing HFF tournament, unchanged

Whether a fold is worth keeping is answered by the same mechanism that judges
every other candidate: the fold enters the population, competes in the same
HFF-scored, incomplete-beta/CDF-normalized tournament as everything else,
across whatever dimensions a run declares (accuracy, validation
generalization, evaluation cost, anything else) — heterogeneous and
unbounded-dimensional fitness is exactly what this tournament machinery is
built to compare. A fold that wins propagates because it won. A fold that
loses simply doesn't propagate — no penalty, no special case, the same as any
other losing mutation.

This is deliberately the two-layer structure Neumann keeps separate (§1): the
e-graph is the semantic layer (≡S — legality), the tournament is the cost
layer (fitness — worth). Neither substitutes for the other, and no new
selection logic, size term, or parsimony rule is introduced by this design.
Evaluation cost *can* be one of the declared fitness dimensions if a run wants
it to be — that is a run-configuration choice already supported by HFF's
unbounded-dimension design, not something this proposal needs to add.

### 3.4 Cross-run promotion (deferred)

A `REF_j`-based fold that recurs often across many individuals/runs is a
candidate for promotion into the persistent replacement dictionary (§2.6),
becoming permanent vocabulary future runs start with. This is the only place
a genuinely new persistent structure might be needed, and it should be
evaluated as a merge into the existing replacement-dictionary mechanism, not
a new parallel one — pending that mechanism's schema being confirmed.

## 4. What is new work vs. what is reused

| Piece | Status |
|---|---|
| E-graph finds cross-gene share-equivalence (hash-cons + saturation) | **Reused as-is** — multi-root assert + existing bounded saturation; no egglog/ruleset changes |
| `terms_to_karva`/`karva_to_terms` round-trip | **Reused as-is** — bidirectional, tested, takes `PsetSpec` by reference; `REF_j` is an ordinary token |
| Static kingdom symbol table (`geneframe::SymbolTable`) | **Reused, extended** — `REF_0..REF_{G-1}` added as ordinary compile-time rows; no dynamic table |
| Design C total typed projection decoder | **Reused, extended** — one more symbol kind (`REF_j`, typed as gene *j*'s output), same fallback/masked-codon path for invalid references |
| `GeneLinker` bitmask excluding hoisted genes | **Reused as-is** — `gene_subsets: true` already enumerates every exclusion |
| HFF tournament / selection | **Reused as-is** — no new selection logic; a fold is an ordinary candidate |
| Fold operator (steps 3.2.1-3.2.6) | **New** — a new fuller-side function, analogous in shape to `snap.rs`'s constant-promotion pass |
| `gpu_eval::Op::GeneRef` (new opcode) | **New** — no current op reads another gene's output |
| Gene-ordered (topological) eval scheduling | **New** — eval currently assumes gene independence; needs an explicit evaluate-in-index-order pass |
| Cross-run promotion into the replacement dictionary | **Deferred** — depends on confirming that mechanism's actual schema |

## 5. Related work (for the reviewer's benefit, and ours)

- **Ferreira's GEP with ADFs and homeotic genes** — the direct precedent for
  §3.1; this design is that mechanism, not an invention.
- **Cartesian GP** — natively DAG-shaped; the standard baseline for any
  circuit/netlist claim (see the separate chip-design exploration in this
  conversation).
- **Library learning: DreamCoder, Stitch, and babble (POPL 2023)** — learn
  reusable abstractions from e-graphs via anti-unification. This is close
  kin to §3.4 (cross-run promotion) and worth reading before finalizing that
  section, since it may already name the mechanism we'd otherwise reinvent.

## 6. Open questions for the reviewer

1. Should the fold operator (§3.2) run every generation, or periodically/
   on-demand on promising individuals only (e.g. on island winners, landing
   results as an extra candidate the way `simplify`/pump-beat mechanisms
   already do)?
2. Mode 1 vs. mode 2 (§2.3, §3.2): what hit-rate delta would justify always
   running saturation rather than hash-consing-only?
3. Confirm the replacement-dictionary's actual schema (§2.6, §3.4) before
   deciding whether cross-run promotion is a merge or a new, parallel
   mechanism.
4. Is restricting `REF_j` to zero-arity (a terminal referencing a whole gene
   output) sufficient, or is there a real case for a parameterized reference
   (closer to library-learning abstraction with holes, cf. babble) worth
   designing for later rather than now?


## 7. Revision 3 — the homeotic tail

### 7.1 What has been built since revision 2

Everything revision 2 listed as new in §4 now exists in some form, and two of
its open questions have measured answers.

| piece | state (commits) |
|---|---|
| `gpu_eval::Op::GeneRef` (opcode 47), bindings `shared_vals`/`shared_base`, two-pass evaluation of a shared definition before the genes that read it | built, fuller `b141fbe`; phylu `fold_to_dag.rs` (`eval_population_folded`, `FoldedChromosome`) |
| share finder, exact and after rewriting: `maximal_shared`, `maximal_shared_saturated` with a marginal-cost DAG extractor (`extract_dag.rs`) and a tuple budget | built and measured, fuller `dffd911`…`72f6130` |
| the cost gate (per-dispatch and per-reference costs calibrated on the device, in operation-equivalents) | built; on an 800-chromosome, three-gene SR population it refused every fold: folding all sharing removed 2.7 % of operations against a second dispatch costing 177–366 operation-equivalents per submission |
| diversity telemetry (distinct genes, distinct chromosomes, most-copied gene) on the stream and both watchers | built, phylu `c49f9de`, `48bc3b8` |
| population-level share classes (`examples/population_share_classes.rs`) | measured: 6,000 genes → 1,670 distinct; 82.5 % of a young population's operator evaluations repeat a subtree evaluated elsewhere; rewriting merged 34 more |

Open question §6.1 (when to run the fold) is answered by the editor template:
on the immigration interval, on the best individuals, landing the result as a
candidate the tournament judges. Open question §6.2 (hash-consing against
saturation) is answered by measurement: exact repetition dominates; saturation
added 2 of 43 matches within chromosomes and 34 of 1,670 across a population.
Saturation stays, bounded, for the cases it alone finds; it is not the engine
of this design.

What the measurements also said, and what this revision answers: on
three-gene symbolic-regression chromosomes the repeat rate is too low for
sharing to pay at run time, and GeneRef is written only by the fold editor,
never drawn by variation, so a fold is a one-off edit the next mutation can
destroy. Revision 2's DAG is something the editor imposes on a tree-shaped
genome. Revision 3 makes the genome DAG-shaped.

### 7.2 The chromosome: a head of programs and a tail of partials

Andrew's layout (2026-10-07):

```
<chromosome>
  <GeneHead>                 the programs: every gene here is linked into the
    <gene/> <gene/> ...      output, scored by the fitness functions
  </GeneHead>
  <HGeneTail>                the homeotic tail: genes that exist only to be
    <gene/> <gene/> ...      read by other genes through a reference; unread
    <gene/> <gene/> ...      ones are non-coding
  </HGeneTail>
</chromosome>
```

This is Ferreira's cellular system turned the useful way round. In her
design the homeotic gene is the *caller*: a program over the ordinary genes'
outputs, which act as its terminals. Here the head genes are the programs and
the tail is the pool of *callees*: typed terminals a head gene, or an earlier
tail gene, may read. The external inputs stay what they are, terminals of the
kingdom's input types; the head genes' outputs go to the linker and the
fitness functions as today; the tail is new vocabulary in between.

Three rules make it closed, acyclic and total, and all three are rules the
engine already has in another place.

**Closure by type.** A reference `HREF_t` is a terminal of the geneframe
whose output type is the root type of tail gene `t`. Design C's total typed
projection resolves it exactly as it resolves every other token: if the
demanded type and the tail gene's root type agree, the reference reads the
tail gene's value; if not, the demanded type's fallback leaf is emitted and
the codon is counted as masked. A reference to a tail gene that does not
express (refused, or itself a chain of masked codons) takes the same fallback.
No new validity mechanism; one more symbol kind, as §3.1 said.

**Acyclicity by index.** A reference may point only *forward* in the tail:
head genes may read any tail gene; tail gene `t` may read only tail genes
`u > t`. Evaluation runs the tail from its last gene to its first, then the
head. This is the parent-before-child rule Karva already relies on inside a
gene, lifted to the chromosome, and it is the gene-ordered two-pass evaluation
`fold_to_dag` already performs, generalised from "definitions, then readers"
to a fixed order over the tail. No cycle detection anywhere.

**Unread tail genes are non-coding.** A tail gene nothing references costs
nothing at evaluation and is not scored. It drifts under mutation and
crossover exactly as Karva's non-coding region does, a neutral store that a
later mutation can bring into use by drawing a reference to it. This is the
property that makes the layout evolvable rather than merely foldable: the
tail is where partial solutions wait.

The reference, once drawable, is just another terminal to the raw operators.
Mutation can create a reference where there was a variable; transposition can
move one; crossover exchanges tail genes between parents, and because the
rule is positional the exchanged gene means "tail slot t" in both. Nothing in
variation needs to know the tail exists.

### 7.3 What share discovery becomes

With the tail in place the share finder has one job, and it is the fold
editor of §3.2 with a destination: find a subtree repeated across the head
genes, exact or after rewriting, move it into an **empty** tail slot, and
replace every occurrence with a reference to that slot. The chromosome's
behaviour is unchanged (share equivalence, Neumann's ≡S) and its shape is now
a DAG the genome itself encodes.

The tail therefore starts empty, which is the state the engine is in today:
every chromosome has an empty homeotic tail, and the share editor fills it.
Two consequences:

- **Filling is an edit, keeping is selection.** The editor lands the folded
  chromosome as a candidate; the tournament keeps it or not. For the
  tournament to prefer it at equal error, something must see the saving:
  the per-individual executed cost the share editor already reports (the
  reachable-set cost, counted once per shared node) becomes a zero-seeking
  HFF objective where a run wants it. Without that objective a fold is
  fitness-neutral and the next mutation erodes it, which is what the
  SR measurement showed.
- **The run-time gate is still the judge of execution.** Whether a shared
  definition is *executed* once or inlined is the cost gate's decision, per
  chromosome, exactly as now; a tail gene read once is an inlined subtree
  with no second dispatch. The genome records the sharing; the evaluator
  decides whether to exploit it. Revision 2's two layers stay separate.

### 7.4 Where it pays, and where it is measured not to

The three-gene SR chromosome does not repeat enough for sharing to pay at
run time, and the gate was right to refuse. Two settings do:

- **A registry kingdom.** Where the terminals are large reusable pieces, the
  Dagnetic shape (`docs/FRONTIER_EVOLUTION.md` §5): view identifiers as typed
  terminals, whole plans repeating across individuals. There the tail holds
  the shared intermediates and the saving is the product.
- **Longer heads and more genes**, where the within-chromosome repeat rate
  rises. Not measured; the gate's calibration note says a different head
  length should expect to recalibrate.

Across a population the exact repeat rate is already high (82.5 % of
operator evaluations at generation 52), but that is a population-level DAG,
not a chromosome's, and the saturating finder does not scale to it;
population-level sharing is a separate mechanism (hash-cons and evaluate each
class once) and is not what the homeotic tail is for.

### 7.5 What is new against what is built, revision 3

| piece | status |
|---|---|
| GeneRef opcode, shared bindings, two-pass evaluation | built (§7.1) |
| share finder, exact and saturated; cost gate | built (§7.1) |
| the tail as a region of the chromosome: layout, `HREF_t` terminals in the geneframe typed as the slot's root type, forward-only rule, tail-last-to-first evaluation order | **new**; rows and decoder, host and device, with the parity test both decoders are held to |
| `HREF_t` drawable by initialisation and variation; dangling and mismatched references to the fallback leaf | **new**; the regex-leaf `arg0` fix phylu is holding belongs in the same decoder change |
| the share editor scheduled on the immigration interval, writing into empty tail slots | **new** scheduling of a built editor |
| executed cost as an optional HFF objective | **new**, small; the number already exists |
| a kingdom whose repeat rate makes it pay | Dagnetic, designed, not built |

### 7.6 Open questions, revision 3

1. **Tail size.** Fixed per kingdom like the Karva tail, or a layout parameter
   the run card carries? A fixed small tail (two to four slots) is enough to
   measure whether anything stays in it.
2. **Can a tail gene read the head?** No, under the forward-only rule; the
   head reads the tail, never the reverse. Confirm this is the intended
   asymmetry, since it is what keeps evaluation one ordered pass.
3. **Parameterised references** (§6.4) remain deferred; a tail gene with
   holes is library learning, and the zero-arity reference is enough to test
   whether a population keeps and reuses partials at all.
4. **The measurement that decides it:** on the first kingdom with a tail, the
   diversity telemetry plus one new counter, how many tail slots are read per
   individual over generations. If that number stays at zero the tail is
   dead weight; if it rises and holds, the population is evolving DAG plans.
