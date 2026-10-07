# Frontier Evolution

## GPU-accelerated evolutionary computing, as Gamakon has built it and where it leads

*This document spans three repositories: `fuller` (define, rewrite, evaluate),
`phylu` (evolve) and `hff` (the fitness function and the earlier Python
engine). It is written for the company, not for a conference. The GECCO draft
(`fuller/papers/geneframe-gecco.tex`) is the technical account and nothing here
contradicts it; where this document quotes a number, it is the paper's number
under the paper's conditions.*

Every statement below is in one of three tiers, and each section says which:

| tier | meaning |
|---|---|
| **Measured** | built, tested, and a run log or test pins it |
| **Required** | not built or not finished; the vision depends on it |
| **Envisioned** | what the measured and required parts would make possible |

---

## 0. Context

**Objective.** Set out what "frontier evolution" is, what has been built, what
remains, and the breadth of what the combination could change.

**Approach.** Start from the working components and derive the properties from
them, rather than asserting properties and hoping the components follow. Each
claim of breadth traces back through a kingdom to a component.

**Unique feature.** A single typed symbol table, the geneframe, in which a
language to evolve is a set of rows, not a new engine; a population that lives
on the GPU for the whole of a generation; and sound expression rewriting that
acts as a mutation operator inside the search.

**Status.** Three kingdoms run on one engine. Symbolic regression reaches the
same order of recovery as AI Feynman on SRBench, under a weaker protocol, in
90 seconds a problem on a laptop. A four-typed regular-expression kingdom runs
with no change to variation or selection. The remaining parts are listed in
Section 4 with their status.

**Why now.** The parts that make evolution cheap (device-resident population),
general (typed table), and sound (rewriting as mutation) exist separately in the
literature and have not been combined. Combined, they change the cost of a
search by two orders of magnitude and the cost of a new domain from an engine to
a table.

---

## 1. What frontier evolution is

Evolutionary computation has always been limited by two costs. The first is
evaluation: a population of programs must be run against data every generation,
and on a CPU that bounds populations to thousands and runs to hours. The second
is the engine: every new domain has needed its own representation, its own
operators, its own repair logic, and so each application has been a project.

Frontier evolution, as Gamakon has built it, removes both costs at once.

**The population lives on the device.** A generation is one command
submission: decode, evaluate, score, rank and select are compute kernels, and
the host reads back kilobytes. On an Apple laptop the engine evaluates between
23,000 and 135,000 individuals a second depending on population, and 250,000 a
second on the regular-expression kingdom; the evaluator alone reaches 62.5
million gene-row evaluations a second. The same code runs on any GPU that wgpu
reaches: Metal, Vulkan, DirectX 12. *(Measured.)*

**A language is rows in a table.** The geneframe is one typed symbol table.
Each symbol carries a many-hot typed arity signature, how many inputs of each
type it consumes and what type it produces, and a kingdom is a query over that
table. The decoder is a total typed projection: every genome, however mutated,
projects to a well-typed tree, so the raw genetic operators are kept unchanged
and no repair step is needed. Adding a kingdom means adding rows and an
evaluation kernel. *(Measured for three kingdoms.)*

**Types make illegal shapes unrepresentable.** The transcendental symbolic
regression kingdom gives the operators of symbolic regression a type that
records transcendental depth. A third nested transcendental has no signature,
so the search cannot build the towers of `sin(cos(exp(...)))` that a penalty
only discourages. The ceiling was set by measuring every SRBench law. *(Measured.)*

**Rewriting is a mutation operator.** Equality saturation over an e-graph
(egglog) edits the best individuals during the search: simplification, constant
snapping, and detection of subexpressions shared across a chromosome's genes,
including sharing visible only after rewriting. The edited gene is re-encoded,
checked by the decoder, and returned to the population for the tournament to
judge. The rewriting is sound: an editor can only produce an equivalent
expression. *(Measured; the share editor is built and measured but not yet
scheduled in the live loop.)*

**Many objectives, no weights.** Hyperspherical fitness (HFF) scores an
individual by the angular distance of its vector of zero-seeking objectives
from an ideal pole. Training error and validation error are both objectives, so
generalisation is selected for rather than checked afterwards, and no parsimony
weight is tuned. The age-layered population structure is reformulated as a
cohort label with a same-cohort tournament preference, so young lines are
protected without a layered population. *(Measured.)*

**Deterministic and bit-exact.** Same seed, same result. The host and device
decoders agree on every gene and on the count of substituted codons; the untyped
symbolic-regression kingdom is held byte-identical to a golden population pinned
before any typing existed. *(Measured; tests.)*

**Two systems, one loop.** phylu evolves; fuller defines the symbol table,
rewrites, and evaluates. The division is what lets rewriting act inside the
search and what lets a new kingdom be added without touching selection.
*(Measured.)*

---

## 2. What works today (Measured)

Numbers are the GECCO draft's, with its conditions. None is a multi-seed study.

| kingdom | types | result | conditions |
|---|---|---|---|
| Symbolic regression (untyped) | one, F | byte-identical base case; the throughput table | 21 Sep 2026 build, Feynman problems |
| Transcendental symbolic regression (TSR) | three, F / T1 / T2 | 74 of 133 SRBench laws recovered by the engine; 79 with fuller's post-search candidates | one development seed, 90 s a problem, population 2,000, SRBench's own equivalence check as the stop |
| Regular expressions | four, Pattern / CharClass / Char / Integer | UK postcode detector: held-out recall 92.4 % single-type to 97.5 % multi-typed; 0 of 150 foreign-format negatives accepted | one seed, 180 s, population 2,600 |

For scale, AI Feynman, the strongest published method on SRBench's
ground-truth track, finds an exact solution in about 53 % of runs at zero
noise, with up to eight hours a run. Our count is the same order, under a weaker
protocol: one seed, 90 seconds on a laptop, the symbolic check as stop
condition. The run on the official seeds without access to the answer has been
begun and not completed. The claim we make is that the two systems together
reach the same order of recovery in a small fraction of the time. That claim is
measured; "better than AI Feynman" is not, and this document does not make it.

The previous certified best of the untyped engine was 62 of 133. The TSR run's
74 is not a controlled comparison against it, and the same-seed run with
typing off has not been done.

---

## 3. What a kingdom is, and the three that exist

A kingdom is: rows in the geneframe (symbols with typed signatures), one
zero-arity fallback leaf per type, an evaluation kernel that turns a decoded
tree into a score vector over the data, and nothing else. Variation, selection,
islands, the immigration interval, HFF, the editors and the telemetry are
shared.

| kingdom | input | fitness objectives | what it produces |
|---|---|---|---|
| Symbolic regression | a table of floats with a target column | training and validation error, extrapolation error | an equation |
| TSR | the same | the same | an equation with at most two nested transcendentals on any path |
| Regular expressions | labelled strings, or their ByteFreq masks | the confusion matrix on training and validation | a POSIX regular expression compiled to a Thompson virtual machine |

The regular-expression kingdom is the existence proof: a genuinely four-typed
language on the same engine, with the difference from symbolic regression
being the rows, the decoder's typing mode, and the kernel.

---

## 4. Required parts

What the vision in Section 5 depends on, and where each stands. "Built" means
tested and in the engine; "partial" means built but not finished or not in the
live loop; "not started" means a design exists at most.

### 4.1 Engine

| part | status | what it unlocks |
|---|---|---|
| Device-resident generation (decode, evaluate, score, HFF, select on one submission) | built | the throughput in Section 2 |
| Total typed projection decoder, host and device, parity-tested | built | any kingdom whose operators fit arity two |
| TSR depth ladder (refusal typing, bottom-up) | built | the SRBench result |
| TSR re-expressed through the projection decoder | not started | one typing mechanism for every kingdom; today the ladder is a fixed rule the device applies and the table only checks |
| Cohort tournaments (ALPS reformulated), two islands, random immigrants | built | protection of young lines without a layered population |
| Editors in the live loop: simplify, snap, typed snap (shape library), beam (functional wraps fitted by least squares), fold (constant collapse, off by default) | built | compact models with recognisable constants; the fold's cost on the device fell from 155 ms to 4 ms a beat |
| Share editor (saturated share-equivalence, DAG extraction, calibrated cost gate) | partial: built and measured, not scheduled in the live loop | shared subexpressions executed once when they pay; a per-individual executed-cost figure for parsimony |
| Counted repetition on the device decoder | partial: table rows and host expansion exist, withheld from device initialisation and variation | regular expressions with `{n,m}` and the Integer type drawable |
| Nested folding (a shared definition that itself contains a reference) | not started | deeper sharing |
| Checkpoint and bit-exact resume (counter-based random draws, periodic by seconds) | built | the official SRBench protocol, which is 1,330 runs of up to eight hours, about a month of one GPU, run in pieces |
| Telemetry stream and watchers (one append-only stream; progress, chart, genealogy, two-dimensional map of the population) | built | a run that can be read while it runs |
| Run card with every switch and the git commit baked in | built | every number traceable to a configuration |
| Synthetic rows as objectives (SMOGD, SMOTE) | built, off in reported runs | selection on data the search has not seen |
| Stream-order ceiling (Horton–Strahler) as a law-likeness sensor | built as a sensor; lost as a hard rule, off | ranking column |
| Multi-seed statistics; the official SRBench protocol | partial: one official run begun and stopped | a reportable benchmark number |
| Multi-GPU islands | not started: no code, no plan | populations of millions; one island per device |
| Replacement dictionary across runs (data-discovered pattern to replacement pairs that persist) | not started | learning across problems, not just within a run |
| Typed snap shape library beyond physics (power laws, growth and saturation, distributions, periodic, engineering) | partial: 11 physics shapes | symbolic regression for every science, not Feynman's |
| Kingdom classifier (the learned router that picks which rewrite family to load) | not started: corpus instrumentation exists | rewriting that scales past the non-confluence of rule families |

### 4.2 The part that gates breadth

Today a kingdom costs rows **and a hand-written evaluation path**: a WGSL
kernel that turns a decoded node array into fitness objective columns, a
bit-exact host reference with parity tests, and a renderer for the watcher.
Variation, decoding, selection, HFF, islands, cohorts, checkpoint and telemetry
are untouched by a new kingdom. The regular-expression kingdom's evaluation
path was a compile kernel, a match kernel, a mask kernel, a scorer and a test
oracle; that path was the whole cost, and everything else was rows. The vision below needs the kernel cost to fall to near zero: either a
generic typed interpreter kernel that executes any decoded tree against any
data by walking the table, or a compiler that emits the kernel from the
table's rows. Neither exists. This is the single required part on which the
breadth of Section 5 depends, and it is stated here so that no one mistakes the
table for the whole cost.

### 4.3 Kingdoms designed but not built

These names are Andrew's, from the nucleotable design notes and the type enum
in `fuller/src/geneframe.rs`. None has rows loaded or a kernel.

| kingdom | designed types | purpose in the design notes |
|---|---|---|
| SQL | S, I, F, B, A, L | evolution of queries; the geneframe's own native language (named as an example only; nothing in code) |
| NLP-English | spaCy entity and phrase labels in; Entity, Relation, Metric, Procedure, Narrative out | sentences parsed to Karva by `lang2karva`; the content layer of the OODA loop |
| Botji | Addr | hierarchical addresses (`book/chapter/paragraph/verse`); the structural index over content |

The type enum declares every one of these types now so that the row structure
stays stable as kingdoms are added; none has a symbol row or a kernel.

---

## 5. What it could inspire (Envisioned)

Each item names the component it depends on and the required part, if any, it
waits for.

**Laws from data at laptop cost.** A dataset with a target column gets an
equation in 90 seconds, with validation built into selection and the constants
snapped to recognisable values. Depends on TSR and the editors (built); waits
for the shape library beyond physics and multi-seed statistics. The audience is
every experimental scientist, engineer and analyst who has a table and wants a
formula, not a model.

**Validators and parsers from examples.** The regular-expression kingdom
descends from ByteFreq, the mask-profiling technique for data quality on read:
show the engine a column's values and their masks, and it evolves the rule that
accepts the valid ones and refuses the rest. Depends on the regex kingdom
(built); waits for counted repetition on the device. Every data pipeline that
hand-writes validation rules is a user.

**Program synthesis in typed domain languages.** The geneframe admits any
language whose operators fit the arity limit: SQL, configuration languages, the
NLP and Botji kingdoms already designed. Each is rows plus a kernel. Waits
entirely on the generic kernel of Section 4.2; until then each kingdom is a
project.

**Interpretable by construction.** The output of every kingdom is a program in
a readable language, not a weight matrix: an equation, a regular expression, a
query. There is nothing to explain after the fact. Depends on the geneframe
(built).

**Evolution as the verifier beside a language model.** DeepMind's FunSearch
(2023) and AlphaEvolve (2025) use a large language model as the mutation
operator inside an evolutionary loop. Frontier evolution is the other half of
that picture: a loop whose mutations are cheap, deterministic and sound, and
whose every artefact is executed and scored on the device. A language model
fits into it as one more editor on the immigration interval, proposing
individuals that the tournament judges like any other, folded into the
existing mechanism rather than built beside it. Depends on the editor template
(built); nothing of this is started.

**Scale.** The laptop figures are one device. Multi-GPU islands make the
population a function of the number of devices, with the immigration interval
as the only coupling. Waits on multi-GPU islands.

**Learning across runs.** The replacement dictionary turns each run's
discovered rewrites into mutations for the next, so the engine gets better at a
domain with use. Waits on the dictionary.

---

## 6. What would show this to be wrong

- **The stepping-stone question.** If the same-seed run with typing off recovers
  as many laws as TSR, the ladder's gain is an artefact of engine changes, not
  of typing. This run has not been done.
- **The official seeds.** If the SRBench count under the official protocol falls
  well below the development-seed count, the 74 to 79 is a development
  artefact. One official run was begun and stopped.
- **A fourth kingdom's kernel.** If SQL or NLP costs as much as regex did, the
  table does not reduce the cost of a domain and Section 4.2 is the whole
  problem.
- **The share editor's economics.** The calibrated gate refuses every fold at
  the measured repeat rate. If that holds at longer heads and other kingdoms,
  shared execution is a parsimony sensor and not a speed-up.

---

## 7. One-paragraph version

Gamakon has built an evolutionary computing engine in which the population
lives on the GPU, a language to evolve is a set of rows in one typed symbol
table, and sound expression rewriting acts as a mutation operator inside the
search. Three kingdoms run on it: symbolic regression, a typed symbolic
regression that cannot build the nested transcendental towers no physical law
has, and a four-typed regular-expression language. On SRBench it recovers 74 to
79 of 133 ground-truth laws in 90 seconds a problem on a laptop, the same
order as AI Feynman under a weaker protocol; on UK postcodes it evolves a
detector with 97.5 % held-out recall that rejects every foreign format. The
part that remains is to make a new kingdom cost rows alone.
