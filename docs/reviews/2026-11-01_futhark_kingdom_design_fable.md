# Review: the Futhark kingdom's master table, views and type system, Fable, 2026-11-01

Reviewer: Fable (advisor, read-only), on fuller `feat/futhark-kingdom` at 352f69e9, sibling phylu beside it. Read in
full: `docs/PLAN_futhark_kingdom.md`, `nucleotable/docs/MultiTypedGEP.md`, `kingdoms/futhark/master/*`, the five
`view.tsv`, `kingdoms/futhark/sniff.py`, `kingdoms/futhark/view.py` (binder, expansion, reachability, the free-class
loop), `src/futhark_view.rs` (the same), `src/kingdom_contract.rs` (heads, bounds, reachability), the alphabet and
mutation of `src/typed_gep.rs`, `src/futhark_rings.rs::point_mutate`, the runner's program-size default in
`examples/futhark_evolve.rs`, phylu's `typed_vary.rs` mutation law, `docs/PLAN_futhark_fitness_fixes.md` sections H
to K, the Codex review of 2026-11-01 and the 2026-10-28 faults review. Every number below was recomputed from the
committed tables with a Python script that reproduces the engine's widths exactly (84,015 and 13,945), so the
breakdowns are the engine's.

## 1. The answer

The view is not the stupid part. Since this morning's commit (352f69e9) the whole master table does go to every
problem: a view only binds the 42 signature templates to the problem's own sizes and inputs, which is the per-problem
terminal set every GEP run has had since Ferreira 2001, and then applies the forward-and-backward reachability you
ruled, which is the textbook removal of useless grammar symbols (non-generating first, then unreachable; Hopcroft and
Ullman) and the same idea as Montana's type possibility tables in strongly typed GP (1995). That part is sound, and
the Python and Rust implementations are identical and correct. What the agents did that has no precedent in GEP, nor
in the GP literature I know, is to put the lambda SCOPE into the TYPE: a value's type is "an i32, inside a loop over
rows carrying an a.cols state, inside a tabulate over cols", so pathfinder's 6 shape classes become 832 types (211
scopes times the classes that exist in each), and your per-type head-and-tail layout then allocates a worst-case
head, tail and constant domain for each of them: 84,015 words, 84 % of them in depth-2 scope types, 10 % in types
that have no function at all, 6.7 % in cells whose alphabet is one symbol, for a reference program that occupies 24
cells. The width itself is not the fault, by your ruling. What it does to the search is: every operator that works on
whole domains (crossover, transposition, inversion) lands on a non-coding cell 99.97 % of the time, which is why the
agents' own operator table shows crossover alone leaving 70 to 79 % of children with their parent's output, and why
point mutation had to be rewritten to aim at coding cells only. A scope typing that records only WHICH variable
classes are in scope (shadowing by class, as Futhark itself allows) gives 30 scopes instead of 211, counted on the
same view, and loses nothing on the 112 real programs in the two pools (0 of 112 reference an outer variable through
a same-kind inner binder). Last, the machinery around the view (two byte-identical implementations, a 2,329-line
generator, 5.2 MB of generated tables and e-graph text per problem committed to git and never loaded) is far heavier
than the 42 templates it serves; one implementation would do.

## 2. Findings, most severe first

### F1. Scopes are encoded as types, and that is where the 607 and 832 come from (severe)

- `kingdoms/futhark/master/frames.tsv`: three frame kinds (`T`, `L`, `F`) over `D:dims` and `S:states`. For
  pathfinder that is 14 frames (`T-rows`, `T-cols`, and `L`/`F` over 2 dims times 3 states). `view.py:204`
  (`context_list`) and the Rust twin build every stack of frames up to depth 2: 1 + 14 + 196 = **211 contexts**.
- Measured from `sig/pathfinder/types.tsv`: 832 types = 155 contexts holding all five classes (`i32`, `ix.rows`,
  `ix.cols`, `a.rows`, `a.cols`) + 56 contexts holding **only** `a.rows` (a depth-2 state variable and nothing else)
  + the one context-free `a.rows.cols`. By depth: 6 types at depth 0, 70 at depth 1, 756 at depth 2.
- The type exists to make the variable terminals legal by type alone (`M0017`, `M0023`, `M0037`: `{V}@{OUT}` is only
  in the alphabet of a type whose context binds `V`). That is the minimal device **if** the context must be the exact
  frame stack. It need not be: Futhark lets an inner binder shadow an outer one, so a context can be the SET of
  variable classes in scope, with variables named by class (`j_cols`, `i_rows`, `s_i32`), and legality is unchanged.
  Counted on pathfinder's 211 contexts: **30** distinct (set of bound classes, shallow flag), 22 without the flag.
- What that costs: a program that reads an OUTER variable from inside a same-class inner binder. Measured on the pools
  (`sig/pathfinder/pool`, `sig/histogram/pool`, every program with status `read`): 54 + 58 = 112 programs, **0**
  reference an outer variable from inside a same-kind inner binder. The four references do not either.
- Precedent: none in GEP (Ferreira's GEP has one data type and no binders). In GP, scoped variables are handled by
  typing the variable with an environment that the decoder tracks (Yu and Clack's PolyGP with lambda abstraction,
  1998; Binard and Felty's System-F GP, 2008) or by attribute/Christiansen grammars (Ortega, de la Cruz, Alfonseca
  2007). Writing the full environment into the type name is a de Bruijn-style encoding that I know no GP system to
  use; it is sound, and it multiplies the alphabet by the number of environments. Classes-as-types, by contrast, has
  a precedent in the target language itself: Futhark's size types (Henriksen and Elsman, 2021).

### F2. The per-type worst-case layout over 832 types pads the gene 3,500 to one, and the padding is where the domain operators land (severe for search, not for memory)

The layout rule is Andrew's own (`MultiTypedGEP.md` §2.1, §2.2; `src/kingdom_contract.rs:326` and `:371`), exact per
type. The type count of F1 is what turns it into this (pathfinder, N = 13; recomputed, equal to the engine's):

| part | words | share |
|---|---:|---:|
| heads (`Σ h_T`, 832 types) | 14,065 | 16.7 % |
| tails (`Σ t_T`) | 45,888 | 54.6 % |
| Dc (`Σ d_T` over 155 i32 types) | 24,062 | 28.6 % |
| **gene** | **84,015** | |
| of which in depth-2 scope types | 70,224 | 83.6 % |
| in types with **no function** (head cells that can only hold terminals) | 8,425 | 10.0 % |
| in cells whose alphabet is **one symbol** (no information) | 5,641 | 6.7 % |
| the reference program's expressed cells | 24 | 0.03 % |
| classical one-type Karva for the same N (n_max = 3): head 37 | 187 | |

- The worst single type is the context-free input `a.rows.cols`: 0 functions, 1 terminal (`walls`), a 12-cell head
  and a **2,029-cell tail**, every cell holding the same symbol. `L` (the leaf bound) is 13,562, against the 27 leaves
  a 13-function program can have, because every one of the 14,065 head cells may be a function.
- Histogram (N = 4): 13,945 words, 79.5 % in depth-2 types, 15.7 % in types with no function; the reference occupies
  7 cells.
- The effect on the search is measured in the agents' own table (`docs/PLAN_futhark_fitness_fixes.md`, "Results so
  far"): crossover alone, 70 to 79 % of children with the parent's output; ring walk, 53 to 56 %. A homologous
  crossover picks one `(gene, type, domain)`; at most 13 of 832 types are coding in any program, so the pick is
  non-coding at least 98 % of the time by type and 99.97 % by cell. Point mutation escapes this only because
  `src/futhark_rings.rs:382` (`point_mutate`) draws from `coding_cells` and retries up to 16 times for an alphabet of
  size 2 or more; it is a targeted operator, not GEP's point mutation.
- phylu's default mutation law would be worse, had this kingdom run there: `typed_vary.rs:586` HIGH mode scales the
  per-cell rate so that "every domain expects the whole gene's edits", so a 100-cell domain of a 84,015-word gene at
  rate 0.02 expects 1,680 hits: every cell redrawn every generation. The signature runner does not use it (it is
  fuller's `futhark_evolve.rs` with host operators), but the A7 matrix kingdom in `shape_typed.rs` does; A7 has 11
  types, where the rule is harmless.

### F3. The runner's default program size is the reference's function count (major)

`examples/futhark_evolve.rs:1766`: `program_size = n_funcs` of the reference unless `--program-size` is given. For
reduce and scan the reference is ONE function, so the layout is a root head of one cell and the gene is 13 and 9 words
(`src/futhark_sig/tests_view.rs:137`): the search space is the root alphabet (16 functions, 2 terminals, 5 literals).
Nothing with two functions can be expressed. For pathfinder every program of up to 13 functions fits, and larger ones
only when their type mix happens to; a faster program is not usually the same size as the slow one. The default should
be a layout parameter of the experiment (a fixed N, say 32, as SR and regex use), never the reference's size.

### F4. The machinery is out of proportion to what it computes (major, cost)

- The view is 42 signature templates plus 3 frames and a 14-line rule table. Serving them: `view.py` 506 lines,
  `src/futhark_view.rs` 939 lines, kept byte-identical by test; `gen_tables.py` 2,329 lines; per problem a committed
  `types.tsv`, `functions.tsv` (pathfinder 961 KB, 6,037 rows), `egraph.datatype` (711 KB, 832 sorts), `egraph.rules`
  (216 KB: the nine i32 identities instantiated 155 times, once per scope), `egraph.rows`, `egraph.sorts`: 5.2 MB for
  pathfinder, regenerated by every table change (352f69e9 touched 13,509 lines), and by design never loaded at runtime
  (`futhark_sig.rs:38`).
- Two implementations exist because the witnesses are Python and the engine is Rust. The witnesses can read a table the
  Rust view writes; the Python view then has no reason to exist, and the committed tables become a cache or a fixture
  (a hash is enough to pin the regression).
- The sniffer is right-sized (176 lines): it reads the entry's typed signature, which is the terminal set and root type,
  and refuses what it cannot read. That is the one per-problem step every GP system has; keep it.

### F5. The reachability rule, as built, is correct and complete (no fault)

`view.py:329` and `src/futhark_view.rs:632`: F is the least fixpoint from the leaf rows; a row is usable when its
arguments are in F; B is the closure from the root over usable rows; kept = usable with output in B; the root must be
in F. This is Hopcroft and Ullman's order (generating first, then reachable), which is the order that leaves no useless
symbol. The two implementations are line-for-line the same algorithm. The context-free class loop
(`view_rows`, both files) is bounded and errors when it does not settle, as it should. `KingdomContract::reachable`
(`kingdom_contract.rs:298`, forward from the root over signatures) is then applied again by the layout and agrees with
B on a cleaned table. Two notes, not faults: the rule is at the type level, so it cannot see that a depth-2 type's
functions are the non-binders only (that is already in the alphabet); and the S8 witness ("every dual has a terminal")
is a layout requirement (a tail needs a terminal), correctly kept as a loud failure for a future master row.

### F6. Dead and duplicated structure (minor)

- `family` column of `master/rows.tsv` (`core`, `mod`, `scan`, `hist`): no consumer in a signature view since 3e; only
  `matrix` is still read (`view.py:286`). Drop the signature families or document them as labels.
- The 56 depth-2 contexts holding only `a.rows` (F1) each give one type with 0 functions and 1 to 2 terminals, 12 head
  cells and a tail: 56 types of pure padding that the reachability rule cannot remove because the type is legitimately
  reachable (`loop.rows` carrying an `a.rows` state inside a depth-1 frame). A set-of-classes context removes them.
- `a_T` counts a type's demand from functions that take it; for a type with no function of its own the head is a tail by
  another name (alphabet = terminals). `heads_from_program_size` could give such a type `h_T = 0` with no change to the
  expressible programs; it is 10 % of pathfinder's width and 15.7 % of histogram's.
- The published pathfinder kernel does not read into the kingdom (tuple, `map5`, `rotate`; pool manifest), and 9 of
  1,575 benchmark functions do: the master's coverage, not the view, is the limit on what the search can be seeded with.

## 3. Recommended design

Keep three things exactly as they are: the master table as the one source; the sniffer as the per-problem signature
reader; the bidirectional reachability as the only trim. They are standard GP practice and correctly built.

Change four:

1. **Contexts are sets, not stacks.** A context is the set of variable classes in scope plus the depth (for the
   `shallow` rule and the work bound). Variables are named by class and shadow; `frames.tsv` keeps the binders, the
   name `c/T-cols/L-rows-i32` becomes `c/{ix.cols,ix.rows,i32}@2`. Pathfinder: 211 contexts to 30, 832 types to at most
   150 (5 classes times 30), the gene to roughly a fifth, every program of the two pools still expressible (measured 0
   losses). The e-graph's rules shrink by the same factor, since they are instantiated per type.
2. **Program size is an experiment parameter**, with a fixed default well above any reference (32), recorded in the
   run card. Reduce and scan at N = 1 are not searches.
3. **One view implementation**, in Rust (`src/futhark_view.rs`), with `View::write` producing the tables the Python
   witnesses check; delete `view.py`'s expansion and the committed per-problem tables, pin the regression by a hash of
   the written text. `gen_tables.py` keeps the matrix kingdom until step 4 retires it.
4. **Domain operators act on coding domains.** Until the type count is down, crossover, transposition and inversion
   should pick among the `(type, domain)` pairs that are coding in at least one parent, as point mutation already
   does; otherwise they are neutral 99.97 % of the time and their measurements say nothing about the operator.

Not recommended: a filter on the table by what the problem's source uses (the one removed today), a per-problem
family list, or any cap on the type count. The search space is set by the alphabet per cell and the depth bound, and
those are right; the padding is the only thing to shrink.

## 4. What to measure to decide

| # | measurement | expected if the recommendation is right |
|---|---|---|
| 1 | Build the set-context view for pathfinder and histogram: types, gene width at N = 13 and 4, and whether all 112 pool programs and the four references read and express | ≤ 150 and ≤ 120 types; width under 20,000 and 4,000; 112 of 112 read |
| 2 | The operator table (`--measure`, 700 children, same seeds) on both layouts: children with the parent's output, per operator | crossover's 70 to 79 % falls by more than half; point mutation unchanged |
| 3 | R against C on histogram and pathfinder, 3 seeds, same budget, both layouts: generations to the first exact program and best HFF per generation | the smaller layout reaches an exact program in fewer evaluations in at least 2 of 3 seeds |
| 4 | Expression time per generation (host and GPU) for 84,015 against the smaller width | proportional to width; this is compute, not storage, and it is the only width cost that counts |
| 5 | Reduce and scan at N = 1 (today) against N = 8: does the search find anything but the root alphabet | at N = 1 the best is one of 18 symbols; at N = 8 the e-graph pump has material |
| 6 | A control for F1's loss: count, in any new pool programs, references to an outer variable through a same-class inner binder | stays 0 or near it; if it does not, the context keeps a count per class (0, 1, 2) instead of a set |

## Sources (stated to the confidence I have)

- Ferreira, C. (2001) "Gene expression programming: a new adaptive algorithm for solving problems", Complex Systems
  13(2); Ferreira (2006) *Gene Expression Programming*, 2nd ed. One data type; head of h, tail of h(n−1)+1; function
  and terminal sets chosen per problem; multigenic chromosomes, homeotic genes and ADFs. No typing, no binders.
- Montana, D. (1995) "Strongly typed genetic programming", Evolutionary Computation 3(2): types on every node, generic
  functions, type possibility tables (which types a tree of depth ≤ d can produce) used to keep every tree legal.
- Hopcroft and Ullman (1979), *Introduction to Automata Theory, Languages and Computation*: removal of useless symbols,
  non-generating first, then unreachable. Whigham (1995) CFG-GP and O'Neill and Ryan's grammatical evolution (2001,
  2003) take a grammar per problem; GE's mapping can fail (wrapping, invalid individuals), so it is not legal by
  construction.
- Yu and Clack (1998) PolyGP; Binard and Felty (2008) "Genetic programming with polymorphic types and higher-order
  functions" (GECCO); Ortega, de la Cruz and Alfonseca (2007) "Christiansen grammar evolution" (IEEE TEC): scoped
  variables handled by an environment or an attribute grammar, not by naming the environment in the type.
- Henriksen and Elsman (2021) "Towards size-dependent types for array programming" (ARRAY): Futhark's size types,
  the precedent for a shape class as a type.
