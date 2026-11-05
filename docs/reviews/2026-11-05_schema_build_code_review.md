# Code review: the fixed schema build (fuller a63edde0, db8d0f4b, 779412d1; phylu fc8461c2)

Reviewed 2026-11-05 against `docs/DESIGN_futhark_types_and_properties.md` (1163a97a), CLAUDE.md and Rules.md. Read
only; nothing built or run.

## Verdict

**Not yet the fixed schema: two blocking faults.** The generic layer (formula language, Flow and Guard tables, table
check, decoder passes 2 to 4, the cost's `advice`/`rescues`) follows the design closely and is kingdom-free. But
(1) phylu's marriage graft was not moved to innermost-first words, so a crossover of any subtree that binds a
variable silently changes which variable its words name; and (2) Futhark master 6 is a second row system: the four
row kinds drive only the decoder's yes/no outcomes, while rendering, interpreting, costing and reading still run on
the legacy rows (`Requirement6`, `Adapter6`, `Site6`, `Substitute6`, `Subject6`, `Binding6`) and on Rust `match`
arms keyed by adapter names, joined to the new tables by a positional index. The generic renderer cannot produce a
legal Futhark program. Its test pins illegal text as the expected output.

Andrew's ruling (2026-11-05) applies to every fix below: there are no compatibility paths and nothing is migrated.
Superseded forms are deleted, not bridged. Anything worth keeping (reference programs, champions, fixtures,
ledgers) is kept as program TEXT and read back through the reader into genes.

## Findings

### Blocking

**1. phylu's graft writes outermost-first words (silent meaning change).** `phylu/src/evolve/typed_resolve.rs:191-216`.
`graft_node` lays out the destination's variables first and the subtree's own after them. It writes an inner
reference as `outer + at`, a free reference as `dst.index_of(..)` without the subtree's own binders in front, and
resolves `choice` the same way. Since 779412d1, `in_scope` puts the innermost first.
*Failure:* graft `tab[j] n (mod k (get vs j))` where one `i64` `i` is in scope. `j` is written as word 1, and the
decoder reads `[j, i][1]` = `i`. The `resolved` tree `graft` returns still says `j`, so `no_op` and the marriage's
keys disagree with the decoder. A free `i` inside a subtree that binds an `i64` is written as 0, which reads as the
subtree's own variable.
The test fc8461c2 edited (`typed_resolve.rs:~570`) grafts a subtree that binds nothing, so it misses this.
*Fix:* compute the word as `at` for an inner reference and as `inner.len() + dst.index_of(..)` for a free one.
Compute `choice` from the innermost-first list. Add a test that grafts a binding subtree on a real master 6 view
and asserts `graft(..).resolved == decoder.decode(offspring)` at the grafted path.

**2. Master 6 is a hidden second schema** (`src/futhark_table.rs:526-680, 958-1075, 1380-1500, 1691, 1833-1860`;
`src/futhark_cost.rs:626-720`; `src/futhark_sig/interp6.rs:108, 674-780`; `src/futhark_reader.rs:2539-2580`).
`Kingdom6::tables` converts the legacy rows into Guards whose repair templates are placeholders (`{self}`, `{k}`)
and whose `calls` name adapters that are not Symbol rows. `legacy_spelling` escapes master 6's own placeholders
(`{1.v0}`, `{1.type}`) as literal braces.
Everything that gives the program its meaning reads `reqs[o.requirement]`, a flattened `requires` list whose order
must equal `tables.guards`' order. Nothing checks that order beyond a length assert in two tests. The meaning is
also hard-coded in three copies of a `match` on `"clampix" | "fit" | "fit.pair" | "fit.inner" | "fit.state" |
"fit.body"` (renderer, cost, interpreter).
*Failure:* a reordered or added guard silently pairs an outcome with another requirement's adapter. A kingdom author
must edit three Rust functions to add an adapter, which breaks "a kingdom is rows".
*Next task must remove:* `Subject6`, `Requirement6`, `Site6`, `Adapter6`, `Substitute6`, `Binding6`,
`Symbol6.{requires, binds, trip}`, `legacy_spelling`, `Render6::adapt`, `adapters6`, `Run::adapt`,
`requirements6`, `Entry::of_kingdom6`, and every `reqs[o.requirement]`.
The master 6 view generator must write the four row kinds directly:
- adapter Symbol rows (`adapter`, `prelude`, Argument and up Flow rows);
- Guard rows with real repair templates;
- spellings in the engine's placeholders;
- `size_names` with their lower bounds.

Render through `Decoder::render`. Charge cost from `Decoder::rescues`/`advice` plus the cost polynomials keyed by
Symbol. The interpreter executes adapter Symbols as ordinary functions. Delete the legacy forms; no bridge.

### Design gaps the next task will hit

**3. The template engine cannot spell a formula in the target language.** `src/typed_environment/template.rs:248`
(`value_text`) writes a formula in the formula language's own syntax. The test
`typed_environment/tests.rs:939` asserts `fit (min(k, n)) 0 ...`, which Futhark rejects; the design's section 5
needs `i64.min k n`. So the test checks a string, not a legal program.
*Fix:* make each operator's spelling a row (a per-kingdom spelling of `min`, `max`, `+`, `*` and literals, for
example Symbol rows of a formula spelling form). Change the test to compile, or compare against a text that
compiles.

**4. `{=formula}` renders `?` into program text.** `template.rs:293`. An unknown value is written as a literal `?`.
*Fix:* a `?` in a rendered value is "our software has a bug", unless a `when` with `known(...)` chose that
template.

**5. A must guard can be left with no applicable repair at run time.** `typed_environment.rs:617, 773` returns a
bug when every repair's guard or `when` fails. `schema.rs:1253` (`resolvable`) checks only that the substitutes
could exist, not that one repair is unconditional. So a row set can pass the check and still fail to decode some
gene, which breaks "every gene legal".
*Fix:* the table check requires the last repair of every must guard to have guard `true` and `when` `true`, and to
be a wrap or a substitute the node always has.

**6. The adapter-symbol check is switched off for kingdoms without adapter symbols.** `schema.rs:1246`
(`has_adapters`). Master 6 has none, so any `calls` name passes.
*Fix:* always require `calls` to name an adapter Symbol. This goes with finding 2.

**7. A non-literal `var.T` count binds nothing, silently.** `typed_environment.rs:658-661`. `?` or a symbolic count
gives 0 variables, so `var.T` falls back to the constant.
*Fix:* a non-literal count at decode time is a bug. Or the table check requires the count to be ground and
literal once the entry is known.

**8. The root repair's `{zero}` is the array's constant, not its element's.** `typed_environment.rs:850` passes
`zero: constant.clone()` for the entry's result type, where `Renderer::zero` (902-907) descends into the element.
The render test sidesteps it by writing a literal `0` (`tests.rs:933`).
*Fix:* use `r.zero(entry.output)`.

**9. Rendering an operand in a form loses state.** `typed_environment.rs:966-969, 989-992`. An operand rendered in
a form (`{k@form}`) uses a new `Renderer` with its own `Fresh`, `preludes` and `used`. Its fresh names can collide
with the outer ones, its adapters' preludes are dropped, and an alternative form used only there is never bound
at the head.
*Fix:* render through `&mut self`, or merge the sub-renderer's `fresh`, `preludes` and `used` back.

**10. The device subset is not enforced.** No code marks or checks the device subset of the formula language (3.4,
3.4.5). Today the list operators never reach a device, because no device path evaluates formulas at all.
*Fix:* when the bytecode lands (8.16), give a kingdom marked for the device a table check that rejects `list`,
`len`, `at`, `concat`, `find`, `for`, `in` and `all`. Until then, write in the module header that the formula
language is host only.

**11. The design's claim that a subtree's own variables keep their meaning holds only for least words.** A word
`w >= c` drawn on the GPU (uniform below 65,536) names `w mod c`, and `c` changes with the landing scope. So an
inner reference drifts under transposition and crossover unless it was written as its least word.
*Fix:* marriage and the reader write least words (they do, through `index_of`). State the limit in design 3.6 for
the GPU operators.

### Kingdom names in code rather than rows (check 1)

**12.** `futhark_table.rs:967-968`: every size name gets lower bound `1` in code, and the holder is found by the
name `"size"`. That is leak 6 (`at_least_one`) moved into the loader. `interp6.rs:291, 361, 472` and
`futhark_table.rs:1654` also branch on `d.name == "size"`. The `work` depth rule (`futhark_table.rs:~970`) is
Futhark's heavy-adapter rule rebuilt in the loader.
*Fix:* the view generator writes the `size_names` column and the repair guards. This goes with finding 2.

**13.** `geneframe.rs:113-128`: the schema's `Ty` keeps the kingdom families `Tsr`, `Futhark1d` and `Futhark`. This
was there before these commits; it is recorded because the schema should be fixed. `impl Entry { of_kingdom6 }`
(`futhark_table.rs:1857`) adds a Futhark constructor to the generic type.

The generic modules (`formula.rs`, `schema.rs`, `typed_environment.rs`, `template.rs`) hold no `Trips`,
`Substitute` (other than the template's three substitutes), `AdapterSite` or Futhark special case. The removal
claimed in 779412d1 is true for them.

### Correctness and hygiene

**14. A malformed catalogue becomes an empty one.** `schema.rs:225`:
`parse_catalogue(..).unwrap_or_default()`. At run time every property then reads as "not defined", which is a
fallback that hides the bug (Rules.md: no failovers). A test catches it under `cargo test`, but not a release
build with an edited TSV.
*Fix:* `expect` with "our software has a bug", or return `Result` from `catalogue()`.

**15. A misplaced doc comment.** `kingdom_contract.rs:287`: `is_variable`'s doc line now sits on `role`.

**16. The decoder checks the entry on every decode.** `typed_environment.rs:599`: `check_entry` parses and checks
every formula of the entry once per individual, on the hot path.
*Fix:* check once (a `CheckedEntry`), as the tables are.

**17. Integer overflow saturates silently.** `formula.rs:250, 280, 568, 571`: saturating arithmetic in normal forms
can turn an overflow into a wrong, plausible literal.
*Fix:* overflow is a bug (checked arithmetic giving `Err`, or `?`).

### Index words (check 5)

The fuller writers and readers agree on innermost-first through `in_scope` and `index_of`: `Environment::choose`
and `index_of`, the reader (`futhark_reader.rs:2245, 2263-2264, 2366, 2381-2388`, which keeps its outermost-first
*search* and then writes innermost-first words, which is consistent), `build6_named` (`futhark_sig.rs:3124`), and
the e-graph bridge `Egraph6::place` and `variables` (`futhark_saturation/master6.rs:247-251, 288`).
`REFERENCES6` and `tests/fixtures/futhark_m6_references.txt` are program text resolved by name, so they are
unaffected. The only inconsistent writer is phylu's graft (finding 1).

Gene words saved before 779412d1 (checkpoints, k-means run slots) now name other variables. Per the ruling they
are not migrated: delete them, and keep any champion as program text, read back through the reader.

### Tests (check 6)

- The tests do test what they claim for the formula language, the table check (one case per branch), the entry
  check, determinism across processes, measured-as-`?` in must guards, and the TSR, WGSL-flag and SQL-group
  examples.
- The pass-5 test pins illegal Futhark (finding 3) and avoids `{zero}` at the root (finding 8).
- No test grafts a binding subtree in phylu (finding 1).
- The reader's histogram floor was lowered from 166 to 164 to fit the new word order. That is acceptable for random
  genes, because `REFERENCES6` still renders byte for byte. But nothing shows that a resolved tree decodes to the
  same outcomes before and after the conversion.
- `random_genes_over_the_real_master6_views...` pins the coupling of finding 2: it asserts
  `t.guards.len() == reqs.len()` and indexes `reqs[*requirement]`.

### What matches the design (check 2 and check 3)

- Pass 2: down flows, with the decode order taken from the down flows' reads.
- Pass 3: `w mod c` counted from the innermost, the constant when `c = 0`.
- Pass 4: up flows in an order without cycles, `out.p` allowed.
- Must guards: in `order`, three-valued, measured values read as `?`. A must guard takes the first repair whose
  guard (in the site's environment) and `when` hold. A substitute replaces the node, which takes the substitute's
  properties.
- Advise guards: never read by the decoder, read by the cost with measured values as they are.
- The formula language is closed, with one spelling per formula, Kleene's three-valued connectives, `<=` reading
  `size_names` lower bounds, and fixed comparison semantics.
