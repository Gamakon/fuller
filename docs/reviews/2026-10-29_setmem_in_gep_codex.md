No: keep Futhark memory allocation, aliasing, and `SetMem` out of this genotype for now.

The signature kingdom expresses source-level, value-semantic Futhark. Its types are values plus binder contexts (`a.n`, `i32`, `ix.n`, and `.../L-...` loop contexts); its loop rows express an array state, not a particular backing allocation. `SetMem` is introduced later by Futhark’s compiler. Making it genetic would expose an implementation choice as a semantic choice, enlarge the search space, and make a legal source gene depend on a particular compiler lowering. The present failure is therefore a backend bug/limitation, not a kingdom infeasibility.

Also, strictly speaking, it is not a homeotic gene. `MultiTypedGEP.html` reserves “homeotic gene” for the chromosome linker; population-wide shared definitions are ADFs. More importantly, an ADF is immutable, population-wide, referentially transparent, and memoised by value. A memory block is mutable storage scoped to one invocation/kernel; its identity can change across loop iterations and paths. An ADF `href` means “evaluate this definition”; `SetMem` means “from this program point, this name denotes that storage.” Similar naming mechanism, fundamentally different object.

The existing backend already makes this distinction well:

- Top-level non-register `SetMem` is a compile-time rename only outside control flow.
- `ScalarSpace` register-memory `SetMem` is a WGSL value copy anywhere, including loop rotation.
- The remaining failures are controlled aliases of other memory spaces, where a global rename is unsound because the binding can differ by branch or iteration.

That is exactly the right boundary to fix in the compiler.

## If memory were deliberately made genetic

It needs a separate imperative/memory kingdom, not a few extra source-Futhark rows. Types must describe an ordered resource state, including named block identity, extent, element type, address space, view layout, and ownership.

A minimal sketch in the table style would be:

```tsv
# types.tsv
ref    dual                         slot  form       literals root meaning
M0001  mem.dev.i32.n:p0:free        mem   mem.dev...          no   device block B0, n i32s, free at phase p0
M0002  mem.dev.i32.n:p0:owned       mem   mem.dev...          no   B0 uniquely owned at phase p0
M0003  view.dev.i32.n:B0:row:ro     arr   a.n                 no   read-only row-major view of B0
M0004  view.dev.i32.n:B0:row:uniq   arr   a.n                 no   unique writable view of B0
M0005  state<p0,B0=free>            state state               yes  no   memory environment before a command
M0006  state<p1,B0=owned>           state state               no   no   environment after allocation
M0007  state<p1,x=B0,y=dead>        state state               no   no   y moved into x; no second writable alias
```

```tsv
# functions.tsv
ref      name                    kind      out                         args
F01001   alloc.B0.n              function  state<p1,B0=owned>          state<p0,B0=free>
F01002   view.owned.B0           function  view...B0:row:uniq          state<p1,B0=owned>
F01003   rebind.x.from.y         function  state<p1,x=B0,y=dead>       state<p1,y=B0,x=dead>
F01004   read.view               function  i32:p1                      view...B0:row:ro;ix.n:p1
F01005   inplace.map.B0          function  state<p2,B0=owned>          state<p1,B0=owned>;kernel.elemwise
F01006   branch.same-poststate   function  state<p2,S>                 bool:p1;cmd<p1,p2,S>;cmd<p1,p2,S>
F01007   loop.fixed-state        function  state<p1,S>                 state<p1,S>;body<p1,S,p1,S>
```

The key is not `block<space, element, size>` alone. It is the changing `state<...>` type. `rebind` must move the unique writable capability, not create two writable aliases. Branches must return the same post-state. A loop body must return the same state shape it receives. An in-place row consumes the unique input view and returns the successor state. Read-only views can be copied; writable views cannot.

This is much larger than current typed GEP. Per-slot output types prove tree edges, but they cannot by themselves prove “this particular resource is used once” across siblings. The type universe must carry the whole finite resource environment, or commands must be represented as state-transforming continuations. Block IDs, phase, and live-set then become types. That is feasible for a deliberately bounded WGSL kernel language; it is a poor fit for the current source-Futhark signature kingdom.

## Consequences if built

The Karva walk would no longer start from merely an output array type. It would walk a `Cmd<state_in,state_out>` root, with frames per resource-state type. The ordered signature already helps, but `Seq`, `If`, and `Loop` would be the important rows: their child state types enforce resource flow.

Variation stays legal only within exact state-transformer domains. Crossover may exchange only equal `Cmd<in,out>` domains. A mutation cannot choose another block name unless its output state agrees. Fold may share pure read-only layout expressions or immutable command templates, but must not turn a live memory block into a population ADF: that would accidentally share storage between candidates. Memory reuse should be a local compiler pass over one expressed command graph, not population-DAG folding.

This would likely make search worse initially: far more types, wider genes, and constrained paths. It could improve search only if the goal is to discover explicit memory schedules in a purpose-built kernel language and memory footprint is part of fitness.

## Recommendation and cheapest test

1. Treat the current ~1-in-6 failures as a WebGPU backend coverage defect. Extend lowering for controlled `SetMem`, rather than changing genes.
2. First classify every failing emitted ImpCode `SetMem` by memory space, loop/branch location, and whether it is loop-carried. Patch the most common one with an SSA-style lowering: distinct bindings per branch/iteration plus an explicit join/loop-carried binding, rather than a mutable global rename.
3. Compile the same saved histogram population before and after; success means those legal source genes all lower and compile. No search redesign, no GPU experiment needed.
4. Only if explicit memory schedule variation later shows a planted win that source-level Futhark cannot express should you prototype the small state-threaded WGSL kingdom above.

In short: `SetMem` resembles an `href` only as a name-resolution operation. It is not an ADF/homeotic definition, and making it one would violate the current population DAG’s purity and lifetime assumptions.