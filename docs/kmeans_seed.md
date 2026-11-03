# The k-means seed: UltraDim's three kernels as a gene of the Futhark kingdom

*fuller, 2026-11-03. Plan `docs/PLAN_futhark_kingdom.md` step 4c, items (c) and (d). The seed is `futhark_sig::kmeans_seed`
(`kingdoms/futhark/sig/kmeans/pool/seed.tsv`, its rendered program `seed.fut`); the proof is `futhark_sig::tests_kmeans_seed`.*

## Context

**Objective.** The k-means problem's seed is UltraDim's own k-means, `lib/projection-voting-kmeans/src/gpu_kmeans.rs`
(read 2026-11-03, read-only): the distance shader (line 555), `assign_points` (608) and `update_centroids` with its Kahan
compensation (657), translated into one program of the Futhark kingdom CONSTRUCT FOR CONSTRUCT, nothing substituted.
The textbook form that stood in the slot until the review of 2026-11-03 (`hist` sums, no compensation, an empty cluster
zeroed, the later centre on a tie) is a pool alternative (`pool/textbook.fut`), never the seed.

**Approach.** Every line of the three WGSL shaders is mapped below to the row of the master table that renders it. Where
the kingdom lacked a construct, the master got the row (version 5, commit 9cbeb292): a loop whose state is a TRIPLE
(sum, compensation, count), as the shader's one loop carries, and the KERNEL BOUNDARY, a `let` at the top of the
program naming the buffer one kernel writes and the next reads. The e-graph must not simplify the compensation: the f32
rule table admits only what step 4a admits (commutativity of `+` and `*`, `-(-a) = a`, the tuple projections), and a test
saturates the Kahan form and checks every variant's bits.

**Unique feature.** The translation is PROVEN against a straight Rust port of the three shaders in f32 with the same loop
order (`kmeans_converged::ultradim_step_f32`): on every stored case the assignments are identical as integers and the
centres bit-identical, through the host interpreter and through `futhark c`; and no farther from the f64 reference
(`reference64.fut`) than the port. The faithful typed view (i32 assignments beside f32) is measured: 14 times the size
with the pairs alone, 131,825 types with the triples and 237,325 with the kernel boundary, over the 65,536 type codes a
kingdom has, so it cannot be loaded at all; the problem's view holds the integers exactly in f32 instead (the decision,
with its numbers, is in "The two views").

**Status.** (c) done: master version 5. (d) done on the CPU: the seed in the problem's slot (`Seed::Translated`), proven on
8 of 8 cases. Not run: the WGSL oracle on the GPU (the lead's; the commands are at the end).

## The three shaders and the seed, side by side

The seed as a gene (`futhark_sig::kmeans_seed`, 131 functions):

```
(let ASSIGN UPDATE)
ASSIGN = (tab (snd.f32.f32 (loop1.k (pair DIST0 0)
           (psel (sub DIST s_f32) (pair DIST (ival.k i_k)) (pair s_f32 s2_f32)))))
DIST0  = (sqrt (loop.d 0 (add s_f32  (mul (sub P C0) (sub P C0)))))      P  = (get.d (row.n.d points i_n) i_d)
DIST   = (sqrt (loop.d 0 (add s3_f32 (mul (sub P C)  (sub P C)))))       C0 = (get.d (row.k.d centres zero) i_d)
                                                                         C  = (get.d (row.k.d centres i_k) i_d)
UPDATE = (tabrows (tab (sel (neg (thd LOOP)) (div (fst LOOP) (thd LOOP)) (get.d (row.k.d centres i_k) i_d))))
LOOP   = (loop.n zero (psel (sub (abs (sub (get.n s_a_n i_n) (ival.k i_k))) 1)
           (triple (add s_f32 (sub X s2_f32))
                   (sub (sub (add s_f32 (sub X s2_f32)) s_f32) (sub X s2_f32))
                   (add s3_f32 1))
           (triple s_f32 s2_f32 s3_f32)))                                X  = (get.d (row.n.d points i_n) i_d)
```

The rendered Futhark is `pool/seed.fut`; the kingdom's variables are named by class and ordinal (`s_f32`, `s2_f32`,
`s3_f32` the first, second and third f32 bound around a node; `i_n`, `i_d`, `i_k` the indices; `s_a_n` the let's array).

### Kernel 1, `compute_distances` (gpu_kmeans.rs 555-587)

| WGSL | the seed | row |
|---|---|---|
| `let idx = global_id.x; ... vector_idx = idx / n_centroids; centroid_idx = idx % n_centroids` | the thread over the (point, centre) pairs is the tabulate over the points (`tab`, binding `i_n`) and, inside it, the loop over the centres (`loop1.k`, binding `i_k`) | M0024, M0074 |
| `var distance: f32 = 0.0;` | `loop.d 0 ...`: the state starts at `0f32` | M0040 |
| `for (var d: u32 = 0u; d < dimensions; d = d + 1u)` | `loop.d`, the counter `i_d` over `d` in order | M0040 |
| `let v = vectors[vector_idx * dimensions + d];` | `get.d (row.n.d points i_n) i_d` | M0036, M0009 |
| `let c = centroids[centroid_idx * dimensions + d];` | `get.d (row.k.d centres i_k) i_d` (`zero` for centre 0) | M0036, M0009, M0021 |
| `distance += (v - c) * (v - c);` | `add s (mul (sub v c) (sub v c))`: the state plus the product of the two differences, each written out as the shader writes it | M0001, M0003, M0002 |
| `distances[...] = sqrt(distance);` | `sqrt (loop.d ...)` | M0061 |

### Kernel 2, `assign_points` (608-634)

| WGSL | the seed | row |
|---|---|---|
| `let vector_idx = global_id.x;` | `tab` over `n`, `i_n` | M0024 |
| `var best_centroid: u32 = 0u; var best_distance: f32 = distances[vector_idx * n_centroids];` | the loop's starting state `(pair DIST0 0)`: the distance to centre 0 (`centres[0]`, the index `zero`) and the index 0 (the f32 literal `0`) | M0068, M0016 |
| `for (var c: u32 = 1u; c < n_centroids; c = c + 1u)` | `loop1.k`: the counter over 1 .. k-1 (`for ti_k < k - 1 do let i_k = ti_k + 1`) | M0074 |
| `let distance = distances[vector_idx * n_centroids + c];` | `DIST`, the distance of centre `i_k` (see the note below) | |
| `if (distance < best_distance) { best_distance = distance; best_centroid = c; }` | `psel (sub DIST s_f32) (pair DIST (ival.k i_k)) (pair s_f32 s2_f32)`: the kingdom's one comparison is `e < 0`, and `d - best < 0` is `d < best` for every pair of f32 values (the infinities and a NaN included: a NaN distance never replaces, as in the shader); the first centre wins a tie | M0072, M0002, M0013 |
| `assignments[vector_idx] = best_centroid;` | `snd.f32.f32 (loop1.k ...)` inside the tabulate: the `[n]` array the `let` names (`s_a_n`) | M0070, M0085 |

### Kernel 3, `update_centroids` (657-694)

| WGSL | the seed | row |
|---|---|---|
| `centroid_idx = centroid_dim_idx / dimensions; dim_idx = centroid_dim_idx % dimensions;` | the thread over the (cluster, coordinate) cells is `tabrows` (binding `i_k`) over `tab` (binding `i_d`): a `[k][d]` array from its rows, a row from its cells | M0063, M0024 |
| `var sum: f32 = 0.0; var sum_c: f32 = 0.0; var count: u32 = 0u;` | `loop.n zero ...`: the TRIPLE state `(0f32, 0f32, 0f32)`, destructured into `s_f32` (sum), `s2_f32` (compensation), `s3_f32` (count) | M0083, M0081 |
| `for (var v: u32 = 0u; v < n_vectors; v = v + 1u)` | `loop.n`, the counter `i_n` over the points in order | M0083 |
| `if (assignments[v] == centroid_idx)` | `psel (sub (abs (sub (get.n s_a_n i_n) (ival.k i_k))) 1) STEP KEEP`: `|a - j| - 1 < 0` is `a == j` for integers; `s_a_n` is the let-bound assignments buffer | M0082, M0007, M0002, M0009, M0013 |
| `let term = vectors[v * dimensions + dim_idx];` | `X = get.d (row.n.d points i_n) i_d` | M0036, M0009 |
| `let y = term - sum_c;` | `sub X s2_f32` | M0002 |
| `let t = sum + y;` | `add s_f32 (sub X s2_f32)` | M0001 |
| `sum_c = (t - sum) - y;` | `sub (sub T s_f32) Y` with `T` and `Y` written out (no `let` below the top: see the note) | M0002 |
| `sum = t;` | the triple's first component is `T` | M0077 |
| `count = count + 1u;` | `add s3_f32 1` (the count as an f32 integer: see "The two views") | M0001 |
| `}` (a point of another cluster) | `KEEP = triple s_f32 s2_f32 s3_f32`, the state unchanged | M0077 |
| `if (count > 0u) { centroids[centroid_dim_idx] = sum / f32(count); }` | `sel (neg (thd LOOP)) (div (fst LOOP) (thd LOOP)) OLD`: `-count < 0` is `count > 0`; the division `sum / count` (the shader's `f32(count)` is the count already held as an f32); else `OLD = get.d (row.k.d centres i_k) i_d`, the previous centre, which is why the program's signature has the centres as an input | M0008, M0006, M0080, M0060, M0078 |

### The host loop (375-425) is the harness's

The iteration, the stopping rule (`|prev - inertia| < 1e-4`, 100 iterations) and the measures are `futhark_sig::kmeans_converged`
(`docs/KMEANS_FITNESS.md`); the program is the body of one iteration, `(points, centres) -> centres`. The host's
`enforce_cluster_balance` runs only under `validate_quality` and is not part of the kernels; `read_assignments` is the
kernel boundary.

## What is not one-to-one, and why

1. **The distance buffer is not materialised.** The shader writes an `n x k` matrix of distances (kernel 1) and reads it
   in kernel 2. The seed computes each distance where kernel 2 reads it (`DIST0` for the start, `DIST` for centre `i_k`),
   as `reference64.fut` and the Rust port do. The values are the same (each distance is one `sqrt` of one in-order sum,
   computed once a (point, centre) pair; the comparisons see the same numbers in the same order). Materialising it would
   need a class `a.n.k` the signature does not have and a second kernel boundary, and two boundaries square the view's
   size (every let class doubles the contexts below it: "The two views").
2. **A value read twice is written twice.** The kingdom has no `let` below the top: `DIST` appears in the compare and in
   the replacement, `Y = X - c` and `T = s + Y` are written out in the compensation, and the cell's loop appears three
   times (the sum, the count twice). The values are one; Futhark's CSE computes each once (the compiled program's loop is
   the interpreter's bit for bit, tested), and the e-graph's shared extraction sees them as one class. The seed is 131
   functions for it (the cell's loop is 22).
3. **The comparisons are spelled through `e < 0`.** The kingdom has one comparison: `distance < best_distance` is
   `distance - best < 0` (exact for every pair of f32 values: a difference of two distinct floats is never zero, and a
   NaN or `inf - inf` compares false both ways, as `<` does); `assignments[v] == centroid_idx` is `|a - j| - 1 < 0`
   (exact for integers); `count > 0u` is `-count < 0` (the count is never negative).
4. **The integers are held in f32** in the problem's view: the assignment (`best_centroid: u32`) is `f32.i64 i_k`, the
   count (`count: u32`) is `s3_f32 + 1`, and the shader's `f32(count)` is the identity. Both are exact below 2^24
   (16,777,216 centres or points a cluster; UltraDim's buffers are bounded by Metal's 128 MiB binding at `n x k x 4`
   bytes, so `n` is below 2^24 whenever `k >= 2`). The faithful two-element translation (`i32` assignments and count,
   `cvt.i32` for the division) differs from the seed in five rows (`snd.f32.i32`, `psel.i32`, `add s_i32 1`,
   `cvt.i32` twice); its view, the one the plan's item 1 asked for, cannot be loaded as a kingdom: the reason measured
   below.
5. **The arithmetic order is the shader's.** `distance += (v - c) * (v - c)` is a multiply then an add, `y = x - c;
   t = s + y; c = (t - s) - y` is three subtractions and an addition in that order, both sequential from 0 over the
   points in index order; `futhark c` emits each primitive as its own statement, so no C compiler contracts the multiply
   and the add into an FMA (the compiled program equals the host interpreter bit for bit on every case, tested). On the
   GPU the WGSL compiler may contract, which is the oracle run's question (below), not the translation's.

## The two views

The plan (4c, item 1) asked for an integer element for the assignments beside f32 in one view. Measured 2026-11-03,
`futhark_view` with the master at version 5:

| view (signature `points [n][d]f32, centres [k][d]f32 -> [k][d]f32`, depth 3) | types | rows | gene at N = 96 | layout |
|---|---:|---:|---:|---:|
| f32 only, master 3 (the view of 4a; its contexts 1 / 15 / 90 / 350 by depth) | 3,804 | 30,041 | 3,044,186 words | 1.5 s |
| f32 only, master 5, no kernel boundary | 8,662 | 65,412 | | |
| **f32 only, master 5, `lets a.n` (the problem's view)** | **14,359** | **110,871** | 13,709,362 words; **27.6 million at N = 192** | 25 s; 34 s at N = 192 |
| f32 and i32, master 4 (the pairs, no triple), no kernel boundary (contexts 1 / 36 / 432 / 3,000) | 53,773 | 416,671 | 62,693,641 words | 488 s |
| f32 and i32, master 5 (the triples), no kernel boundary | **131,825** (over the 65,536 type codes, `Ty::MAX_FUTHARK_DUALS`: the kingdom does not load) | 1,014,061 | | |
| f32 and i32, master 5, `lets a.i32.n` (the plan's view) | **237,325** (over the type codes) | | | |

The cause is the contexts: a context is the multiset of the classes bound around a node, and the second element doubles
the loop-state classes (its scalar and its rank-1 array over each dimension) and adds its pairs and triples, so the frames
a dimension offers go from 5 to 12 (version 4) and 16 (version 5) and the depth-3 contexts from 350 to 3,000 and beyond
(85 % of the types are depth-3 bodies). A gene of 62 million words is 250 MB, over Metal's 128 MiB storage-buffer
binding, and the layout takes 8 minutes; at version 5 the two-element view has 131,825 types without the kernel boundary
and 237,325 with it, over the 65,536 codes a `Ty::Futhark(u16)` can name, so the kingdom does not load
(`tests_kmeans_seed::the_two_element_view_...`; loading the with-let view alone takes 4.5 minutes). The one-element view with the
kernel boundary is 4.5 times the view of 4a; its gene at the problem's program size (192: the seed is 131 functions, its
planted slowdown 183) is 27.6 million words, 110 MB a gene, which fits one binding but is a cost the device population
must budget. The decision to hold the integers in f32 is a representation decision and is Andrew's to ratify; the i32
translation is five rows away from the seed (above) and the switch is one line of the specification (`sniff.py
--elements i32 --lets a.i32.n`) when the type codes and the layout can carry it.

The triple is master rows (`S:tuples3`, frames `L3`/`F3`), not a nested pair `((f32,f32),f32)`: the class grammar admits
a nested tuple's text, but the pair loop destructures its state into two SCALAR variables (the frames bind scalar
classes), so a nested pair as a loop state would have no variable for its inner pair; three scalar components in one frame
is what the shader's three `var`s are.

## The proof (`futhark_sig::tests_kmeans_seed`, CPU, 2026-11-03)

On every stored case (`tests/corpus/futhark_problems/kmeans/`: 5 train, 2 held, 1 test; scales 10^-3 to 10^5, mixed
scales, unequal clusters with two empty, the breaking case), against the Rust port of the three shaders in f32 with the
same loop order (`ultradim_step_f32`, Kahan on) and the f64 reference (`ultradim_step_f64` = `reference64.fut`, bit for bit):

| measure | count |
|---|---:|
| cases | 8 |
| assignments identical as integers (the seed's let-bound buffer, the port, the f64 reference) | 8 of 8 |
| centres bit-identical to the f32 port, host interpreter | 8 of 8 |
| centres bit-identical, compiled with `futhark c` (the plain program and the quality program of every case) | 8 of 8 |
| within the f64 reference's distance: the seed's largest cell error equals the port's on every case (4.2e-8 at scale 1, 3.4e-11 at 10^-3, 3.6e-5 at 10^3, 4.5e-3 at 10^5, 1.07e-6 on the breaking case, where the plain sum is 3.4e-5) | 8 of 8 |
| the planted slowdown (the assignments computed twice, `map2 f32.min a a`) the same bits | 8 of 8 |

The seed runs to UltraDim's stopping point on every fitness case compiled (the runner's control and `tests_checklist`,
`futhark c`); the table above was measured on the eight uniform magnitude cases of 2026-11-03 09:27Z, replaced the
same day by the dbpedia cases (`docs/KMEANS_FITNESS.md`; `tests_kmeans_seed` now proves the seed on 7 of 7 of them,
compiled, and on the small one interpreted). The host interpreter is not run on the dbpedia-sized cases (a minute a
step at 1,000 x 1,536 x 16).

## The GPU (the lead's)

The script `/private/tmp/claude-501/fbench/kmeans_gpu_controls.sh` runs these in order and writes the logs beside it:

```
# the seed through the converged judge on the GPU: the controls (the compiled seed CONVERGES on every fitness case,
# the program's own GPU time read from the shim's timestamps) and the in-run A/A (the seed as a candidate against
# itself: every input 1/2 if the device is deterministic)
cargo run --release --features gpu,wgsl --example futhark_evolve -- --problem kmeans --regime P --eval single --backend gpu --evaluations 1 --minutes 30 --seed 1 --out /private/tmp/claude-501/fbench/kmeans_seed_gpu
# the run from the seed, P regime, the k-means scheme (quality first; one interleaved pair a case; the budget: a
# candidate is 6 cases x about 20 iterations x 2 sides of one process each, of the order of two minutes)
cargo run --release --features gpu,wgsl --example futhark_evolve -- --problem kmeans --regime P --eval single --backend gpu --evaluations 200 --minutes 600 --seed 1 --out /private/tmp/claude-501/fbench/kmeans_P_s1
evowatch /private/tmp/claude-501/fbench/kmeans_P_s1
```

The WGSL oracle (plan item 4): the seed's assignments against UltraDim's own shaders through the wgpu harness on the same
cases; the centres' error class is the compensated one (above). Not run here: no GPU.
