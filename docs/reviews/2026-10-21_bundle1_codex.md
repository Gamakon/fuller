# Codex review, bundle 1 (fuller 99b12a06, 3f4cdf40, 0edc7794; phylu 6aea7aba, b7745f08), 2026-10-21

Findings (highest severity first):

1. **High — TSR “all laws encode” assertion is missing.**  
   [phylu/src/evolve/tsr_typed.rs:1011-1015](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/tsr_typed.rs:1011) only fails on `single_fail`; `tsr_cell` is printed but never asserted. Concrete failure: make every `OrderDepth` write-back return `Err` (or remove one required TSR row): `tsr_feynman_law_as_karva` still passes provided Single succeeds. This invalidates the commit’s TSR coverage claim.

2. **High — GPU law test can pass when neither gene encodes a law.**  
   [tsr_typed.rs:1069-1081](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/tsr_typed.rs:1069) appends a re-expressed gene only on `Ok`, then always evaluates/asserts the original source expression. Concrete failure: make `encode_law` return `Err` for both kingdoms; the batch contains only `nodes`, target agreement still passes, and no assertion requires either `single_gene` or `tsr_gene` to be `Ok`.

3. **Medium — `math_to_nodes[_n]` is globally behavior-changing, including rejection boundaries.**  
   [fuller/src/gpu_eval.rs:753-787](/Users/andrewmorgan/Dev/gamakon/fuller/src/gpu_eval.rs:753) rewrites for every caller, not just gene write-back: engine scoring/snap silently `continue`s on conversion failure, templates return `None`, and GPU fold/kitchen-sink paths change their evaluated tree. Concrete repro: `math_to_nodes_n("(Pow (Var \"x\") (Num 5.0))", vars, 3)` formerly succeeds with three nodes; it now expands to five nodes and fails the limit. Near `MAX_NODES`, valid candidates can therefore disappear silently. Scope the rewrite to the gene encoder, or explicitly accept/document the API change and test the boundary.

4. **Medium — numerical semantics are not preserved or tested.**  
   The old GPU `Pow` executes `pow(a,b)` [gpu_eval.rs:589-595](/Users/andrewmorgan/Dev/gamakon/fuller/src/gpu_eval.rs:589); the rewrite uses rounded `f32` multiplication chains. They are not bit-identical: for `x = 1.1f32`, `powf(x, 5)` versus `x² * x³` differs by one ULP in standard f32 implementations. Negative bases also change from WGSL `pow`’s problematic/undefined domain to valid integer multiplication, e.g. `Pow(-2, 4)`. That may be desirable for gene encoding, but is not acceptable as an unqualified change to scoring/snap/fold semantics. The new test uses `f64` host multiplication [gpu_eval.rs:2014-2035](/Users/andrewmorgan/Dev/gamakon/fuller/src/gpu_eval.rs:2014), so it cannot detect GPU f32 drift.

5. **Medium — data-driven tests can pass vacuously.**  
   [tsr_typed.rs:789-803](/Users/andrewmorgan/Dev/gamakon/phylu/src/evolve/tsr_typed.rs:789) skips missing datasets and asserts no minimum/expected law count; with the CSV present but all dataset paths absent, both law tests run zero cases and pass. The fuller parser test has the same skip pattern [infix.rs:393-400](/Users/andrewmorgan/Dev/gamakon/fuller/src/infix.rs:393). Assert the expected available count (98) and required named laws.

Parser precedence/associativity itself looks correct for the supported Python subset: power is right-associative; unary minus binds looser on the left and is permitted on the exponent (`-x**2`, `2**-x`, `a**b**c`). `3f4cdf40` is a correct manifest-only fix.
