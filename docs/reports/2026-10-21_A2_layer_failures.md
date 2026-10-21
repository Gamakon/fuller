# A2: which legality layers random WGSL genes and children fail

**Context.** Task A2 of `docs/PLAN_true_gep_wgsl.md` (true multi-typed GEP for WGSL): the table of layer and reason failures of typed-initialiser genes and variation-operator children, which lists the legality rules still outside the types. **Status: stopped, no measurement reported.** The first version of the tool (phylu `examples/wgsl_layer_failures.rs`) drew on the host reference of the typed initialiser and operators; the lead ruled that A2 must use the device kernels, and that escape rows are removed first (A0), so the measurement waits for the revised brief.

Only small smoke runs were made (interface 0, 256 and 4,000 children per arm); they are not results and are not tabulated here.
