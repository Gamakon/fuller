Gene 583 of seed 1 at the K = 2 table (commit 9b7484d7) hangs the Metal GPU at 256 invocations (d = 16, m = 4) although its work is
(32 + 64 * 787) node executions per invocation. naga_msl_reduced.metal is naga's MSL for the reduced kernel (gene583_reduced.wgsl),
run with Apple's compiler through the Metal API alone (run_msl.swift: swiftc -O run_msl.swift; ./run_msl naga_msl_reduced.metal 8),
no wgpu: HANG after 8 s. It needs all three parts (an empty counted loop, a determinant of a zero matrix, a break loop with a
dynamic-lane store); removing any one terminates in under a second. Without naga's loop-bound counters it still hangs.
RUN ONLY ALONE, WITH A TIME LIMIT, UNDER THE GPU LOCK: a hung kernel keeps the GPU busy after its process is killed.
