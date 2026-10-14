TASK. Write a compute kernel that computes the weighted Gram matrix H = Phi^T Phi over a band of items, accumulating into a resident output band, exactly as the reference kernel below does, but faster.

CONTRACT (fixed, the harness binds exactly these):
```wgsl
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;         // m x d, item-major: item t's d values are phi[t*d .. t*d+d]
@group(0) @binding(1) var<storage, read_write> h: array<f32>;     // r_rows x d, the output-row band: h[lr*d + c] for local row lr in [0, r_rows), column c in [0, d)
@group(0) @binding(2) var<uniform> cfg: Cfg;
```
SEMANTICS. For every local row lr < r_rows and column c < d, with the global row gr = r_off + lr: h[lr*d + c] += sum over t in [0, m) of phi[t*d + gr] * phi[t*d + c]. The driver dispatches the kernel once per item band; h already holds earlier bands' partial sums, so the kernel must READ-ADD-WRITE h, never overwrite it. Rows outside [0, r_rows) and columns outside [0, d) must not be written. gr can exceed d - 1 when r_off + r_rows > d: such rows must not be read or written. A cell may be written by exactly one thread. d and m are arbitrary (not multiples of anything); r_rows <= d.

ENTRY POINT. `@compute @workgroup_size(X, Y, 1) fn main(...)` with any builtins you need. The FIRST LINE of your kernel must be the comment `// grid X Y BX BY` where X, Y are your workgroup size and BX, BY are the cells per thread along columns and rows (1 if one cell per thread). The harness dispatches ceil(d / (X*BX)) x ceil(r_rows / (Y*BY)) x 1 workgroups. Use workgroup_id, local_invocation_id or global_invocation_id consistently with that.

CONSTRAINTS. Only the three bindings above. No other global variables except var<workgroup> arrays (at most 16 KiB in total). workgroupBarrier() only in uniform control flow (never after a conditional return; guard with if-blocks instead). Every loop must terminate: bound item loops by cfg.m, tile loops by a literal. f32 arithmetic only; the result must match the reference to within f32 rounding (reordering the sum is fine).

SPEED NOTE (measured on this machine). Index every var<workgroup> and private array with a mask into its length, e.g. wg_a[(i) & 2047u] for an array of 2,048 and acc[(q) & 3u] for one of 4 (lengths powers of two): the Metal compiler then drops its bounds checks. The same kernel ran 1.7x faster masked (221.6 ms unmasked, 127.8 ms masked, at d 2048, m 65536). Storage buffers (phi, h) cannot be masked; guard them as the reference does.

## Strategies
- one thread per output cell, as the reference kernel does, but with any improvement you see
- register blocking: each thread computes a 2 x 2 block of H, reading each phi value once per block row and column
- register blocking: each thread computes a 4 x 4 block of H with vec4 loads of four consecutive columns
- a workgroup tile: stage a tile of items' phi values (the block's rows and columns) in var<workgroup> memory with workgroupBarrier, then accumulate from shared memory
- vector loads: read four consecutive columns of phi as a vec4 and keep vec4 accumulators
- split accumulators: several independent partial sums over interleaved items, added at the end
- an unrolled item loop: four or eight items per iteration with the remainder handled after
- combine a workgroup tile of items in shared memory with a 4 x 4 register block per thread
- combine register blocking with vec4 loads and an unrolled item loop
- a structure unlike the reference kernel: choose any correct organisation of the work you believe is fastest on an Apple M3 Max GPU under Metal
