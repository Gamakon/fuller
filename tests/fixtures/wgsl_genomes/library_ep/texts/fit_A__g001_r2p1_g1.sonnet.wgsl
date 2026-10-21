// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pc: array<f32, 64>;
    pc[0] = 0.0;
    pc[1] = 0.0;
    pc[2] = 0.0;
    pc[3] = 0.0;
    let rb = wid.y * 32u;
    let cb = wid.x * 32u;
    let ld_k = lidx >> 5u;
    let ld_i = lidx & 31u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        var f0: f32 = 0.0;
        var f1: f32 = 0.0;
        if (t + ld_k < cfg.m) {
            if (rb + ld_i < cfg.r_rows && cfg.r_off + rb + ld_i < cfg.d) {
                f0 = phi[(t + ld_k) * cfg.d + cfg.r_off + rb + ld_i];
            }
            if (cb + ld_i < cfg.d) {
                f1 = phi[(t + ld_k) * cfg.d + cb + ld_i];
            }
        }
        wg_a[lidx] = f0;
        wg_b[lidx] = f1;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let a0 = wg_a[k * 32u + lid.y];
            let a1 = wg_a[k * 32u + lid.y + 16u];
            let b0 = wg_b[k * 32u + lid.x];
            let b1 = wg_b[k * 32u + lid.x + 16u];
            pc[0] = fma(a0, b0, pc[0]);
            pc[1] = fma(a0, b1, pc[1]);
            pc[2] = fma(a1, b0, pc[2]);
            pc[3] = fma(a1, b1, pc[3]);
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
        let row = rb + lid.y + 16u * i;
        if (row < cfg.r_rows && cfg.r_off + row < cfg.d) {
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let col = cb + lid.x + 16u * j;
                if (col < cfg.d) {
                    h[row * cfg.d + col] = h[row * cfg.d + col] + pc[i * 2u + j];
                }
            }
        }
    }
}
