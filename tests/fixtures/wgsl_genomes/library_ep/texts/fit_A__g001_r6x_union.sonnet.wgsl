// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 16>;
    var pb: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        pa[q] = 0.0;
        pb[q] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let item = t + (idx >> 6u);
            let c = idx & 63u;
            let row = wid.y * 64u + c;
            let col = wid.x * 64u + c;
            var f0: f32 = 0.0;
            var f1: f32 = 0.0;
            if (item < m) {
                if (row < cfg.r_rows && (cfg.r_off + row) < d) {
                    f0 = phi[item * d + cfg.r_off + row];
                }
                if (col < d) {
                    f1 = phi[item * d + col];
                }
            }
            wg_a[idx] = f0;
            wg_b[idx] = f1;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 8u; kk = kk + 1u) {
            let o0 = (kk * 2u) * 64u;
            let o1 = o0 + 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a0 = wg_a[o0 + lid.y + 16u * i];
                let a1 = wg_a[o1 + lid.y + 16u * i];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let b0 = wg_b[o0 + lid.x + 16u * j];
                    let b1 = wg_b[o1 + lid.x + 16u * j];
                    pa[i * 4u + j] = pa[i * 4u + j] + a0 * b0;
                    pb[i * 4u + j] = pb[i * 4u + j] + a1 * b1;
                }
            }
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let row = wid.y * 64u + lid.y + 16u * k;
        if (row < cfg.r_rows && (cfg.r_off + row) < d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x + 16u * k2;
                if (col < d) {
                    h[row * d + col] = h[row * d + col] + (pa[k * 4u + k2] + pb[k * 4u + k2]);
                }
            }
        }
    }
}
