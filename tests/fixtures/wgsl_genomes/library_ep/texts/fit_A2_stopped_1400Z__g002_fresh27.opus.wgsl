// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;
    var s00: f32 = 0.0; var s01: f32 = 0.0; var s10: f32 = 0.0; var s11: f32 = 0.0;
    var m00: f32 = 0.0; var m01: f32 = 0.0; var m10: f32 = 0.0; var m11: f32 = 0.0;
    var cnt: u32 = 0u;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 32u) {
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let e = lidx + 256u * j;
            let it = t0 + (e >> 5u);
            let cc = e & 31u;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m) {
                let base = it * cfg.d;
                let gc = col0 + cc;
                if (gc < cfg.d) { va = phi[base + gc]; }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) { vb = phi[base + gr]; }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        var p00: f32 = 0.0; var p01: f32 = 0.0; var p10: f32 = 0.0; var p11: f32 = 0.0;
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let o = kk * 32u;
            let a0 = wg_a[(o + lid.x) & 1023u];
            let a1 = wg_a[(o + lid.x + 16u) & 1023u];
            let b0 = wg_b[(o + lid.y) & 1023u];
            let b1 = wg_b[(o + lid.y + 16u) & 1023u];
            p00 = fma(b0, a0, p00);
            p01 = fma(b0, a1, p01);
            p10 = fma(b1, a0, p10);
            p11 = fma(b1, a1, p11);
        }
        m00 = m00 + p00; m01 = m01 + p01; m10 = m10 + p10; m11 = m11 + p11;
        cnt = cnt + 1u;
        if (cnt >= 32u) {
            s00 = s00 + m00; s01 = s01 + m01; s10 = s10 + m10; s11 = s11 + m11;
            m00 = 0.0; m01 = 0.0; m10 = 0.0; m11 = 0.0;
            cnt = 0u;
        }
        workgroupBarrier();
    }
    s00 = s00 + m00; s01 = s01 + m01; s10 = s10 + m10; s11 = s11 + m11;
    let c0 = col0 + lid.x;
    let c1 = c0 + 16u;
    let lr0 = row0 + lid.y;
    let lr1 = lr0 + 16u;
    if (lr0 < cfg.r_rows && cfg.r_off + lr0 < cfg.d) {
        if (c0 < cfg.d) { let idx = lr0 * cfg.d + c0; h[idx] = h[idx] + s00; }
        if (c1 < cfg.d) { let idx = lr0 * cfg.d + c1; h[idx] = h[idx] + s01; }
    }
    if (lr1 < cfg.r_rows && cfg.r_off + lr1 < cfg.d) {
        if (c0 < cfg.d) { let idx = lr1 * cfg.d + c0; h[idx] = h[idx] + s10; }
        if (c1 < cfg.d) { let idx = lr1 * cfg.d + c1; h[idx] = h[idx] + s11; }
    }
}
