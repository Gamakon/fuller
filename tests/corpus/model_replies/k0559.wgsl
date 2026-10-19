// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> ta: array<f32, 1024>;
var<workgroup> tb: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let col0 = wid.x * 32u;
    let row0 = wid.y * 32u;
    let d = cfg.d;
    let m = cfg.m;
    var o00: f32 = 0.0; var o01: f32 = 0.0; var o10: f32 = 0.0; var o11: f32 = 0.0;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let i = lidx + 256u * k;
            let it = t0 + (i >> 5u);
            let j = i & 31u;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (it < m) {
                let base = it * d;
                let c = col0 + j;
                if (c < d) { a = phi[base + c]; }
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) { b = phi[base + gr]; }
            }
            ta[i] = a;
            tb[i] = b;
        }
        workgroupBarrier();
        var a00: f32 = 0.0; var a01: f32 = 0.0; var a10: f32 = 0.0; var a11: f32 = 0.0;
        var e00: f32 = 0.0; var e01: f32 = 0.0; var e10: f32 = 0.0; var e11: f32 = 0.0;
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 2u) {
            let p = kk * 32u;
            let c0 = ta[p + lid.x];
            let c1 = ta[p + lid.x + 16u];
            let r0 = tb[p + lid.y];
            let r1 = tb[p + lid.y + 16u];
            a00 = a00 + r0 * c0;
            a01 = a01 + r0 * c1;
            a10 = a10 + r1 * c0;
            a11 = a11 + r1 * c1;
            let q = p + 32u;
            let d0 = ta[q + lid.x];
            let d1 = ta[q + lid.x + 16u];
            let s0 = tb[q + lid.y];
            let s1 = tb[q + lid.y + 16u];
            e00 = e00 + s0 * d0;
            e01 = e01 + s0 * d1;
            e10 = e10 + s1 * d0;
            e11 = e11 + s1 * d1;
        }
        o00 = o00 + (a00 + e00);
        o01 = o01 + (a01 + e01);
        o10 = o10 + (a10 + e10);
        o11 = o11 + (a11 + e11);
        workgroupBarrier();
    }
    let cA = col0 + lid.x;
    let cB = cA + 16u;
    let lA = row0 + lid.y;
    let lB = lA + 16u;
    if (lA < cfg.r_rows && cfg.r_off + lA < d) {
        if (cA < d) { let idx = lA * d + cA; h[idx] = h[idx] + o00; }
        if (cB < d) { let idx = lA * d + cB; h[idx] = h[idx] + o01; }
    }
    if (lB < cfg.r_rows && cfg.r_off + lB < d) {
        if (cA < d) { let idx = lB * d + cA; h[idx] = h[idx] + o10; }
        if (cB < d) { let idx = lB * d + cB; h[idx] = h[idx] + o11; }
    }
}
