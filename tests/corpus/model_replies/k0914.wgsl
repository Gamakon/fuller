// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = idx >> 6u;
            let cc = idx & 63u;
            let ti = t + item;
            var vb: f32 = 0.0;
            var va: f32 = 0.0;
            if (ti < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    vb = phi[ti * d + gc];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[ti * d + gr];
                }
            }
            wg_a[idx & 1023u] = va;
            wg_b[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let base = kk * 64u;
            let ra = base + lid.y * 4u;
            let rb = base + lid.x * 4u;
            let b0 = wg_b[(rb) & 1023u];
            let b1 = wg_b[(rb + 1u) & 1023u];
            let b2 = wg_b[(rb + 2u) & 1023u];
            let b3 = wg_b[(rb + 3u) & 1023u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let a = wg_a[(ra + j) & 1023u];
                acc[(j * 4u) & 15u] = fma(a, b0, acc[(j * 4u) & 15u]);
                acc[(j * 4u + 1u) & 15u] = fma(a, b1, acc[(j * 4u + 1u) & 15u]);
                acc[(j * 4u + 2u) & 15u] = fma(a, b2, acc[(j * 4u + 2u) & 15u]);
                acc[(j * 4u + 3u) & 15u] = fma(a, b3, acc[(j * 4u + 3u) & 15u]);
            }
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + lid.y * 4u + j;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = col0 + lid.x * 4u + i;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
