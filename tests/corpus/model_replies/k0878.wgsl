// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + k * 256u;
            let item = t + (e >> 6u);
            let cc = e & 63u;
            var fb: f32 = 0.0;
            var fa: f32 = 0.0;
            if (item < m) {
                let col = colBase + cc;
                if (col < d) {
                    fb = phi[item * d + col];
                }
                let lr = rowBase + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    fa = phi[item * d + gr];
                }
            }
            wg_a[e & 1023u] = fa;
            wg_b[e & 1023u] = fb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let ab = kk * 64u + lid.y * 4u;
            let bb = kk * 64u + lid.x * 4u;
            let bv = vec4<f32>(wg_b[bb & 1023u], wg_b[(bb + 1u) & 1023u], wg_b[(bb + 2u) & 1023u], wg_b[(bb + 3u) & 1023u]);
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[(ab + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + a * bv;
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            let c0 = colBase + lid.x * 4u;
            let rb = lr * d;
            if (c0 < d) { h[rb + c0] = h[rb + c0] + v.x; }
            if (c0 + 1u < d) { h[rb + c0 + 1u] = h[rb + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[rb + c0 + 2u] = h[rb + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[rb + c0 + 3u] = h[rb + c0 + 3u] + v.w; }
        }
    }
}
