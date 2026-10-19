// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let e = lidx + 256u * q;
            let tt = e >> 6u;
            let cc = e & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let c = col0 + cc;
                if (c < d) {
                    va = phi[t * d + c];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    vb = phi[t * d + cfg.r_off + lr];
                }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ba = k * 64u + tx * 4u;
            let bb = k * 64u + ty * 4u;
            let a = vec4<f32>(wg_a[ba & 1023u], wg_a[(ba + 1u) & 1023u], wg_a[(ba + 2u) & 1023u], wg_a[(ba + 3u) & 1023u]);
            let b0 = wg_b[bb & 1023u];
            let b1 = wg_b[(bb + 1u) & 1023u];
            let b2 = wg_b[(bb + 2u) & 1023u];
            let b3 = wg_b[(bb + 3u) & 1023u];
            a0 = a0 + b0 * a;
            a1 = a1 + b1 * a;
            a2 = a2 + b2 * a;
            a3 = a3 + b3 * a;
        }
        workgroupBarrier();
    }
    let cbase = col0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            var r = a0;
            if (i == 1u) { r = a1; }
            if (i == 2u) { r = a2; }
            if (i == 3u) { r = a3; }
            let base = lr * d + cbase;
            if (cbase < d) { h[base] = h[base] + r.x; }
            if (cbase + 1u < d) { h[base + 1u] = h[base + 1u] + r.y; }
            if (cbase + 2u < d) { h[base + 2u] = h[base + 2u] + r.z; }
            if (cbase + 3u < d) { h[base + 3u] = h[base + 3u] + r.w; }
        }
    }
}
