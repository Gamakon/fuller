// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 8>;
    for (var r = 0u; r < 8u; r = r + 1u) {
        acc[r & 7u] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (var t = 0u; t < m; t = t + 16u) {
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = lidx + 128u * j;
            let item = idx >> 6u;
            let cc = idx & 63u;
            let ti = t + item;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (ti < m) {
                let gc = col0 + cc;
                if (gc < d) {
                    va = phi[ti * d + gc];
                }
                let lr = row0 + cc;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                    vb = phi[ti * d + cfg.r_off + lr];
                }
            }
            wg_a[idx & 1023u] = va;
            wg_b[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let ab = k * 64u + lid.x * 4u;
            let a = vec4<f32>(wg_a[ab & 1023u], wg_a[(ab + 1u) & 1023u], wg_a[(ab + 2u) & 1023u], wg_a[(ab + 3u) & 1023u]);
            let bb = k * 64u + lid.y * 8u;
            for (var r = 0u; r < 8u; r = r + 1u) {
                let b = wg_b[(bb + r) & 1023u];
                acc[r & 7u] = acc[r & 7u] + vec4<f32>(b) * a;
            }
        }
        workgroupBarrier();
    }
    let cbase = col0 + lid.x * 4u;
    for (var r = 0u; r < 8u; r = r + 1u) {
        let lr = row0 + lid.y * 8u + r;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            let v = acc[r & 7u];
            let base = lr * d;
            if (cbase < d) { h[base + cbase] = h[base + cbase] + v.x; }
            if (cbase + 1u < d) { h[base + cbase + 1u] = h[base + cbase + 1u] + v.y; }
            if (cbase + 2u < d) { h[base + cbase + 2u] = h[base + cbase + 2u] + v.z; }
            if (cbase + 3u < d) { h[base + cbase + 3u] = h[base + cbase + 3u] + v.w; }
        }
    }
}
