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
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    // initial tile load into buffer 0
    for (var k = 0u; k < 2u; k = k + 1u) {
        let idx = lidx + k * 256u;
        let it = idx >> 6u;
        let cc = idx & 63u;
        var fb = 0.0;
        var fa = 0.0;
        if (it < m) {
            if (col0 + cc < d) {
                fb = phi[it * d + col0 + cc];
            }
            if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) {
                fa = phi[it * d + cfg.r_off + row0 + cc];
            }
        }
        wg_a[idx & 1023u] = fa;
        wg_b[idx & 1023u] = fb;
    }
    workgroupBarrier();

    var buf = 0u;
    for (var t = 0u; t < m; t = t + 8u) {
        let nb = buf ^ 1u;
        let tn = t + 8u;
        if (tn < m) {
            for (var k = 0u; k < 2u; k = k + 1u) {
                let idx = lidx + k * 256u;
                let it = tn + (idx >> 6u);
                let cc = idx & 63u;
                var fb = 0.0;
                var fa = 0.0;
                if (it < m) {
                    if (col0 + cc < d) {
                        fb = phi[it * d + col0 + cc];
                    }
                    if (row0 + cc < cfg.r_rows && cfg.r_off + row0 + cc < d) {
                        fa = phi[it * d + cfg.r_off + row0 + cc];
                    }
                }
                wg_a[(nb * 512u + idx) & 1023u] = fa;
                wg_b[(nb * 512u + idx) & 1023u] = fb;
            }
        }
        let base = buf * 512u;
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            let ro = base + kk * 64u + lid.y * 4u;
            let co = base + kk * 64u + lid.x * 4u;
            let b = vec4<f32>(wg_b[co & 1023u], wg_b[(co + 1u) & 1023u], wg_b[(co + 2u) & 1023u], wg_b[(co + 3u) & 1023u]);
            a0 = a0 + wg_a[ro & 1023u] * b;
            a1 = a1 + wg_a[(ro + 1u) & 1023u] * b;
            a2 = a2 + wg_a[(ro + 2u) & 1023u] * b;
            a3 = a3 + wg_a[(ro + 3u) & 1023u] * b;
        }
        workgroupBarrier();
        buf = nb;
    }

    var accs = array<vec4<f32>, 4>(a0, a1, a2, a3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = accs[i & 3u];
            let c = col0 + lid.x * 4u;
            if (c < d) { h[lr * d + c] = h[lr * d + c] + v.x; }
            if (c + 1u < d) { h[lr * d + c + 1u] = h[lr * d + c + 1u] + v.y; }
            if (c + 2u < d) { h[lr * d + c + 2u] = h[lr * d + c + 2u] + v.z; }
            if (c + 3u < d) { h[lr * d + c + 3u] = h[lr * d + c + 3u] + v.w; }
        }
    }
}
