// grid 8 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 256>;
var<workgroup> wg_b: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let cbase = wid.x * 32u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        {
            let it = lidx >> 3u;
            let col = cbase + (lidx & 7u) * 4u;
            let ok = (t + it) < cfg.m;
            let base = (t + it) * cfg.d + col;
            var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (ok && col < cfg.d) { v.x = phi[base]; }
            if (ok && (col + 1u) < cfg.d) { v.y = phi[base + 1u]; }
            if (ok && (col + 2u) < cfg.d) { v.z = phi[base + 2u]; }
            if (ok && (col + 3u) < cfg.d) { v.w = phi[base + 3u]; }
            wg_a[lidx] = v;
        }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let it = idx >> 4u;
            let lr = rbase + (idx & 15u) * 4u;
            let ok = (t + it) < cfg.m;
            let base = (t + it) * cfg.d + cfg.r_off + lr;
            var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (ok && lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) { v.x = phi[base]; }
            if (ok && (lr + 1u) < cfg.r_rows && (cfg.r_off + lr + 1u) < cfg.d) { v.y = phi[base + 1u]; }
            if (ok && (lr + 2u) < cfg.r_rows && (cfg.r_off + lr + 2u) < cfg.d) { v.z = phi[base + 2u]; }
            if (ok && (lr + 3u) < cfg.r_rows && (cfg.r_off + lr + 3u) < cfg.d) { v.w = phi[base + 3u]; }
            wg_b[idx] = v;
        }
        workgroupBarrier();
        let s = ((t >> 4u) & 3u) * 16u;
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let a = wg_a[kk * 8u + lid.x];
            let b = wg_b[kk * 16u + lid.y];
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pa[s + i * 4u + j] = pa[s + i * 4u + j] + b[i] * a[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let q = i * 4u + j;
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + ((pa[q] + pa[q + 16u]) + (pa[q + 32u] + pa[q + 48u]));
                }
            }
        }
    }
}
