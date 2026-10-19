// grid 8 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pc: array<f32, 16>;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        pa[k] = 0.0;
    }
    let cbase = wid.x * 32u;
    let rbase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let it = idx >> 5u;
            let col = cbase + (idx & 31u);
            var v: f32 = 0.0;
            if ((t + it) < cfg.m && col < cfg.d) {
                v = phi[(t + it) * cfg.d + col];
            }
            wg_a[idx] = v;
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let it = idx >> 6u;
            let lr = rbase + (idx & 63u);
            var v: f32 = 0.0;
            if ((t + it) < cfg.m && lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                v = phi[(t + it) * cfg.d + cfg.r_off + lr];
            }
            wg_b[idx] = v;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pc[k] = 0.0;
        }
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a = vec4<f32>(wg_a[kk * 32u + lid.x * 4u], wg_a[kk * 32u + lid.x * 4u + 1u], wg_a[kk * 32u + lid.x * 4u + 2u], wg_a[kk * 32u + lid.x * 4u + 3u]);
            let b = vec4<f32>(wg_b[kk * 64u + lid.y * 4u], wg_b[kk * 64u + lid.y * 4u + 1u], wg_b[kk * 64u + lid.y * 4u + 2u], wg_b[kk * 64u + lid.y * 4u + 3u]);
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    pc[i * 4u + j] = pc[i * 4u + j] + b[i] * a[j];
                }
            }
        }
        let s = ((t >> 5u) & 3u) * 16u;
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            pa[s + k] = pa[s + k] + pc[k];
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
