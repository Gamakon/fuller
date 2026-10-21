// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var tot: array<vec4<f32>, 8>;
    var s: array<vec4<f32>, 8>;
    for (var q: u32 = 0u; q < 8u; q = q + 1u) {
        tot[q] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let q4 = (idx & 15u) * 4u;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + q4 + j;
                    if (c < d) {
                        va[j] = phi[base + c];
                    }
                    let lr = rowBase + q4 + j;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) {
                        vb[j] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx] = va;
            wg_vb[idx] = vb;
        }
        workgroupBarrier();
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            s[q] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a0 = wg_va[k * 16u + lid.x];
            let a1 = wg_va[k * 16u + 8u + lid.x];
            let bv = wg_vb[k * 16u + lid.y];
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let bi = vec4<f32>(bv[i]);
                s[i * 2u] = s[i * 2u] + bi * a0;
                s[i * 2u + 1u] = s[i * 2u + 1u] + bi * a1;
            }
        }
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            tot[q] = tot[q] + s[q];
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + hh * 32u + lid.x * 4u + j;
                    if (c < d) {
                        let o = lr * d + c;
                        h[o] = h[o] + tot[i * 2u + hh][j];
                    }
                }
            }
        }
    }
}
