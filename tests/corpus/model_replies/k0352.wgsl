// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 4>;
    var pw: array<vec4<f32>, 4>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var s: u32 = 0u;
    var j: u32 = 0u;
    var i: u32 = 0u;
    for (i = 0u; i < 4u; i = i + 1u) {
        pv[i] = vec4<f32>(0.0);
    }
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (s = 0u; s < 2u; s = s + 1u) {
            let q = lidx + 256u * s;
            let kk = q >> 4u;
            let cv = (q & 15u) * 4u;
            let item = t + kk;
            var va = vec4<f32>(0.0);
            var vb = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (j = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + cv + j;
                    if (c < cfg.d) {
                        va[j] = phi[item * cfg.d + c];
                    }
                    let lr = rowBase + cv + j;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        vb[j] = phi[item * cfg.d + cfg.r_off + lr];
                    }
                }
            }
            wg_va[q] = va;
            wg_vb[q] = vb;
        }
        workgroupBarrier();
        for (i = 0u; i < 4u; i = i + 1u) {
            pw[i] = vec4<f32>(0.0);
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b = wg_vb[k * 16u + lid.y];
            pw[0] = pw[0] + vec4<f32>(b.x) * a;
            pw[1] = pw[1] + vec4<f32>(b.y) * a;
            pw[2] = pw[2] + vec4<f32>(b.z) * a;
            pw[3] = pw[3] + vec4<f32>(b.w) * a;
        }
        for (i = 0u; i < 4u; i = i + 1u) {
            pv[i] = pv[i] + pw[i];
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (j = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[i][j];
                }
            }
        }
    }
}
