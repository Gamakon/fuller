// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 256>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 4>;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    let ld_item = lidx >> 4u;
    let ld_q = (lidx & 15u) * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        let item = t + ld_item;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (item < cfg.m) {
            let base = item * cfg.d;
            for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                let c = colbase + ld_q + e;
                if (c < cfg.d) {
                    a[e] = phi[base + c];
                }
                let lr = rowbase + ld_q + e;
                if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                    b[e] = phi[base + cfg.r_off + lr];
                }
            }
        }
        wg_va[lidx] = a;
        wg_vb[lidx] = b;
        workgroupBarrier();
        var t0 = vec4<f32>(0.0);
        var t1 = vec4<f32>(0.0);
        var t2 = vec4<f32>(0.0);
        var t3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            t0 = t0 + vec4<f32>(vb.x) * va;
            t1 = t1 + vec4<f32>(vb.y) * va;
            t2 = t2 + vec4<f32>(vb.z) * va;
            t3 = t3 + vec4<f32>(vb.w) * va;
        }
        pv[0] = pv[0] + t0;
        pv[1] = pv[1] + t1;
        pv[2] = pv[2] + t2;
        pv[3] = pv[3] + t3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[i][j];
                }
            }
        }
    }
}
