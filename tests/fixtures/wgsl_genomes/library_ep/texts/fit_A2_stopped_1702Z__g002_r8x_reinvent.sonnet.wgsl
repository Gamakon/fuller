// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let cbase = wid.x * 64u;
    let rbase = wid.y * 64u;
    var pv: array<vec4<f32>, 4>;
    pv[0] = vec4<f32>(0.0);
    pv[1] = vec4<f32>(0.0);
    pv[2] = vec4<f32>(0.0);
    pv[3] = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let idx = lidx + 256u * j;
            let item = t + (idx >> 4u);
            let q = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let col = cbase + q + e;
                    if (col < cfg.d) {
                        a[e] = phi[item * cfg.d + col];
                    }
                    let row = rbase + q + e;
                    if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
                        b[e] = phi[item * cfg.d + cfg.r_off + row];
                    }
                }
            }
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var a0 = vec4<f32>(0.0);
        var a1 = vec4<f32>(0.0);
        var a2 = vec4<f32>(0.0);
        var a3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            a0 = a0 + vec4<f32>(vb.x) * va;
            a1 = a1 + vec4<f32>(vb.y) * va;
            a2 = a2 + vec4<f32>(vb.z) * va;
            a3 = a3 + vec4<f32>(vb.w) * va;
        }
        pv[0] = pv[0] + a0;
        pv[1] = pv[1] + a1;
        pv[2] = pv[2] + a2;
        pv[3] = pv[3] + a3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let acc = pv[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cbase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + acc[j];
                }
            }
        }
    }
}
