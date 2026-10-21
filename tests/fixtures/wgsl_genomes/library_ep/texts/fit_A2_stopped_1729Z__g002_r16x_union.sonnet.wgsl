// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0); var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0); var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
    var b4 = vec4<f32>(0.0); var b5 = vec4<f32>(0.0); var b6 = vec4<f32>(0.0); var b7 = vec4<f32>(0.0);
    let d = cfg.d;
    let cb = wid.x * 64u;
    let rb = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let t0 = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let idx = lidx + 128u * k2;
                let it = t0 + (idx >> 4u);
                let c4 = (idx & 15u) * 4u;
                var xa = vec4<f32>(0.0);
                var xb = vec4<f32>(0.0);
                if (it < cfg.m) {
                    let base = it * d;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        let c = cb + c4 + j;
                        if (c < d) { xa[j] = phi[base + c]; }
                        let lr = rb + c4 + j;
                        if (lr < cfg.r_rows && cfg.r_off + lr < d) { xb[j] = phi[base + cfg.r_off + lr]; }
                    }
                }
                wg_va[idx] = xa;
                wg_vb[idx] = xb;
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let o = q * 32u;
                let r = o + lid.y * 2u;
                let x0 = wg_va[o + lid.x];
                let x1 = wg_va[o + 16u + lid.x];
                let y0 = wg_vb[r];
                let y1 = wg_vb[r + 1u];
                let y2 = wg_vb[r + 16u];
                let y3 = wg_vb[r + 17u];
                v0 = v0 + (vec4<f32>(y0.x) * x0 + vec4<f32>(y2.x) * x1);
                v1 = v1 + (vec4<f32>(y0.y) * x0 + vec4<f32>(y2.y) * x1);
                v2 = v2 + (vec4<f32>(y0.z) * x0 + vec4<f32>(y2.z) * x1);
                v3 = v3 + (vec4<f32>(y0.w) * x0 + vec4<f32>(y2.w) * x1);
                v4 = v4 + (vec4<f32>(y1.x) * x0 + vec4<f32>(y3.x) * x1);
                v5 = v5 + (vec4<f32>(y1.y) * x0 + vec4<f32>(y3.y) * x1);
                v6 = v6 + (vec4<f32>(y1.z) * x0 + vec4<f32>(y3.z) * x1);
                v7 = v7 + (vec4<f32>(y1.w) * x0 + vec4<f32>(y3.w) * x1);
            }
            if ((k & 1u) == 0u) {
                a0 = a0 + v0; a1 = a1 + v1; a2 = a2 + v2; a3 = a3 + v3;
                a4 = a4 + v4; a5 = a5 + v5; a6 = a6 + v6; a7 = a7 + v7;
            } else {
                b0 = b0 + v0; b1 = b1 + v1; b2 = b2 + v2; b3 = b3 + v3;
                b4 = b4 + v4; b5 = b5 + v5; b6 = b6 + v6; b7 = b7 + v7;
            }
            workgroupBarrier();
        }
    }
    var pv: array<vec4<f32>, 8>;
    pv[0] = a0 + b0; pv[1] = a1 + b1; pv[2] = a2 + b2; pv[3] = a3 + b3;
    pv[4] = a4 + b4; pv[5] = a5 + b5; pv[6] = a6 + b6; pv[7] = a7 + b7;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < d) {
                    let ix = lr * d + c;
                    h[ix] = h[ix] + pv[k][j];
                }
            }
        }
    }
}
