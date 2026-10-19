// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 8>;
    var pw: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
        pw[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        // load A tile: 16 items x 128 columns
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let i = lidx + 256u * k2;
            let item = t + (i >> 5u);
            let c0 = wid.x * 128u + (i & 31u) * 4u;
            var a = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d + c0;
                if (c0 < cfg.d) { a.x = phi[base]; }
                if (c0 + 1u < cfg.d) { a.y = phi[base + 1u]; }
                if (c0 + 2u < cfg.d) { a.z = phi[base + 2u]; }
                if (c0 + 3u < cfg.d) { a.w = phi[base + 3u]; }
            }
            wg_va[i] = a;
        }
        // load B tile: 16 items x 64 rows
        {
            let i = lidx;
            let item = t + (i >> 4u);
            let lr0 = wid.y * 64u + (i & 15u) * 4u;
            let g0 = cfg.r_off + lr0;
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d + g0;
                if (lr0 < cfg.r_rows && g0 < cfg.d) { b.x = phi[base]; }
                if (lr0 + 1u < cfg.r_rows && g0 + 1u < cfg.d) { b.y = phi[base + 1u]; }
                if (lr0 + 2u < cfg.r_rows && g0 + 2u < cfg.d) { b.z = phi[base + 2u]; }
                if (lr0 + 3u < cfg.r_rows && g0 + 3u < cfg.d) { b.w = phi[base + 3u]; }
            }
            wg_vb[i] = b;
        }
        workgroupBarrier();
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        var v4 = vec4<f32>(0.0);
        var v5 = vec4<f32>(0.0);
        var v6 = vec4<f32>(0.0);
        var v7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 32u + lid.x];
            let b0 = wg_vb[k * 16u + lid.y * 2u];
            let b1 = wg_vb[k * 16u + lid.y * 2u + 1u];
            v0 = v0 + vec4<f32>(b0.x) * a;
            v1 = v1 + vec4<f32>(b0.y) * a;
            v2 = v2 + vec4<f32>(b0.z) * a;
            v3 = v3 + vec4<f32>(b0.w) * a;
            v4 = v4 + vec4<f32>(b1.x) * a;
            v5 = v5 + vec4<f32>(b1.y) * a;
            v6 = v6 + vec4<f32>(b1.z) * a;
            v7 = v7 + vec4<f32>(b1.w) * a;
        }
        if (((t >> 4u) & 1u) == 0u) {
            pv[0] = pv[0] + v0; pv[1] = pv[1] + v1; pv[2] = pv[2] + v2; pv[3] = pv[3] + v3;
            pv[4] = pv[4] + v4; pv[5] = pv[5] + v5; pv[6] = pv[6] + v6; pv[7] = pv[7] + v7;
        } else {
            pw[0] = pw[0] + v0; pw[1] = pw[1] + v1; pw[2] = pw[2] + v2; pw[3] = pw[3] + v3;
            pw[4] = pw[4] + v4; pw[5] = pw[5] + v5; pw[6] = pw[6] + v6; pw[7] = pw[7] + v7;
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            let s = pv[k] + pw[k];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + s[j];
                }
            }
        }
    }
}
