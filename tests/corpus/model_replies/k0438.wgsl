// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    var v4: vec4<f32> = vec4<f32>(0.0);
    var v5: vec4<f32> = vec4<f32>(0.0);
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var va: vec4<f32> = vec4<f32>(0.0);
    var vb0: vec4<f32> = vec4<f32>(0.0);
    var vb1: vec4<f32> = vec4<f32>(0.0);
    var vs: vec4<f32> = vec4<f32>(0.0);
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 128u;
    for (t = 0u; t < ((cfg.m + 15u) / 16u); t = t + 1u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let e = lidx + k * 256u;
            let item = t * 16u + e / 32u;
            v0 = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    let col = col0 + (e % 32u) * 4u + k2;
                    if (col < cfg.d) {
                        v0[k2 & 3u] = phi[item * cfg.d + col];
                    }
                }
            }
            wg_vb[e & 511u] = v0;
        }
        {
            let e = lidx;
            let item = t * 16u + e / 16u;
            v1 = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    let row = row0 + (e % 16u) * 4u + k2;
                    if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
                        v1[k2 & 3u] = phi[item * cfg.d + cfg.r_off + row];
                    }
                }
            }
            wg_va[e & 511u] = v1;
        }
        workgroupBarrier();
        for (k = 0u; k < 16u; k = k + 1u) {
            va = wg_va[(k * 16u + lid.y) & 511u];
            vb0 = wg_vb[(k * 32u + lid.x * 2u) & 511u];
            vb1 = wg_vb[(k * 32u + lid.x * 2u + 1u) & 511u];
            v2 = v2 + vec4<f32>(va.x) * vb0;
            v3 = v3 + vec4<f32>(va.x) * vb1;
            v4 = v4 + vec4<f32>(va.y) * vb0;
            v5 = v5 + vec4<f32>(va.y) * vb1;
            v6 = v6 + vec4<f32>(va.z) * vb0;
            v7 = v7 + vec4<f32>(va.z) * vb1;
            v8 = v8 + vec4<f32>(va.w) * vb0;
            v9 = v9 + vec4<f32>(va.w) * vb1;
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let r = gid.y * 4u + k;
        if (r < cfg.r_rows && (cfg.r_off + r) < cfg.d) {
            for (k2 = 0u; k2 < 8u; k2 = k2 + 1u) {
                let c = gid.x * 8u + k2;
                if (c < cfg.d) {
                    vs = v2;
                    if (k == 0u && k2 >= 4u) { vs = v3; }
                    if (k == 1u) { vs = v4; if (k2 >= 4u) { vs = v5; } }
                    if (k == 2u) { vs = v6; if (k2 >= 4u) { vs = v7; } }
                    if (k == 3u) { vs = v8; if (k2 >= 4u) { vs = v9; } }
                    let idx = r * cfg.d + c;
                    h[idx] = h[idx] + vs[k2 & 3u];
                }
            }
        }
    }
}
