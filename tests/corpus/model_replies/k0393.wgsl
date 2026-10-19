// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var nt: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = t + e / 16u;
            let q = (e % 16u) * 4u;
            var a: vec4<f32> = vec4<f32>(0.0);
            var b: vec4<f32> = vec4<f32>(0.0);
            if (it < cfg.m) {
                let base = it * cfg.d;
                for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    let lr = wid.y * 64u + q + k2;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        a[k2] = phi[base + cfg.r_off + lr];
                    }
                    let c = wid.x * 64u + q + k2;
                    if (c < cfg.d) {
                        b[k2] = phi[base + c];
                    }
                }
            }
            wg_va[e & 511u] = a;
            wg_vb[e & 511u] = b;
        }
        workgroupBarrier();
        for (k = 0u; k < 4u; k = k + 1u) {
            pv[k] = vec4<f32>(0.0);
        }
        for (k = 0u; k < 32u; k = k + 1u) {
            let av = wg_va[(k * 16u + lid.y) & 511u];
            let bv = wg_vb[(k * 16u + lid.x) & 511u];
            pv[0] = pv[0] + av.x * bv;
            pv[1] = pv[1] + av.y * bv;
            pv[2] = pv[2] + av.z * bv;
            pv[3] = pv[3] + av.w * bv;
        }
        for (k = 0u; k < 4u; k = k + 1u) {
            pv[k + 4u] = pv[k + 4u] + pv[k];
        }
        nt = nt + 1u;
        if (nt == 16u) {
            for (k = 0u; k < 4u; k = k + 1u) {
                pv[k + 8u] = pv[k + 8u] + pv[k + 4u];
                pv[k + 4u] = vec4<f32>(0.0);
            }
            nt = 0u;
        }
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        pv[k + 8u] = pv[k + 8u] + pv[k + 4u];
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
            if ((((gid.y * 4u) + min(k, 3u))) < cfg.r_rows && (((gid.x * 4u) + min(k2, 3u))) < cfg.d && (cfg.r_off + (((gid.y * 4u) + min(k, 3u)))) < cfg.d) { h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] = h[((((gid.y * 4u) + min(k, 3u))) * cfg.d + (((gid.x * 4u) + min(k2, 3u))))] + pv[k + 8u][k2]; }
        }
    }
}
