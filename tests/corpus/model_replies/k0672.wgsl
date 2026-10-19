// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 8>;
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    var s4 = vec4<f32>(0.0); var s5 = vec4<f32>(0.0); var s6 = vec4<f32>(0.0); var s7 = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let tb = t + k * 32u;
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let i7 = lidx + 128u * k2;
                let item = tb + (i7 >> 4u);
                let cc = (i7 & 15u) * 4u;
                var a = vec4<f32>(0.0);
                var b = vec4<f32>(0.0);
                if (item < cfg.m) {
                    let base = item * cfg.d;
                    let c0 = wid.x * 64u + cc;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        if (c0 + j < cfg.d) { a[j] = phi[base + c0 + j]; }
                    }
                    let l0 = wid.y * 64u + cc;
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        if (l0 + j < cfg.r_rows && cfg.r_off + l0 + j < cfg.d) { b[j] = phi[base + cfg.r_off + l0 + j]; }
                    }
                }
                wg_va[i7 & 511u] = a;
                wg_vb[i7 & 511u] = b;
            }
            workgroupBarrier();
            var v0 = vec4<f32>(0.0); var v1 = vec4<f32>(0.0); var v2 = vec4<f32>(0.0); var v3 = vec4<f32>(0.0);
            var v4 = vec4<f32>(0.0); var v5 = vec4<f32>(0.0); var v6 = vec4<f32>(0.0); var v7 = vec4<f32>(0.0);
            for (var q: u32 = 0u; q < 16u; q = q + 1u) {
                let i11 = q * 32u;
                let i12 = i11 + lid.y * 2u;
                let v8 = wg_va[(i11 + lid.x) & 511u];
                let v9 = wg_va[(i11 + 16u + lid.x) & 511u];
                let v10 = wg_vb[i12 & 511u];
                let v11 = wg_vb[(i12 + 1u) & 511u];
                let v12 = wg_vb[(i12 + 16u) & 511u];
                let v13 = wg_vb[(i12 + 17u) & 511u];
                v0 = v0 + (vec4<f32>(v10.x) * v8 + vec4<f32>(v12.x) * v9);
                v1 = v1 + (vec4<f32>(v10.y) * v8 + vec4<f32>(v12.y) * v9);
                v2 = v2 + (vec4<f32>(v10.z) * v8 + vec4<f32>(v12.z) * v9);
                v3 = v3 + (vec4<f32>(v10.w) * v8 + vec4<f32>(v12.w) * v9);
                v4 = v4 + (vec4<f32>(v11.x) * v8 + vec4<f32>(v13.x) * v9);
                v5 = v5 + (vec4<f32>(v11.y) * v8 + vec4<f32>(v13.y) * v9);
                v6 = v6 + (vec4<f32>(v11.z) * v8 + vec4<f32>(v13.z) * v9);
                v7 = v7 + (vec4<f32>(v11.w) * v8 + vec4<f32>(v13.w) * v9);
            }
            s0 = s0 + v0; s1 = s1 + v1; s2 = s2 + v2; s3 = s3 + v3;
            s4 = s4 + v4; s5 = s5 + v5; s6 = s6 + v6; s7 = s7 + v7;
            workgroupBarrier();
        }
        pv[0] = pv[0] + s0; pv[1] = pv[1] + s1; pv[2] = pv[2] + s2; pv[3] = pv[3] + s3;
        pv[4] = pv[4] + s4; pv[5] = pv[5] + s5; pv[6] = pv[6] + s6; pv[7] = pv[7] + s7;
        s0 = vec4<f32>(0.0); s1 = vec4<f32>(0.0); s2 = vec4<f32>(0.0); s3 = vec4<f32>(0.0);
        s4 = vec4<f32>(0.0); s5 = vec4<f32>(0.0); s6 = vec4<f32>(0.0); s7 = vec4<f32>(0.0);
    }
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = gid.y * 8u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
