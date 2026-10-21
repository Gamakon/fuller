// grid 16 16 4 4
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
    var v6: vec4<f32> = vec4<f32>(0.0);
    var v7: vec4<f32> = vec4<f32>(0.0);
    var v8: vec4<f32> = vec4<f32>(0.0);
    var v9: vec4<f32> = vec4<f32>(0.0);
    var v10: vec4<f32> = vec4<f32>(0.0);
    var v11: vec4<f32> = vec4<f32>(0.0);
    var v12: vec4<f32> = vec4<f32>(0.0);
    var v13: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let i6 = lidx + 256u * k;
            let it = t + (i6 >> 4u);
            let c0 = wid.x * 64u + (i6 & 15u) * 4u;
            let r0 = wid.y * 64u + (i6 & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (it < m) {
                let base = it * d;
                if (c0 < d) { a.x = phi[base + c0]; }
                if (c0 + 1u < d) { a.y = phi[base + c0 + 1u]; }
                if (c0 + 2u < d) { a.z = phi[base + c0 + 2u]; }
                if (c0 + 3u < d) { a.w = phi[base + c0 + 3u]; }
                if (r0 < rr && ro + r0 < d) { b.x = phi[base + ro + r0]; }
                if (r0 + 1u < rr && ro + r0 + 1u < d) { b.y = phi[base + ro + r0 + 1u]; }
                if (r0 + 2u < rr && ro + r0 + 2u < d) { b.z = phi[base + ro + r0 + 2u]; }
                if (r0 + 3u < rr && ro + r0 + 3u < d) { b.w = phi[base + ro + r0 + 3u]; }
            }
            wg_va[i6 & 511u] = a;
            wg_vb[i6 & 511u] = b;
        }
        workgroupBarrier();
        v0 = vec4<f32>(0.0);
        v1 = vec4<f32>(0.0);
        v2 = vec4<f32>(0.0);
        v3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let v4 = wg_va[(k * 16u + lid.x) & 511u];
            let v5 = wg_vb[(k * 16u + lid.y) & 511u];
            v0 = v0 + vec4<f32>(v5.x) * v4;
            v1 = v1 + vec4<f32>(v5.y) * v4;
            v2 = v2 + vec4<f32>(v5.z) * v4;
            v3 = v3 + vec4<f32>(v5.w) * v4;
        }
        v6 = v6 + v0;
        v7 = v7 + v1;
        v8 = v8 + v2;
        v9 = v9 + v3;
        if (((t >> 5u) & 15u) == 15u) {
            v10 = v10 + v6;
            v11 = v11 + v7;
            v12 = v12 + v8;
            v13 = v13 + v9;
            v6 = vec4<f32>(0.0);
            v7 = vec4<f32>(0.0);
            v8 = vec4<f32>(0.0);
            v9 = vec4<f32>(0.0);
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0] = v10 + v6;
    pv[1] = v11 + v7;
    pv[2] = v12 + v8;
    pv[3] = v13 + v9;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = gid.y * 4u + k;
        if (lr < rr && ro + lr < d) {
            for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = gid.x * 4u + k2;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + pv[k][k2];
                }
            }
        }
    }
}
