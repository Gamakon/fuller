// grid 16 16 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var e0: vec4<f32> = vec4<f32>(0.0); var e1: vec4<f32> = vec4<f32>(0.0); var e2: vec4<f32> = vec4<f32>(0.0); var e3: vec4<f32> = vec4<f32>(0.0);
    var e4: vec4<f32> = vec4<f32>(0.0); var e5: vec4<f32> = vec4<f32>(0.0); var e6: vec4<f32> = vec4<f32>(0.0); var e7: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0); var o1: vec4<f32> = vec4<f32>(0.0); var o2: vec4<f32> = vec4<f32>(0.0); var o3: vec4<f32> = vec4<f32>(0.0);
    var o4: vec4<f32> = vec4<f32>(0.0); var o5: vec4<f32> = vec4<f32>(0.0); var o6: vec4<f32> = vec4<f32>(0.0); var o7: vec4<f32> = vec4<f32>(0.0);
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 16u + idx / 32u;
            let rb = wid.y * 128u + (idx % 32u) * 4u;
            v0 = vec4<f32>(0.0);
            if (item < cfg.m) {
                for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    let row = rb + k2;
                    if (row < cfg.r_rows && (cfg.r_off + row) < cfg.d) {
                        v0[k2 & 3u] = phi[item * cfg.d + cfg.r_off + row];
                    }
                }
            }
            wg_va[idx & 511u] = v0;
        }
        {
            let itemb = t * 16u + lidx / 16u;
            let cb = wid.x * 64u + (lidx % 16u) * 4u;
            v1 = vec4<f32>(0.0);
            if (itemb < cfg.m) {
                for (var k2: u32 = 0u; k2 < 4u; k2 = k2 + 1u) {
                    if ((cb + k2) < cfg.d) {
                        v1[k2 & 3u] = phi[itemb * cfg.d + cb + k2];
                    }
                }
            }
            wg_vb[lidx & 511u] = v1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 2u) {
            let a0 = wg_va[(k * 32u + lid.y * 2u) & 511u];
            let a1 = wg_va[(k * 32u + lid.y * 2u + 1u) & 511u];
            let b = wg_vb[(k * 16u + lid.x) & 511u];
            e0 = e0 + vec4<f32>(a0.x) * b;
            e1 = e1 + vec4<f32>(a0.y) * b;
            e2 = e2 + vec4<f32>(a0.z) * b;
            e3 = e3 + vec4<f32>(a0.w) * b;
            e4 = e4 + vec4<f32>(a1.x) * b;
            e5 = e5 + vec4<f32>(a1.y) * b;
            e6 = e6 + vec4<f32>(a1.z) * b;
            e7 = e7 + vec4<f32>(a1.w) * b;
            let c0 = wg_va[(k * 32u + 32u + lid.y * 2u) & 511u];
            let c1 = wg_va[(k * 32u + 32u + lid.y * 2u + 1u) & 511u];
            let bb = wg_vb[(k * 16u + 16u + lid.x) & 511u];
            o0 = o0 + vec4<f32>(c0.x) * bb;
            o1 = o1 + vec4<f32>(c0.y) * bb;
            o2 = o2 + vec4<f32>(c0.z) * bb;
            o3 = o3 + vec4<f32>(c0.w) * bb;
            o4 = o4 + vec4<f32>(c1.x) * bb;
            o5 = o5 + vec4<f32>(c1.y) * bb;
            o6 = o6 + vec4<f32>(c1.z) * bb;
            o7 = o7 + vec4<f32>(c1.w) * bb;
        }
        workgroupBarrier();
    }
    pv[0] = e0 + o0;
    pv[1] = e1 + o1;
    pv[2] = e2 + o2;
    pv[3] = e3 + o3;
    pv[4] = e4 + o4;
    pv[5] = e5 + o5;
    pv[6] = e6 + o6;
    pv[7] = e7 + o7;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = gid.y * 8u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let rv = pv[i & 15u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + rv[j & 3u];
                }
            }
        }
    }
}
