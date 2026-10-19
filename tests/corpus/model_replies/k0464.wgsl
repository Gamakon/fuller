// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let tx = lid.x;
    let ty = lid.y;
    let col_base = wid.x * 64u;
    let row_base = wid.y * 64u;
    let li = lidx >> 4u;
    let lq = lidx & 15u;
    let lc = col_base + lq * 4u;
    let lr = row_base + lq * 4u;
    var e0: vec4<f32> = vec4<f32>(0.0); var e1: vec4<f32> = vec4<f32>(0.0); var e2: vec4<f32> = vec4<f32>(0.0); var e3: vec4<f32> = vec4<f32>(0.0);
    var e4: vec4<f32> = vec4<f32>(0.0); var e5: vec4<f32> = vec4<f32>(0.0); var e6: vec4<f32> = vec4<f32>(0.0); var e7: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0); var o1: vec4<f32> = vec4<f32>(0.0); var o2: vec4<f32> = vec4<f32>(0.0); var o3: vec4<f32> = vec4<f32>(0.0);
    var o4: vec4<f32> = vec4<f32>(0.0); var o5: vec4<f32> = vec4<f32>(0.0); var o6: vec4<f32> = vec4<f32>(0.0); var o7: vec4<f32> = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        var va: vec4<f32> = vec4<f32>(0.0);
        var vb: vec4<f32> = vec4<f32>(0.0);
        let it = t + li;
        if (it < cfg.m) {
            let base = it * cfg.d;
            if (lc < cfg.d) { vb.x = phi[base + lc]; }
            if (lc + 1u < cfg.d) { vb.y = phi[base + lc + 1u]; }
            if (lc + 2u < cfg.d) { vb.z = phi[base + lc + 2u]; }
            if (lc + 3u < cfg.d) { vb.w = phi[base + lc + 3u]; }
            if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) { va.x = phi[base + cfg.r_off + lr]; }
            if (lr + 1u < cfg.r_rows && cfg.r_off + lr + 1u < cfg.d) { va.y = phi[base + cfg.r_off + lr + 1u]; }
            if (lr + 2u < cfg.r_rows && cfg.r_off + lr + 2u < cfg.d) { va.z = phi[base + cfg.r_off + lr + 2u]; }
            if (lr + 3u < cfg.r_rows && cfg.r_off + lr + 3u < cfg.d) { va.w = phi[base + cfg.r_off + lr + 3u]; }
        }
        wg_va[lidx & 511u] = va;
        wg_vb[lidx & 511u] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 8u; k = k + 2u) {
            let a0 = wg_va[(k * 16u + ty * 2u) & 511u];
            let a1 = wg_va[(k * 16u + ty * 2u + 1u) & 511u];
            let b = wg_vb[(k * 16u + tx) & 511u];
            e0 = e0 + a0.x * b;
            e1 = e1 + a0.y * b;
            e2 = e2 + a0.z * b;
            e3 = e3 + a0.w * b;
            e4 = e4 + a1.x * b;
            e5 = e5 + a1.y * b;
            e6 = e6 + a1.z * b;
            e7 = e7 + a1.w * b;
            let c0 = wg_va[(k * 16u + 16u + ty * 2u) & 511u];
            let c1 = wg_va[(k * 16u + 16u + ty * 2u + 1u) & 511u];
            let bb = wg_vb[(k * 16u + 16u + tx) & 511u];
            o0 = o0 + c0.x * bb;
            o1 = o1 + c0.y * bb;
            o2 = o2 + c0.z * bb;
            o3 = o3 + c0.w * bb;
            o4 = o4 + c1.x * bb;
            o5 = o5 + c1.y * bb;
            o6 = o6 + c1.z * bb;
            o7 = o7 + c1.w * bb;
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 16>;
    pv[0] = e0 + o0;
    pv[1] = e1 + o1;
    pv[2] = e2 + o2;
    pv[3] = e3 + o3;
    pv[4] = e4 + o4;
    pv[5] = e5 + o5;
    pv[6] = e6 + o6;
    pv[7] = e7 + o7;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let rl = row_base + ty * 8u + i;
        if (rl < cfg.r_rows && cfg.r_off + rl < cfg.d) {
            let s = pv[i & 15u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col_base + tx * 4u + j;
                if (c < cfg.d) {
                    let idx = rl * cfg.d + c;
                    h[idx] = h[idx] + s[j];
                }
            }
        }
    }
}
