// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var e0: vec4<f32> = vec4<f32>(0.0);
    var e1: vec4<f32> = vec4<f32>(0.0);
    var e2: vec4<f32> = vec4<f32>(0.0);
    var e3: vec4<f32> = vec4<f32>(0.0);
    var o0: vec4<f32> = vec4<f32>(0.0);
    var o1: vec4<f32> = vec4<f32>(0.0);
    var o2: vec4<f32> = vec4<f32>(0.0);
    var o3: vec4<f32> = vec4<f32>(0.0);
    var c0: vec4<f32> = vec4<f32>(0.0);
    var c1: vec4<f32> = vec4<f32>(0.0);
    var c2: vec4<f32> = vec4<f32>(0.0);
    var c3: vec4<f32> = vec4<f32>(0.0);
    var av: vec4<f32> = vec4<f32>(0.0);
    var bv: vec4<f32> = vec4<f32>(0.0);
    var pv: array<vec4<f32>, 16>;
    var e: u32 = 0u;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            e = lidx + 256u * k;
            f0 = 0.0;
            f1 = 0.0;
            if ((t + (e >> 6u)) < cfg.m) {
                if ((col0 + (e & 63u)) < cfg.d) {
                    f0 = phi[(t + (e >> 6u)) * cfg.d + col0 + (e & 63u)];
                }
                if (((row0 + (e & 63u)) < cfg.r_rows) && ((cfg.r_off + row0 + (e & 63u)) < cfg.d)) {
                    f1 = phi[(t + (e >> 6u)) * cfg.d + cfg.r_off + row0 + (e & 63u)];
                }
            }
            wg_va[(e >> 2u) & 511u][e & 3u] = f0;
            wg_vb[(e >> 2u) & 511u][e & 3u] = f1;
        }
        workgroupBarrier();
        c0 = vec4<f32>(0.0);
        c1 = vec4<f32>(0.0);
        c2 = vec4<f32>(0.0);
        c3 = vec4<f32>(0.0);
        for (k = 0u; k < 32u; k = k + 1u) {
            av = wg_va[((k * 16u) + lid.x) & 511u];
            bv = wg_vb[((k * 16u) + lid.y) & 511u];
            c0 = c0 + bv.x * av;
            c1 = c1 + bv.y * av;
            c2 = c2 + bv.z * av;
            c3 = c3 + bv.w * av;
        }
        if (((t >> 5u) & 1u) == 0u) {
            e0 = e0 + c0;
            e1 = e1 + c1;
            e2 = e2 + c2;
            e3 = e3 + c3;
        } else {
            o0 = o0 + c0;
            o1 = o1 + c1;
            o2 = o2 + c2;
            o3 = o3 + c3;
        }
        workgroupBarrier();
    }
    pv[0] = e0 + o0;
    pv[1] = e1 + o1;
    pv[2] = e2 + o2;
    pv[3] = e3 + o3;
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = row0 + lid.y * 4u + k;
        if ((lr < cfg.r_rows) && ((cfg.r_off + lr) < cfg.d)) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let c = col0 + lid.x * 4u + k2;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + pv[k & 15u][k2 & 3u];
                }
            }
        }
    }
}
