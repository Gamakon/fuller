// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u; var k: u32 = 0u; var k2: u32 = 0u; var j: u32 = 0u;
    for (k = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (k = 0u; k < 2u; k = k + 1u) {
            let idx = lidx + 256u * k;
            let row = idx >> 4u;
            let cv = idx & 15u;
            let tt = t + row;
            var va: vec4<f32> = vec4<f32>(0.0);
            var vb: vec4<f32> = vec4<f32>(0.0);
            if (tt < cfg.m) {
                for (j = 0u; j < 4u; j = j + 1u) {
                    let col = wid.x * 64u + cv * 4u + j;
                    let lr = wid.y * 64u + cv * 4u + j;
                    if (col < cfg.d) {
                        va[j] = phi[tt * cfg.d + col];
                    }
                    if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                        vb[j] = phi[tt * cfg.d + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx] = va;
            wg_vb[idx] = vb;
        }
        workgroupBarrier();
        var f0: f32 = 0.0; var f1: f32 = 0.0; var f2: f32 = 0.0; var f3: f32 = 0.0;
        var f4: f32 = 0.0; var f5: f32 = 0.0; var f6: f32 = 0.0; var f7: f32 = 0.0;
        var f8: f32 = 0.0; var f9: f32 = 0.0; var f10: f32 = 0.0; var f11: f32 = 0.0;
        var f12: f32 = 0.0; var f13: f32 = 0.0; var f14: f32 = 0.0; var f15: f32 = 0.0;
        for (k = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b = wg_vb[k * 16u + lid.y];
            f0 = f0 + b.x * a.x;
            f1 = f1 + b.x * a.y;
            f2 = f2 + b.x * a.z;
            f3 = f3 + b.x * a.w;
            f4 = f4 + b.y * a.x;
            f5 = f5 + b.y * a.y;
            f6 = f6 + b.y * a.z;
            f7 = f7 + b.y * a.w;
            f8 = f8 + b.z * a.x;
            f9 = f9 + b.z * a.y;
            f10 = f10 + b.z * a.z;
            f11 = f11 + b.z * a.w;
            f12 = f12 + b.w * a.x;
            f13 = f13 + b.w * a.y;
            f14 = f14 + b.w * a.z;
            f15 = f15 + b.w * a.w;
        }
        let p = ((t >> 5u) & 1u) * 16u;
        pa[p + 0u] = pa[p + 0u] + f0;
        pa[p + 1u] = pa[p + 1u] + f1;
        pa[p + 2u] = pa[p + 2u] + f2;
        pa[p + 3u] = pa[p + 3u] + f3;
        pa[p + 4u] = pa[p + 4u] + f4;
        pa[p + 5u] = pa[p + 5u] + f5;
        pa[p + 6u] = pa[p + 6u] + f6;
        pa[p + 7u] = pa[p + 7u] + f7;
        pa[p + 8u] = pa[p + 8u] + f8;
        pa[p + 9u] = pa[p + 9u] + f9;
        pa[p + 10u] = pa[p + 10u] + f10;
        pa[p + 11u] = pa[p + 11u] + f11;
        pa[p + 12u] = pa[p + 12u] + f12;
        pa[p + 13u] = pa[p + 13u] + f13;
        pa[p + 14u] = pa[p + 14u] + f14;
        pa[p + 15u] = pa[p + 15u] + f15;
        workgroupBarrier();
    }
    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
                let col = wid.x * 64u + lid.x * 4u + k2;
                if (col < cfg.d) {
                    let ix = lr * cfg.d + col;
                    h[ix] = h[ix] + (pa[k * 4u + k2] + pa[k * 4u + k2 + 16u]);
                }
            }
        }
    }
}
