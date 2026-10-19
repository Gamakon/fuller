// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var t0: vec4<f32> = vec4<f32>(0.0); var t1: vec4<f32> = vec4<f32>(0.0); var t2: vec4<f32> = vec4<f32>(0.0); var t3: vec4<f32> = vec4<f32>(0.0);
    var t4: vec4<f32> = vec4<f32>(0.0); var t5: vec4<f32> = vec4<f32>(0.0); var t6: vec4<f32> = vec4<f32>(0.0); var t7: vec4<f32> = vec4<f32>(0.0);
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + 128u * k;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + c4 + j;
                    if (c < cfg.d) {
                        a[j] = phi[base + c];
                    }
                    let lr = rowBase + c4 + j;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        b[j] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0); var p1: vec4<f32> = vec4<f32>(0.0); var p2: vec4<f32> = vec4<f32>(0.0); var p3: vec4<f32> = vec4<f32>(0.0);
        var p4: vec4<f32> = vec4<f32>(0.0); var p5: vec4<f32> = vec4<f32>(0.0); var p6: vec4<f32> = vec4<f32>(0.0); var p7: vec4<f32> = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 1u) {
            let a0 = wg_va[kk * 16u + lid.x];
            let a1 = wg_va[kk * 16u + 8u + lid.x];
            let b = wg_vb[kk * 16u + lid.y];
            p0 = p0 + vec4<f32>(b.x) * a0;
            p1 = p1 + vec4<f32>(b.y) * a0;
            p2 = p2 + vec4<f32>(b.z) * a0;
            p3 = p3 + vec4<f32>(b.w) * a0;
            p4 = p4 + vec4<f32>(b.x) * a1;
            p5 = p5 + vec4<f32>(b.y) * a1;
            p6 = p6 + vec4<f32>(b.z) * a1;
            p7 = p7 + vec4<f32>(b.w) * a1;
        }
        t0 = t0 + p0; t1 = t1 + p1; t2 = t2 + p2; t3 = t3 + p3;
        t4 = t4 + p4; t5 = t5 + p5; t6 = t6 + p6; t7 = t7 + p7;
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 16>;
    pv[0] = t0; pv[1] = t1; pv[2] = t2; pv[3] = t3;
    pv[4] = t4; pv[5] = t5; pv[6] = t6; pv[7] = t7;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var hh: u32 = 0u; hh < 2u; hh = hh + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + hh * 32u + lid.x * 4u + j;
                    if (c < cfg.d) {
                        let o = lr * cfg.d + c;
                        h[o] = h[o] + pv[hh * 4u + i][j];
                    }
                }
            }
        }
    }
}
