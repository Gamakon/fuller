// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var t0: vec4<f32> = vec4<f32>(0.0);
    var t1: vec4<f32> = vec4<f32>(0.0);
    var t2: vec4<f32> = vec4<f32>(0.0);
    var t3: vec4<f32> = vec4<f32>(0.0);
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let idx = lidx + 256u * k2;
            let item = t + (idx >> 4u);
            let c4 = (idx & 15u) * 4u;
            var fa: vec4<f32> = vec4<f32>(0.0);
            var fb: vec4<f32> = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    let cc = colBase + c4 + q;
                    if (cc < cfg.d) {
                        fa[q] = phi[base + cc];
                    }
                    let rr = rowBase + c4 + q;
                    if (rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) {
                        fb[q] = phi[base + cfg.r_off + rr];
                    }
                }
            }
            wg_va[idx] = fa;
            wg_vb[idx] = fb;
        }
        workgroupBarrier();
        var p0: vec4<f32> = vec4<f32>(0.0);
        var p1: vec4<f32> = vec4<f32>(0.0);
        var p2: vec4<f32> = vec4<f32>(0.0);
        var p3: vec4<f32> = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.x];
            let b = wg_vb[k * 16u + lid.y];
            p0 = p0 + vec4<f32>(b.x) * a;
            p1 = p1 + vec4<f32>(b.y) * a;
            p2 = p2 + vec4<f32>(b.z) * a;
            p3 = p3 + vec4<f32>(b.w) * a;
        }
        t0 = t0 + p0;
        t1 = t1 + p1;
        t2 = t2 + p2;
        t3 = t3 + p3;
        workgroupBarrier();
    }
    var tot: array<vec4<f32>, 4>;
    tot[0] = t0;
    tot[1] = t1;
    tot[2] = t2;
    tot[3] = t3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + tot[i][j];
                }
            }
        }
    }
}
