// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var v0: vec4<f32> = vec4<f32>(0.0);
    var v1: vec4<f32> = vec4<f32>(0.0);
    var v2: vec4<f32> = vec4<f32>(0.0);
    var v3: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let rbase = wid.y * 64u;
    let cbase = wid.x * 64u;
    for (var t: u32 = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let s = lidx + q * 256u;
            let it = t * 32u + (s / 16u);
            let c4 = (s % 16u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (it < cfg.m) {
                let base = it * d;
                for (var e: u32 = 0u; e < 4u; e = e + 1u) {
                    let lr = rbase + c4 + e;
                    let cc = cbase + c4 + e;
                    var av: f32 = 0.0;
                    var bv: f32 = 0.0;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                        av = phi[base + cfg.r_off + lr];
                    }
                    if (cc < d) {
                        bv = phi[base + cc];
                    }
                    a[e] = av;
                    b[e] = bv;
                }
            }
            wg_va[s] = a;
            wg_vb[s] = b;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[k * 16u + lid.y];
            let b = wg_vb[k * 16u + lid.x];
            v0 = v0 + a.x * b;
            v1 = v1 + a.y * b;
            v2 = v2 + a.z * b;
            v3 = v3 + a.w * b;
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rbase + lid.y * 4u + i;
        var acc = v0;
        if (i == 1u) { acc = v1; }
        if (i == 2u) { acc = v2; }
        if (i == 3u) { acc = v3; }
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let cc = cbase + lid.x * 4u + j;
                if (cc < d) {
                    let idx = lr * d + cc;
                    h[idx] = h[idx] + acc[j];
                }
            }
        }
    }
}
