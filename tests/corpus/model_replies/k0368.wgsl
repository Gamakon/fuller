// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let idx = lidx + 256u * k2;
            let item = t + (idx >> 4u);
            let off = (idx & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * cfg.d;
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let c = colBase + off + j;
                    if (c < cfg.d) {
                        a[j] = phi[base + c];
                    }
                    let lr = rowBase + off + j;
                    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                        b[j] = phi[base + cfg.r_off + lr];
                    }
                }
            }
            wg_va[idx] = a;
            wg_vb[idx] = b;
        }
        workgroupBarrier();
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            s0 = s0 + vec4<f32>(vb.x) * va;
            s1 = s1 + vec4<f32>(vb.y) * va;
            s2 = s2 + vec4<f32>(vb.z) * va;
            s3 = s3 + vec4<f32>(vb.w) * va;
        }
        let o = ((t >> 5u) & 1u) * 4u;
        pv[o] = pv[o] + s0;
        pv[o + 1u] = pv[o + 1u] + s1;
        pv[o + 2u] = pv[o + 2u] + s2;
        pv[o + 3u] = pv[o + 3u] + s3;
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let lr = rowBase + lid.y * 4u + k;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            let sm = pv[k] + pv[k + 4u];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + sm[j];
                }
            }
        }
    }
}
