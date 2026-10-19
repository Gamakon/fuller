// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var t0 = vec4<f32>(0.0);
    var t1 = vec4<f32>(0.0);
    var t2 = vec4<f32>(0.0);
    var t3 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < cfg.m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let i = lidx + 256u * j;
            let item = t + (i >> 4u);
            let off = (i & 15u) * 4u;
            var a = vec4<f32>(0.0);
            var b = vec4<f32>(0.0);
            if (item < cfg.m) {
                let base = item * d;
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    let c = colBase + off + q;
                    if (c < d) { a[q] = phi[base + c]; }
                    let lr = rowBase + off + q;
                    if (lr < cfg.r_rows && cfg.r_off + lr < d) { b[q] = phi[base + cfg.r_off + lr]; }
                }
            }
            wg_va[i] = a;
            wg_vb[i] = b;
        }
        workgroupBarrier();
        var v0 = vec4<f32>(0.0);
        var v1 = vec4<f32>(0.0);
        var v2 = vec4<f32>(0.0);
        var v3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            v0 = v0 + vec4<f32>(vb.x) * va;
            v1 = v1 + vec4<f32>(vb.y) * va;
            v2 = v2 + vec4<f32>(vb.z) * va;
            v3 = v3 + vec4<f32>(vb.w) * va;
        }
        t0 = t0 + v0;
        t1 = t1 + v1;
        t2 = t2 + v2;
        t3 = t3 + v3;
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + lid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var row = t0;
            if (j == 1u) { row = t1; }
            if (j == 2u) { row = t2; }
            if (j == 3u) { row = t3; }
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let c = colBase + lid.x * 4u + q;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + row[q];
                }
            }
        }
    }
}
