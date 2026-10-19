// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);
    let item = lidx >> 4u;
    let c4 = (lidx & 15u) * 4u;
    let colBase = wid.x * 64u + c4;
    let rowLocal = wid.y * 64u + c4;
    let ntiles = (cfg.m + 15u) / 16u;
    for (var t: u32 = 0u; t < ntiles; t = t + 1u) {
        let tt = t * 16u + item;
        var a = vec4<f32>(0.0);
        var b = vec4<f32>(0.0);
        if (tt < cfg.m) {
            let base = tt * cfg.d;
            for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                let col = colBase + q;
                if (col < cfg.d) {
                    a[q] = phi[base + col];
                }
                let rl = rowLocal + q;
                if (rl < cfg.r_rows && (cfg.r_off + rl) < cfg.d) {
                    b[q] = phi[base + cfg.r_off + rl];
                }
            }
        }
        wg_va[lidx] = a;
        wg_vb[lidx] = b;
        workgroupBarrier();
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let va = wg_va[k * 16u + lid.x];
            let vb = wg_vb[k * 16u + lid.y];
            s0 = s0 + vec4<f32>(vb.x) * va;
            s1 = s1 + vec4<f32>(vb.y) * va;
            s2 = s2 + vec4<f32>(vb.z) * va;
            s3 = s3 + vec4<f32>(vb.w) * va;
        }
        acc[0] = acc[0] + s0;
        acc[1] = acc[1] + s1;
        acc[2] = acc[2] + s2;
        acc[3] = acc[3] + s3;
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x * 4u + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[i][j];
                }
            }
        }
    }
}
