// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 256>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var sa: array<vec4<f32>, 8>;
    var sb: array<vec4<f32>, 8>;
    var loc: array<vec4<f32>, 8>;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        sa[i] = vec4<f32>(0.0);
        sb[i] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let q = lidx + 256u * k;
            let gt = t + (q >> 5u);
            let col = wid.x * 128u + (q & 31u) * 4u;
            var v = vec4<f32>(0.0);
            if (gt < m) {
                let base = gt * d + col;
                if (col < d) { v.x = phi[base]; }
                if (col + 1u < d) { v.y = phi[base + 1u]; }
                if (col + 2u < d) { v.z = phi[base + 2u]; }
                if (col + 3u < d) { v.w = phi[base + 3u]; }
            }
            wg_a[q] = v;
        }
        {
            let gt = t + (lidx >> 4u);
            let row = wid.y * 64u + (lidx & 15u) * 4u;
            let gr = cfg.r_off + row;
            var v = vec4<f32>(0.0);
            if (gt < m) {
                let base = gt * d + gr;
                if (row < cfg.r_rows && gr < d) { v.x = phi[base]; }
                if (row + 1u < cfg.r_rows && gr + 1u < d) { v.y = phi[base + 1u]; }
                if (row + 2u < cfg.r_rows && gr + 2u < d) { v.z = phi[base + 2u]; }
                if (row + 3u < cfg.r_rows && gr + 3u < d) { v.w = phi[base + 3u]; }
            }
            wg_b[lidx] = v;
        }
        workgroupBarrier();
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            loc[i] = vec4<f32>(0.0);
        }
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_a[k * 32u + lid.x];
            let b0 = wg_b[k * 16u + lid.y * 2u];
            let b1 = wg_b[k * 16u + lid.y * 2u + 1u];
            loc[0] = loc[0] + vec4<f32>(b0.x) * a;
            loc[1] = loc[1] + vec4<f32>(b0.y) * a;
            loc[2] = loc[2] + vec4<f32>(b0.z) * a;
            loc[3] = loc[3] + vec4<f32>(b0.w) * a;
            loc[4] = loc[4] + vec4<f32>(b1.x) * a;
            loc[5] = loc[5] + vec4<f32>(b1.y) * a;
            loc[6] = loc[6] + vec4<f32>(b1.z) * a;
            loc[7] = loc[7] + vec4<f32>(b1.w) * a;
        }
        if (((t >> 4u) & 1u) == 0u) {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) { sa[i] = sa[i] + loc[i]; }
        } else {
            for (var i: u32 = 0u; i < 8u; i = i + 1u) { sb[i] = sb[i] + loc[i]; }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = gid.y * 8u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let s = sa[i] + sb[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = gid.x * 4u + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + s[j];
                }
            }
        }
    }
}
