// grid 32 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wvb: array<vec4<f32>, 512>;
var<workgroup> wva: array<vec4<f32>, 512>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0: vec4<f32> = vec4<f32>(0.0);
    var a1: vec4<f32> = vec4<f32>(0.0);
    var a2: vec4<f32> = vec4<f32>(0.0);
    var a3: vec4<f32> = vec4<f32>(0.0);
    let d = cfg.d;
    let colbase = wid.x * 128u;
    let rowbase = wid.y * 32u;
    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 2u; q = q + 1u) {
            let e = lidx + q * 256u;
            let tt = e >> 5u;
            let cx = e & 31u;
            let t = t0 + tt;
            let col = colbase + cx * 4u;
            var v = vec4<f32>(0.0);
            if (t < cfg.m) {
                let base = t * d + col;
                if (col < d) { v.x = phi[base]; }
                if (col + 1u < d) { v.y = phi[base + 1u]; }
                if (col + 2u < d) { v.z = phi[base + 2u]; }
                if (col + 3u < d) { v.w = phi[base + 3u]; }
            }
            wvb[e] = v;
        }
        if (lidx < 128u) {
            let tt = lidx >> 3u;
            let ry = lidx & 7u;
            let t = t0 + tt;
            let lr = rowbase + ry * 4u;
            var v = vec4<f32>(0.0);
            if (t < cfg.m) {
                let base = t * d + cfg.r_off + lr;
                if (lr < cfg.r_rows && cfg.r_off + lr < d) { v.x = phi[base]; }
                if (lr + 1u < cfg.r_rows && cfg.r_off + lr + 1u < d) { v.y = phi[base + 1u]; }
                if (lr + 2u < cfg.r_rows && cfg.r_off + lr + 2u < d) { v.z = phi[base + 2u]; }
                if (lr + 3u < cfg.r_rows && cfg.r_off + lr + 3u < d) { v.w = phi[base + 3u]; }
            }
            wva[lidx] = v;
        }
        workgroupBarrier();
        for (var g: u32 = 0u; g < 4u; g = g + 1u) {
            let k = g * 4u;
            let b0 = wvb[(k) * 32u + lid.x];
            let b1 = wvb[(k + 1u) * 32u + lid.x];
            let b2 = wvb[(k + 2u) * 32u + lid.x];
            let b3 = wvb[(k + 3u) * 32u + lid.x];
            let r0 = wva[(k) * 8u + lid.y];
            let r1 = wva[(k + 1u) * 8u + lid.y];
            let r2 = wva[(k + 2u) * 8u + lid.y];
            let r3 = wva[(k + 3u) * 8u + lid.y];
            a0 = a0 + ((b0 * r0.x + b1 * r1.x) + (b2 * r2.x + b3 * r3.x));
            a1 = a1 + ((b0 * r0.y + b1 * r1.y) + (b2 * r2.y + b3 * r3.y));
            a2 = a2 + ((b0 * r0.z + b1 * r1.z) + (b2 * r2.z + b3 * r3.z));
            a3 = a3 + ((b0 * r0.w + b1 * r1.w) + (b2 * r2.w + b3 * r3.w));
        }
        workgroupBarrier();
    }
    var pv: array<vec4<f32>, 4>;
    pv[0] = a0;
    pv[1] = a1;
    pv[2] = a2;
    pv[3] = a3;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let cb = colbase + lid.x * 4u;
            let rv = pv[i];
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = cb + j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + rv[j];
                }
            }
        }
    }
}
