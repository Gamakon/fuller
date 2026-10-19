// grid 32 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 256>;
@compute @workgroup_size(32, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pv: array<vec4<f32>, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        pv[k] = vec4<f32>(0.0);
    }
    let d = cfg.d;
    let m = cfg.m;
    let r_rows = cfg.r_rows;
    let r_off = cfg.r_off;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        for (var k2: u32 = 0u; k2 < 2u; k2 = k2 + 1u) {
            let s = lidx + 256u * k2;
            let item = t + (s >> 5u);
            let col = wid.x * 128u + (s & 31u) * 4u;
            var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
            if (item < m) {
                let base = item * d + col;
                if (col < d) { a0 = phi[base]; }
                if (col + 1u < d) { a1 = phi[base + 1u]; }
                if (col + 2u < d) { a2 = phi[base + 2u]; }
                if (col + 3u < d) { a3 = phi[base + 3u]; }
            }
            wg_va[s] = vec4<f32>(a0, a1, a2, a3);
        }
        {
            let s = lidx;
            let item = t + (s >> 4u);
            let lr = wid.y * 64u + (s & 15u) * 4u;
            let gr = r_off + lr;
            var b0: f32 = 0.0; var b1: f32 = 0.0; var b2: f32 = 0.0; var b3: f32 = 0.0;
            if (item < m) {
                let base = item * d + gr;
                if (lr < r_rows && gr < d) { b0 = phi[base]; }
                if (lr + 1u < r_rows && gr + 1u < d) { b1 = phi[base + 1u]; }
                if (lr + 2u < r_rows && gr + 2u < d) { b2 = phi[base + 2u]; }
                if (lr + 3u < r_rows && gr + 3u < d) { b3 = phi[base + 3u]; }
            }
            wg_vb[s] = vec4<f32>(b0, b1, b2, b3);
        }
        workgroupBarrier();
        var w0 = vec4<f32>(0.0); var w1 = vec4<f32>(0.0); var w2 = vec4<f32>(0.0); var w3 = vec4<f32>(0.0);
        var w4 = vec4<f32>(0.0); var w5 = vec4<f32>(0.0); var w6 = vec4<f32>(0.0); var w7 = vec4<f32>(0.0);
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = wg_va[k * 32u + lid.x];
            let ba = wg_vb[k * 16u + lid.y * 2u];
            let bb = wg_vb[k * 16u + lid.y * 2u + 1u];
            w0 = w0 + vec4<f32>(ba.x) * a;
            w1 = w1 + vec4<f32>(ba.y) * a;
            w2 = w2 + vec4<f32>(ba.z) * a;
            w3 = w3 + vec4<f32>(ba.w) * a;
            w4 = w4 + vec4<f32>(bb.x) * a;
            w5 = w5 + vec4<f32>(bb.y) * a;
            w6 = w6 + vec4<f32>(bb.z) * a;
            w7 = w7 + vec4<f32>(bb.w) * a;
        }
        pv[0] = pv[0] + w0;
        pv[1] = pv[1] + w1;
        pv[2] = pv[2] + w2;
        pv[3] = pv[3] + w3;
        pv[4] = pv[4] + w4;
        pv[5] = pv[5] + w5;
        pv[6] = pv[6] + w6;
        pv[7] = pv[7] + w7;
        workgroupBarrier();
    }
    let col0 = wid.x * 128u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let lr = row0 + k;
        if (lr < r_rows && r_off + lr < d) {
            let v = pv[k];
            let idx = lr * d + col0;
            if (col0 < d) { h[idx] = h[idx] + v.x; }
            if (col0 + 1u < d) { h[idx + 1u] = h[idx + 1u] + v.y; }
            if (col0 + 2u < d) { h[idx + 2u] = h[idx + 2u] + v.z; }
            if (col0 + 3u < d) { h[idx + 3u] = h[idx + 3u] + v.w; }
        }
    }
}
