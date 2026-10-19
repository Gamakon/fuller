// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_va: array<vec4<f32>, 512>;
var<workgroup> wg_vb: array<vec4<f32>, 512>;

fn store_row(lr: u32, c0: u32, v: vec4<f32>) {
    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
        let base = lr * cfg.d + c0;
        if (c0 < cfg.d) { h[base] = h[base] + v.x; }
        if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
        if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
        if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 128u * k;
            let tt = t + (e >> 4u);
            let off = (e & 15u) * 4u;
            var v0 = vec4<f32>(0.0);
            var v1 = vec4<f32>(0.0);
            if (tt < m) {
                let rb = tt * d;
                let c = colbase + off;
                if (c < d) { v0.x = phi[rb + c]; }
                if (c + 1u < d) { v0.y = phi[rb + c + 1u]; }
                if (c + 2u < d) { v0.z = phi[rb + c + 2u]; }
                if (c + 3u < d) { v0.w = phi[rb + c + 3u]; }
                let lr = rowbase + off;
                let g = cfg.r_off + lr;
                if (lr < cfg.r_rows && g < d) { v1.x = phi[rb + g]; }
                if (lr + 1u < cfg.r_rows && g + 1u < d) { v1.y = phi[rb + g + 1u]; }
                if (lr + 2u < cfg.r_rows && g + 2u < d) { v1.z = phi[rb + g + 2u]; }
                if (lr + 3u < cfg.r_rows && g + 3u < d) { v1.w = phi[rb + g + 3u]; }
            }
            wg_va[e & 511u] = v0;
            wg_vb[e & 511u] = v1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let a = wg_va[(k * 16u + lid.x) & 511u];
            let b0 = wg_vb[(k * 16u + lid.y * 2u) & 511u];
            let b1 = wg_vb[(k * 16u + lid.y * 2u + 1u) & 511u];
            a0 = a0 + b0.x * a;
            a1 = a1 + b0.y * a;
            a2 = a2 + b0.z * a;
            a3 = a3 + b0.w * a;
            a4 = a4 + b1.x * a;
            a5 = a5 + b1.y * a;
            a6 = a6 + b1.z * a;
            a7 = a7 + b1.w * a;
        }
        workgroupBarrier();
    }
    let r0 = rowbase + lid.y * 8u;
    let c0 = colbase + lid.x * 4u;
    store_row(r0, c0, a0);
    store_row(r0 + 1u, c0, a1);
    store_row(r0 + 2u, c0, a2);
    store_row(r0 + 3u, c0, a3);
    store_row(r0 + 4u, c0, a4);
    store_row(r0 + 5u, c0, a5);
    store_row(r0 + 6u, c0, a6);
    store_row(r0 + 7u, c0, a7);
}
