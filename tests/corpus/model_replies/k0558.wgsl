// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sa: array<vec4<f32>, 512>;
var<workgroup> sb: array<vec4<f32>, 512>;
fn store_row(lr: u32, cb: u32, lo: vec4<f32>, hi: vec4<f32>) {
    let d = cfg.d;
    if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
        let base = lr * d;
        if (cb < d) { h[base + cb] = h[base + cb] + lo.x; }
        if (cb + 1u < d) { h[base + cb + 1u] = h[base + cb + 1u] + lo.y; }
        if (cb + 2u < d) { h[base + cb + 2u] = h[base + cb + 2u] + lo.z; }
        if (cb + 3u < d) { h[base + cb + 3u] = h[base + cb + 3u] + lo.w; }
        if (cb + 4u < d) { h[base + cb + 4u] = h[base + cb + 4u] + hi.x; }
        if (cb + 5u < d) { h[base + cb + 5u] = h[base + cb + 5u] + hi.y; }
        if (cb + 6u < d) { h[base + cb + 6u] = h[base + cb + 6u] + hi.z; }
        if (cb + 7u < d) { h[base + cb + 7u] = h[base + cb + 7u] + hi.w; }
    }
}
@compute @workgroup_size(8, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var t0 = vec4<f32>(0.0); var t1 = vec4<f32>(0.0); var t2 = vec4<f32>(0.0); var t3 = vec4<f32>(0.0);
    var t4 = vec4<f32>(0.0); var t5 = vec4<f32>(0.0); var t6 = vec4<f32>(0.0); var t7 = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 128u) {
        var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0); var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
        var b4 = vec4<f32>(0.0); var b5 = vec4<f32>(0.0); var b6 = vec4<f32>(0.0); var b7 = vec4<f32>(0.0);
        for (var kc: u32 = 0u; kc < 4u; kc = kc + 1u) {
            let tb = t + kc * 32u;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let i = lidx + 128u * j;
                let it = tb + (i >> 4u);
                let q = (i & 15u) * 4u;
                var va = vec4<f32>(0.0);
                var vb = vec4<f32>(0.0);
                if (it < m) {
                    let base = it * d;
                    let c = col0 + q;
                    if (c < d) { va.x = phi[base + c]; }
                    if (c + 1u < d) { va.y = phi[base + c + 1u]; }
                    if (c + 2u < d) { va.z = phi[base + c + 2u]; }
                    if (c + 3u < d) { va.w = phi[base + c + 3u]; }
                    let lr = row0 + q;
                    let g = ro + lr;
                    if (lr < rr && g < d) { vb.x = phi[base + g]; }
                    if (lr + 1u < rr && g + 1u < d) { vb.y = phi[base + g + 1u]; }
                    if (lr + 2u < rr && g + 2u < d) { vb.z = phi[base + g + 2u]; }
                    if (lr + 3u < rr && g + 3u < d) { vb.w = phi[base + g + 3u]; }
                }
                sa[i] = va;
                sb[i] = vb;
            }
            workgroupBarrier();
            var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
            var c4 = vec4<f32>(0.0); var c5 = vec4<f32>(0.0); var c6 = vec4<f32>(0.0); var c7 = vec4<f32>(0.0);
            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                let ia = k * 32u + lid.x * 2u;
                let a0 = sa[ia];
                let a1 = sa[ia + 1u];
                let a2 = sa[ia + 16u];
                let a3 = sa[ia + 17u];
                let p = sb[k * 32u + lid.y];
                let s = sb[k * 32u + 16u + lid.y];
                c0 = c0 + (p.x * a0 + s.x * a2);
                c1 = c1 + (p.x * a1 + s.x * a3);
                c2 = c2 + (p.y * a0 + s.y * a2);
                c3 = c3 + (p.y * a1 + s.y * a3);
                c4 = c4 + (p.z * a0 + s.z * a2);
                c5 = c5 + (p.z * a1 + s.z * a3);
                c6 = c6 + (p.w * a0 + s.w * a2);
                c7 = c7 + (p.w * a1 + s.w * a3);
            }
            b0 = b0 + c0; b1 = b1 + c1; b2 = b2 + c2; b3 = b3 + c3;
            b4 = b4 + c4; b5 = b5 + c5; b6 = b6 + c6; b7 = b7 + c7;
            workgroupBarrier();
        }
        t0 = t0 + b0; t1 = t1 + b1; t2 = t2 + b2; t3 = t3 + b3;
        t4 = t4 + b4; t5 = t5 + b5; t6 = t6 + b6; t7 = t7 + b7;
    }
    let lr0 = row0 + lid.y * 4u;
    let cb = col0 + lid.x * 8u;
    store_row(lr0, cb, t0, t1);
    store_row(lr0 + 1u, cb, t2, t3);
    store_row(lr0 + 2u, cb, t4, t5);
    store_row(lr0 + 3u, cb, t6, t7);
}
