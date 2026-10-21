// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wa: array<vec4<f32>, 256>;
var<workgroup> wb: array<vec4<f32>, 256>;

fn store_row(lr: u32, c0: u32, v: vec4<f32>) {
    if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
        let base = lr * cfg.d + c0;
        if (c0 < cfg.d) { h[base] = h[base] + v.x; }
        if (c0 + 1u < cfg.d) { h[base + 1u] = h[base + 1u] + v.y; }
        if (c0 + 2u < cfg.d) { h[base + 2u] = h[base + 2u] + v.z; }
        if (c0 + 3u < cfg.d) { h[base + 3u] = h[base + 3u] + v.w; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let ro = cfg.r_off;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    let lk = lidx >> 4u;
    let lj = (lidx & 15u) * 4u;
    let lc = col0 + lj;
    let lrw = row0 + lj;
    let lg = ro + lrw;
    var s0 = vec4<f32>(0.0); var s1 = vec4<f32>(0.0); var s2 = vec4<f32>(0.0); var s3 = vec4<f32>(0.0);
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 1024u) {
        var r0 = vec4<f32>(0.0); var r1 = vec4<f32>(0.0); var r2 = vec4<f32>(0.0); var r3 = vec4<f32>(0.0);
        for (var b: u32 = 0u; b < 16u && (t0 + b * 64u) < m; b = b + 1u) {
            var q0 = vec4<f32>(0.0); var q1 = vec4<f32>(0.0); var q2 = vec4<f32>(0.0); var q3 = vec4<f32>(0.0);
            for (var u: u32 = 0u; u < 4u; u = u + 1u) {
                let t = t0 + b * 64u + u * 16u + lk;
                var va = vec4<f32>(0.0);
                var vb = vec4<f32>(0.0);
                if (t < m) {
                    let base = t * d;
                    if (lc + 3u < d) {
                        va = vec4<f32>(phi[base + lc], phi[base + lc + 1u], phi[base + lc + 2u], phi[base + lc + 3u]);
                    } else {
                        if (lc < d) { va.x = phi[base + lc]; }
                        if (lc + 1u < d) { va.y = phi[base + lc + 1u]; }
                        if (lc + 2u < d) { va.z = phi[base + lc + 2u]; }
                    }
                    if (lrw + 3u < rr && lg + 3u < d) {
                        vb = vec4<f32>(phi[base + lg], phi[base + lg + 1u], phi[base + lg + 2u], phi[base + lg + 3u]);
                    } else {
                        if (lrw < rr && lg < d) { vb.x = phi[base + lg]; }
                        if (lrw + 1u < rr && lg + 1u < d) { vb.y = phi[base + lg + 1u]; }
                        if (lrw + 2u < rr && lg + 2u < d) { vb.z = phi[base + lg + 2u]; }
                        if (lrw + 3u < rr && lg + 3u < d) { vb.w = phi[base + lg + 3u]; }
                    }
                }
                wa[lidx] = va;
                wb[lidx] = vb;
                workgroupBarrier();
                var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0); var p3 = vec4<f32>(0.0);
                for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                    let a = wa[k * 16u + lid.x];
                    let bv = wb[k * 16u + lid.y];
                    p0 = fma(vec4<f32>(bv.x), a, p0);
                    p1 = fma(vec4<f32>(bv.y), a, p1);
                    p2 = fma(vec4<f32>(bv.z), a, p2);
                    p3 = fma(vec4<f32>(bv.w), a, p3);
                }
                q0 = q0 + p0; q1 = q1 + p1; q2 = q2 + p2; q3 = q3 + p3;
                workgroupBarrier();
            }
            r0 = r0 + q0; r1 = r1 + q1; r2 = r2 + q2; r3 = r3 + q3;
        }
        s0 = s0 + r0; s1 = s1 + r1; s2 = s2 + r2; s3 = s3 + r3;
    }
    let lr0 = row0 + lid.y * 4u;
    let c0 = col0 + lid.x * 4u;
    store_row(lr0, c0, s0);
    store_row(lr0 + 1u, c0, s1);
    store_row(lr0 + 2u, c0, s2);
    store_row(lr0 + 3u, c0, s3);
}
