// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_c: array<vec4<f32>, 512>;
var<workgroup> wg_r: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    let d = cfg.d;
    let m = cfg.m;
    let rr = cfg.r_rows;
    let roff = cfg.r_off;
    let col_base = wid.x * 64u;
    let row_base = wid.y * 64u;
    var t0: vec4<f32> = vec4<f32>(0.0);
    var t1: vec4<f32> = vec4<f32>(0.0);
    var t2: vec4<f32> = vec4<f32>(0.0);
    var t3: vec4<f32> = vec4<f32>(0.0);
    for (var t: u32 = 0u; t < m; t = t + 32u) {
        for (var j: u32 = 0u; j < 2u; j = j + 1u) {
            let i = lidx + 256u * j;
            let it = t + (i >> 4u);
            let cc = (i & 15u) * 4u;
            var vc: vec4<f32> = vec4<f32>(0.0);
            var vr: vec4<f32> = vec4<f32>(0.0);
            if (it < m) {
                let base = it * d;
                let c = col_base + cc;
                if (c < d) { vc.x = phi[base + c]; }
                if (c + 1u < d) { vc.y = phi[base + c + 1u]; }
                if (c + 2u < d) { vc.z = phi[base + c + 2u]; }
                if (c + 3u < d) { vc.w = phi[base + c + 3u]; }
                let lr = row_base + cc;
                let gr = roff + lr;
                if (lr < rr && gr < d) { vr.x = phi[base + gr]; }
                if (lr + 1u < rr && gr + 1u < d) { vr.y = phi[base + gr + 1u]; }
                if (lr + 2u < rr && gr + 2u < d) { vr.z = phi[base + gr + 2u]; }
                if (lr + 3u < rr && gr + 3u < d) { vr.w = phi[base + gr + 3u]; }
            }
            wg_c[i] = vc;
            wg_r[i] = vr;
        }
        workgroupBarrier();
        var a0: vec4<f32> = vec4<f32>(0.0);
        var a1: vec4<f32> = vec4<f32>(0.0);
        var a2: vec4<f32> = vec4<f32>(0.0);
        var a3: vec4<f32> = vec4<f32>(0.0);
        var b0: vec4<f32> = vec4<f32>(0.0);
        var b1: vec4<f32> = vec4<f32>(0.0);
        var b2: vec4<f32> = vec4<f32>(0.0);
        var b3: vec4<f32> = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < 32u; kk = kk + 2u) {
            let o0 = kk * 16u;
            let o1 = o0 + 16u;
            let c0 = wg_c[o0 + lid.x];
            let r0 = wg_r[o0 + lid.y];
            let c1 = wg_c[o1 + lid.x];
            let r1 = wg_r[o1 + lid.y];
            a0 = a0 + r0.x * c0;
            a1 = a1 + r0.y * c0;
            a2 = a2 + r0.z * c0;
            a3 = a3 + r0.w * c0;
            b0 = b0 + r1.x * c1;
            b1 = b1 + r1.y * c1;
            b2 = b2 + r1.z * c1;
            b3 = b3 + r1.w * c1;
        }
        t0 = t0 + (a0 + b0);
        t1 = t1 + (a1 + b1);
        t2 = t2 + (a2 + b2);
        t3 = t3 + (a3 + b3);
        workgroupBarrier();
    }
    let c_out = col_base + lid.x * 4u;
    let r_out = row_base + lid.y * 4u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = r_out + j;
        if (lr < rr && roff + lr < d) {
            var acc: vec4<f32> = t0;
            if (j == 1u) { acc = t1; }
            if (j == 2u) { acc = t2; }
            if (j == 3u) { acc = t3; }
            let rowb = lr * d;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = c_out + i;
                if (c < d) {
                    let idx = rowb + c;
                    h[idx] = h[idx] + acc[i];
                }
            }
        }
    }
}
