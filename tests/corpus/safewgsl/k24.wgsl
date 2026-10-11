// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 512>;
var<workgroup> sb: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let rowBase = wg.y * 64u;
    let colBase = wg.x * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    var t0: u32 = 0u;
    loop {
        if (t0 >= m) { break; }
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let e = tid + k * 256u;
            let tt = e >> 6u;
            let cc = e & 63u;
            let t = t0 + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lrow = rowBase + cc;
                let grow = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && grow < d) {
                    va = phi[t * d + grow];
                }
                let col = colBase + cc;
                if (col < d) {
                    vb = phi[t * d + col];
                }
            }
            sa[e] = va;
            sb[e] = vb;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 8u; tt = tt + 1u) {
            let ab = tt * 64u + lid.y * 4u;
            let bb = tt * 64u + lid.x * 4u;
            let a = vec4<f32>(sa[ab], sa[ab + 1u], sa[ab + 2u], sa[ab + 3u]);
            let b = vec4<f32>(sb[bb], sb[bb + 1u], sb[bb + 2u], sb[bb + 3u]);
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
        t0 = t0 + 8u;
    }

    let c0 = colBase + lid.x * 4u;
    let r0 = rowBase + lid.y * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows) {
            var v = acc0;
            if (i == 1u) { v = acc1; }
            if (i == 2u) { v = acc2; }
            if (i == 3u) { v = acc3; }
            let base = lr * d;
            if (c0 < d) { h[base + c0] = h[base + c0] + v.x; }
            if (c0 + 1u < d) { h[base + c0 + 1u] = h[base + c0 + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + c0 + 2u] = h[base + c0 + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + c0 + 3u] = h[base + c0 + 3u] + v.w; }
        }
    }
}
