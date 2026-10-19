// grid 8 8 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 8u + lid.x;
    let colBase = wg.x * 32u;
    let rowBase = wg.y * 32u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i] = vec4<f32>(0.0);
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i = 0u; i < 8u; i = i + 1u) {
            let idx = tid + i * 64u;
            let k = idx / 32u;
            let cc = idx % 32u;
            let t = t0 + k;
            let colg = colBase + cc;
            let lrow = rowBase + cc;
            let grow = cfg.r_off + lrow;
            var bv = 0.0;
            var av = 0.0;
            if (t < m) {
                if (colg < d) {
                    bv = phi[t * d + colg];
                }
                if (lrow < cfg.r_rows && grow < d) {
                    av = phi[t * d + grow];
                }
            }
            Bs[idx] = bv;
            As[idx] = av;
        }
        workgroupBarrier();

        for (var k = 0u; k < 16u; k = k + 1u) {
            let bo = k * 32u + lid.x * 4u;
            let ao = k * 32u + lid.y * 4u;
            let b = vec4<f32>(Bs[bo], Bs[bo + 1u], Bs[bo + 2u], Bs[bo + 3u]);
            let a = vec4<f32>(As[ao], As[ao + 1u], As[ao + 2u], As[ao + 3u]);
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + lid.x * 4u;
    let r0 = rowBase + lid.y * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let v = acc[i];
            let base = lr * d + c0;
            if (c0 < d) { h[base] = h[base] + v.x; }
            if (c0 + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (c0 + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (c0 + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}