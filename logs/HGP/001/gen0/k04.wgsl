// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    // loader roles: k = ly (item within tile), q = lx (vec4 slot)
    let lc = col0 + lx * 4u;
    let lrow = row0 + lx * 4u;

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + ly;
        var bv = vec4<f32>(0.0);
        var av = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            if (lc + 3u < d) {
                bv = vec4<f32>(phi[base + lc], phi[base + lc + 1u], phi[base + lc + 2u], phi[base + lc + 3u]);
            } else {
                if (lc < d) { bv.x = phi[base + lc]; }
                if (lc + 1u < d) { bv.y = phi[base + lc + 1u]; }
                if (lc + 2u < d) { bv.z = phi[base + lc + 2u]; }
            }
            let g0 = cfg.r_off + lrow;
            if (lrow < cfg.r_rows && g0 < d) { av.x = phi[base + g0]; }
            if (lrow + 1u < cfg.r_rows && g0 + 1u < d) { av.y = phi[base + g0 + 1u]; }
            if (lrow + 2u < cfg.r_rows && g0 + 2u < d) { av.z = phi[base + g0 + 2u]; }
            if (lrow + 3u < cfg.r_rows && g0 + 3u < d) { av.w = phi[base + g0 + 3u]; }
        }
        sb[ly * 16u + lx] = bv;
        sa[ly * 16u + lx] = av;
        workgroupBarrier();

        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let b = sb[k * 16u + lx];
            let a = sa[k * 16u + ly];
            acc0 = acc0 + a.x * b;
            acc1 = acc1 + a.y * b;
            acc2 = acc2 + a.z * b;
            acc3 = acc3 + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = col0 + lx * 4u;
    let r0 = row0 + ly * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = r0 + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            var a = acc0;
            if (i == 1u) { a = acc1; }
            if (i == 2u) { a = acc2; }
            if (i == 3u) { a = acc3; }
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + a.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + a.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + a.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + a.w; }
        }
    }
}
