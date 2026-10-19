// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 128>;
var<workgroup> sb: array<vec4<f32>, 128>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let isA = l < 128u;
    let slot = l & 127u;
    let lk = slot >> 4u;
    let lv = slot & 15u;
    let rowBase = wid.y * 64u;
    let colBase = wid.x * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        acc[q & 3u] = vec4<f32>(0.0);
    }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 8u) {
        let t = t0 + lk;
        var v = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            if (isA) {
                let lr0 = rowBase + lv * 4u;
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    let lr = lr0 + q;
                    let gr = cfg.r_off + lr;
                    if (lr < cfg.r_rows && gr < d) {
                        v[q] = phi[base + gr];
                    }
                }
            } else {
                let c0 = colBase + lv * 4u;
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    let c = c0 + q;
                    if (c < d) {
                        v[q] = phi[base + c];
                    }
                }
            }
        }
        if (isA) {
            sa[slot & 127u] = v;
        } else {
            sb[slot & 127u] = v;
        }
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let a = sa[(k * 16u + ty) & 127u];
            let b = sb[(k * 16u + tx) & 127u];
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = colBase + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + ty * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            let a = acc[i & 3u];
            let o = lr * d + c0;
            if (c0 + 0u < d) { h[o] = h[o] + a.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + a.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + a.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + a.w; }
        }
    }
}
