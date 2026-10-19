// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = li.x;
    let ty = li.y;
    let tid = ty * 16u + tx;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;
    let q = tid & 15u;
    let tl = tid >> 4u;

    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0);
    acc[1] = vec4<f32>(0.0);
    acc[2] = vec4<f32>(0.0);
    acc[3] = vec4<f32>(0.0);

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        let t = t0 + tl;
        var va = vec4<f32>(0.0);
        var vb = vec4<f32>(0.0);
        if (t < m) {
            let base = t * d;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + q * 4u + j;
                if (c < d) {
                    vb[j] = phi[base + c];
                }
                let lr = row0 + q * 4u + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va[j] = phi[base + gr];
                }
            }
        }
        sa[tid & 255u] = va;
        sb[tid & 255u] = vb;
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let a = sa[(k * 16u + ty) & 255u];
            let b = sb[(k * 16u + tx) & 255u];
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }

    let c0 = col0 + tx * 4u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        if (lr < cfg.r_rows) {
            let v = acc[i & 3u];
            let o = lr * d + c0;
            if (c0 < d) { h[o] = h[o] + v.x; }
            if (c0 + 1u < d) { h[o + 1u] = h[o + 1u] + v.y; }
            if (c0 + 2u < d) { h[o + 2u] = h[o + 2u] + v.z; }
            if (c0 + 3u < d) { h[o + 3u] = h[o + 3u] + v.w; }
        }
    }
}
