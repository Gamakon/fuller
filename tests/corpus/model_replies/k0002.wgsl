// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<vec4<f32>, 256>;
var<workgroup> sb: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;
    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let t = e >> 6u;
            let j = e & 63u;
            let gi = t0 + t;
            var va = 0.0;
            var vb = 0.0;
            if (gi < m) {
                let lr = row0 + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[gi * d + gr];
                }
                let c = col0 + j;
                if (c < d) {
                    vb = phi[gi * d + c];
                }
            }
            let vi = (e >> 2u) & 255u;
            sa[vi][j & 3u] = va;
            sb[vi][j & 3u] = vb;
        }
        workgroupBarrier();
        for (var t = 0u; t < 16u; t = t + 1u) {
            let a = sa[(t * 16u + ty) & 255u];
            let b = sb[(t * 16u + tx) & 255u];
            acc[0] = acc[0] + a.x * b;
            acc[1] = acc[1] + a.y * b;
            acc[2] = acc[2] + a.z * b;
            acc[3] = acc[3] + a.w * b;
        }
        workgroupBarrier();
    }
    let cb = col0 + tx * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            let base = lr * d + cb;
            if (cb < d) { h[base] = h[base] + v.x; }
            if (cb + 1u < d) { h[base + 1u] = h[base + 1u] + v.y; }
            if (cb + 2u < d) { h[base + 2u] = h[base + 2u] + v.z; }
            if (cb + 3u < d) { h[base + 3u] = h[base + 3u] + v.w; }
        }
    }
}
