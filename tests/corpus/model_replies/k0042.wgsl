// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let l = ty * 16u + tx;
    let ld_item = l >> 4u;
    let ld_c = (l & 15u) * 4u;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i & 3u] = vec4<f32>(0.0);
    }

    for (var t0 = 0u; t0 < m; t0 = t0 + 4u * 4u) {
        let t = t0 + ld_item;
        let tin = t < m;
        let base = t * d;
        for (var q = 0u; q < 4u; q = q + 1u) {
            let cc = col0 + ld_c + q;
            let lrw = row0 + ld_c + q;
            let gr = cfg.r_off + lrw;
            var vb = 0.0;
            var va = 0.0;
            if (tin && cc < d) {
                vb = phi[base + cc];
            }
            if (tin && lrw < cfg.r_rows && gr < d) {
                va = phi[base + gr];
            }
            let si = ld_item * 64u + ld_c + q;
            sb[si & 1023u] = vb;
            sa[si & 1023u] = va;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            let bv = vec4<f32>(
                sb[(o + tx * 4u) & 1023u],
                sb[(o + tx * 4u + 1u) & 1023u],
                sb[(o + tx * 4u + 2u) & 1023u],
                sb[(o + tx * 4u + 3u) & 1023u]);
            for (var i = 0u; i < 4u; i = i + 1u) {
                let a = sa[(o + ty * 4u + i) & 1023u];
                acc[i & 3u] = acc[i & 3u] + a * bv;
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty * 4u + i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            let v = acc[i & 3u];
            let c = col0 + tx * 4u;
            let rb = lr * d;
            if (c < d) { h[rb + c] = h[rb + c] + v.x; }
            if (c + 1u < d) { h[rb + c + 1u] = h[rb + c + 1u] + v.y; }
            if (c + 2u < d) { h[rb + c + 2u] = h[rb + c + 2u] + v.z; }
            if (c + 3u < d) { h[rb + c + 3u] = h[rb + c + 3u] + v.w; }
        }
    }
}
