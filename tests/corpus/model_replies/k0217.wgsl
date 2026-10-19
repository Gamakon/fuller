// grid 16 8 8 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 64>;
    for (var q: u32 = 0u; q < 64u; q = q + 1u) { acc[q & 63u] = 0.0; }
    var ra: array<f32, 8>;
    var rb: array<f32, 8>;
    let d = cfg.d;
    let m = cfg.m;
    let col0 = wid.x * 128u;
    let row0 = wid.y * 64u;
    for (var t: u32 = 0u; t < m; t = t + 16u) {
        // load A tile: 16 items x 128 columns
        for (var q: u32 = 0u; q < 16u; q = q + 1u) {
            let e = lidx + 128u * q;
            let it = e >> 7u;
            let cc = e & 127u;
            let gc = col0 + cc;
            var v: f32 = 0.0;
            if ((t + it) < m && gc < d) {
                v = phi[(t + it) * d + gc];
            }
            wg_a[e & 2047u] = v;
        }
        // load B tile: 16 items x 64 rows
        for (var q: u32 = 0u; q < 8u; q = q + 1u) {
            let e = lidx + 128u * q;
            let it = e >> 6u;
            let rr = e & 63u;
            let lr = row0 + rr;
            var v: f32 = 0.0;
            if ((t + it) < m && lr < cfg.r_rows && (cfg.r_off + lr) < d) {
                v = phi[(t + it) * d + cfg.r_off + lr];
            }
            wg_b[e & 1023u] = v;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                ra[j & 7u] = wg_a[(k * 128u + j * 16u + lid.x) & 2047u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                rb[i & 7u] = wg_b[(k * 64u + i * 8u + lid.y) & 1023u];
            }
            for (var i: u32 = 0u; i < 8u; i = i + 1u) {
                let bv = rb[i & 7u];
                for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                    let ai = (i * 8u + j) & 63u;
                    acc[ai] = acc[ai] + bv * ra[j & 7u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let lr = row0 + i * 8u + lid.y;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 8u; j = j + 1u) {
                let c = col0 + j * 16u + lid.x;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 8u + j) & 63u];
                }
            }
        }
    }
}
