// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let tid = lid.y * 16u + lid.x;
    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        for (var k: u32 = 0u; k < 2u; k = k + 1u) {
            let idx = tid + k * 256u;
            let item = t + (idx >> 6u);
            let cl = idx & 63u;
            let col = wid.x * 64u + cl;
            let lrow = wid.y * 64u + cl;
            let grow = cfg.r_off + lrow;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (item < cfg.m) {
                if (col < cfg.d) {
                    b = phi[item * cfg.d + col];
                }
                if (lrow < cfg.r_rows && grow < cfg.d) {
                    a = phi[item * cfg.d + grow];
                }
            }
            wg_a[idx & 511u] = a;
            wg_b[idx & 511u] = b;
        }
        workgroupBarrier();
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let av = wg_a[(s * 64u + lid.y * 4u + i) & 511u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = fma(av, wg_b[(s * 64u + lid.x * 4u + j) & 511u], acc[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + i;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lid.x * 4u + j;
                if (c < cfg.d) {
                    let o = lr * cfg.d + c;
                    h[o] = h[o] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
