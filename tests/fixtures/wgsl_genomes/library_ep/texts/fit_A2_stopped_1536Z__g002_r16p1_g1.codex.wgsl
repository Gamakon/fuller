// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };

@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;

@compute @workgroup_size(8, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    var acc: array<f32, 64> = array<f32, 64>();
    var k: u32 = 0u;
    var q: u32 = 0u;
    var t: u32 = 0u;
    var tt: u32 = 0u;

    for (k = 0u; k < 32u; k = k + 1u) {
        acc[k] = 0.0;
    }

    for (t = 0u; t < cfg.m; t = t + 32u) {
        for (q = 0u; q < 16u; q = q + 1u) {
            let i = lidx + q * 128u;
            let item = t + i / 64u;
            let f = i & 63u;
            let c = wid.x * 64u + f;
            let lr = wid.y * 64u + f;
            var av: f32 = 0.0;
            var bv: f32 = 0.0;

            if (item < cfg.m) {
                if (c < cfg.d) {
                    av = phi[item * cfg.d + c];
                }
                if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
                    bv = phi[item * cfg.d + cfg.r_off + lr];
                }
            }

            wg_a[i] = av;
            wg_b[i] = bv;
        }

        workgroupBarrier();

        for (tt = 0u; tt < 32u; tt = tt + 1u) {
            let row_base = tt * 64u + lid.y * 4u;
            let col_base = tt * 64u + lid.x * 8u;

            for (k = 0u; k < 4u; k = k + 1u) {
                let rv = wg_b[row_base + k];
                acc[k * 8u + 0u] = acc[k * 8u + 0u] + rv * wg_a[col_base + 0u];
                acc[k * 8u + 1u] = acc[k * 8u + 1u] + rv * wg_a[col_base + 1u];
                acc[k * 8u + 2u] = acc[k * 8u + 2u] + rv * wg_a[col_base + 2u];
                acc[k * 8u + 3u] = acc[k * 8u + 3u] + rv * wg_a[col_base + 3u];
                acc[k * 8u + 4u] = acc[k * 8u + 4u] + rv * wg_a[col_base + 4u];
                acc[k * 8u + 5u] = acc[k * 8u + 5u] + rv * wg_a[col_base + 5u];
                acc[k * 8u + 6u] = acc[k * 8u + 6u] + rv * wg_a[col_base + 6u];
                acc[k * 8u + 7u] = acc[k * 8u + 7u] + rv * wg_a[col_base + 7u];
            }
        }

        workgroupBarrier();
    }

    for (k = 0u; k < 4u; k = k + 1u) {
        let lr = wid.y * 64u + lid.y * 4u + k;
        if (lr < cfg.r_rows && cfg.r_off + lr < cfg.d) {
            for (q = 0u; q < 8u; q = q + 1u) {
                let c = wid.x * 64u + lid.x * 8u + q;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[k * 8u + q];
                }
            }
        }
    }
}
