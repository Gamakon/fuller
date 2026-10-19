// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<vec4<f32>, 512>;
var<workgroup> wg_b: array<vec4<f32>, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let col0 = wid.x * 64u + lid.x * 4u;
    let row0 = wid.y * 64u + lid.y * 4u;

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);

    for (var t: u32 = 0u; t < cfg.m; t = t + 128u) {
        for (var tile: u32 = 0u; tile < 4u; tile = tile + 1u) {
            let tile_t = t + tile * 32u;

            for (var q: u32 = 0u; q < 2u; q = q + 1u) {
                let p = lidx + q * 256u;
                let item = tile_t + (p >> 4u);
                let feature = (p & 15u) * 4u;

                var a0: f32 = 0.0;
                var a1: f32 = 0.0;
                var a2: f32 = 0.0;
                var a3: f32 = 0.0;
                var b0: f32 = 0.0;
                var b1: f32 = 0.0;
                var b2: f32 = 0.0;
                var b3: f32 = 0.0;

                if (item < cfg.m) {
                    let abase = item * cfg.d + wid.x * 64u + feature;
                    if (wid.x * 64u + feature < cfg.d) {
                        a0 = phi[abase];
                    }
                    if (wid.x * 64u + feature + 1u < cfg.d) {
                        a1 = phi[abase + 1u];
                    }
                    if (wid.x * 64u + feature + 2u < cfg.d) {
                        a2 = phi[abase + 2u];
                    }
                    if (wid.x * 64u + feature + 3u < cfg.d) {
                        a3 = phi[abase + 3u];
                    }

                    let brow = wid.y * 64u + feature;
                    let bbase = item * cfg.d + cfg.r_off + brow;
                    if (brow < cfg.r_rows && cfg.r_off + brow < cfg.d) {
                        b0 = phi[bbase];
                    }
                    if (brow + 1u < cfg.r_rows && cfg.r_off + brow + 1u < cfg.d) {
                        b1 = phi[bbase + 1u];
                    }
                    if (brow + 2u < cfg.r_rows && cfg.r_off + brow + 2u < cfg.d) {
                        b2 = phi[bbase + 2u];
                    }
                    if (brow + 3u < cfg.r_rows && cfg.r_off + brow + 3u < cfg.d) {
                        b3 = phi[bbase + 3u];
                    }
                }

                wg_a[p] = vec4<f32>(a0, a1, a2, a3);
                wg_b[p] = vec4<f32>(b0, b1, b2, b3);
            }

            workgroupBarrier();

            var acc0 = vec4<f32>(0.0);
            var acc1 = vec4<f32>(0.0);
            var acc2 = vec4<f32>(0.0);
            var acc3 = vec4<f32>(0.0);

            for (var k: u32 = 0u; k < 16u; k = k + 1u) {
                let ax0 = wg_a[k * 16u + lid.x];
                let ax1 = wg_a[(k + 16u) * 16u + lid.x];
                let by0 = wg_b[k * 16u + lid.y];
                let by1 = wg_b[(k + 16u) * 16u + lid.y];

                acc0 = acc0 + ax0 * by0.x + ax1 * by1.x;
                acc1 = acc1 + ax0 * by0.y + ax1 * by1.y;
                acc2 = acc2 + ax0 * by0.z + ax1 * by1.z;
                acc3 = acc3 + ax0 * by0.w + ax1 * by1.w;
            }

            sum0 = sum0 + acc0;
            sum1 = sum1 + acc1;
            sum2 = sum2 + acc2;
            sum3 = sum3 + acc3;

            workgroupBarrier();
        }
    }

    if (row0 < cfg.r_rows && cfg.r_off + row0 < cfg.d) {
        if (col0 < cfg.d) {
            let i = row0 * cfg.d + col0;
            h[i] = h[i] + sum0.x;
        }
        if (col0 + 1u < cfg.d) {
            let i = row0 * cfg.d + col0 + 1u;
            h[i] = h[i] + sum0.y;
        }
        if (col0 + 2u < cfg.d) {
            let i = row0 * cfg.d + col0 + 2u;
            h[i] = h[i] + sum0.z;
        }
        if (col0 + 3u < cfg.d) {
            let i = row0 * cfg.d + col0 + 3u;
            h[i] = h[i] + sum0.w;
        }
    }
    if (row0 + 1u < cfg.r_rows && cfg.r_off + row0 + 1u < cfg.d) {
        if (col0 < cfg.d) {
            let i = (row0 + 1u) * cfg.d + col0;
            h[i] = h[i] + sum1.x;
        }
        if (col0 + 1u < cfg.d) {
            let i = (row0 + 1u) * cfg.d + col0 + 1u;
            h[i] = h[i] + sum1.y;
        }
        if (col0 + 2u < cfg.d) {
            let i = (row0 + 1u) * cfg.d + col0 + 2u;
            h[i] = h[i] + sum1.z;
        }
        if (col0 + 3u < cfg.d) {
            let i = (row0 + 1u) * cfg.d + col0 + 3u;
            h[i] = h[i] + sum1.w;
        }
    }
    if (row0 + 2u < cfg.r_rows && cfg.r_off + row0 + 2u < cfg.d) {
        if (col0 < cfg.d) {
            let i = (row0 + 2u) * cfg.d + col0;
            h[i] = h[i] + sum2.x;
        }
        if (col0 + 1u < cfg.d) {
            let i = (row0 + 2u) * cfg.d + col0 + 1u;
            h[i] = h[i] + sum2.y;
        }
        if (col0 + 2u < cfg.d) {
            let i = (row0 + 2u) * cfg.d + col0 + 2u;
            h[i] = h[i] + sum2.z;
        }
        if (col0 + 3u < cfg.d) {
            let i = (row0 + 2u) * cfg.d + col0 + 3u;
            h[i] = h[i] + sum2.w;
        }
    }
    if (row0 + 3u < cfg.r_rows && cfg.r_off + row0 + 3u < cfg.d) {
        if (col0 < cfg.d) {
            let i = (row0 + 3u) * cfg.d + col0;
            h[i] = h[i] + sum3.x;
        }
        if (col0 + 1u < cfg.d) {
            let i = (row0 + 3u) * cfg.d + col0 + 1u;
            h[i] = h[i] + sum3.y;
        }
        if (col0 + 2u < cfg.d) {
            let i = (row0 + 3u) * cfg.d + col0 + 2u;
            h[i] = h[i] + sum3.z;
        }
        if (col0 + 3u < cfg.d) {
            let i = (row0 + 3u) * cfg.d + col0 + 3u;
            h[i] = h[i] + sum3.w;
        }
    }
}
