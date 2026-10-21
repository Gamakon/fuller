// grid 8 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };

@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_cols: array<f32, 2048>;
var<workgroup> wg_rows: array<f32, 2048>;

@compute @workgroup_size(8, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col0 = wid.x * 64u + lid.x * 8u;
    let row0 = wid.y * 64u + lid.y * 4u;
    let gr0 = cfg.r_off + row0;

    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0);
    var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0);
    var a7 = vec4<f32>(0.0);

    for (var tb: u32 = 0u; tb < cfg.m; tb = tb + 32u) {
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let q = lidx + j * 128u;
            let it = q / 64u;
            let f = q - it * 64u;
            let t = tb + it;

            var xc: f32 = 0.0;
            var xr: f32 = 0.0;
            if (t < cfg.m) {
                let c = wid.x * 64u + f;
                if (c < cfg.d) {
                    xc = phi[t * cfg.d + c];
                }
                let lr = wid.y * 64u + f;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < cfg.d) {
                    xr = phi[t * cfg.d + gr];
                }
            }
            wg_cols[q] = xc;
            wg_rows[q] = xr;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < 32u; k = k + 1u) {
            let b = k * 64u;
            let rv = vec4<f32>(
                wg_rows[b + lid.y * 4u],
                wg_rows[b + lid.y * 4u + 1u],
                wg_rows[b + lid.y * 4u + 2u],
                wg_rows[b + lid.y * 4u + 3u]
            );
            a0 = a0 + rv * wg_cols[b + lid.x * 8u];
            a1 = a1 + rv * wg_cols[b + lid.x * 8u + 1u];
            a2 = a2 + rv * wg_cols[b + lid.x * 8u + 2u];
            a3 = a3 + rv * wg_cols[b + lid.x * 8u + 3u];
            a4 = a4 + rv * wg_cols[b + lid.x * 8u + 4u];
            a5 = a5 + rv * wg_cols[b + lid.x * 8u + 5u];
            a6 = a6 + rv * wg_cols[b + lid.x * 8u + 6u];
            a7 = a7 + rv * wg_cols[b + lid.x * 8u + 7u];
        }

        workgroupBarrier();
    }

    for (var ry: u32 = 0u; ry < 4u; ry = ry + 1u) {
        let lr = row0 + ry;
        let gr = gr0 + ry;
        if (lr < cfg.r_rows && gr < cfg.d) {
            if (col0 < cfg.d) {
                let p = lr * cfg.d + col0;
                h[p] = h[p] + a0[ry];
            }
            if (col0 + 1u < cfg.d) {
                let p = lr * cfg.d + col0 + 1u;
                h[p] = h[p] + a1[ry];
            }
            if (col0 + 2u < cfg.d) {
                let p = lr * cfg.d + col0 + 2u;
                h[p] = h[p] + a2[ry];
            }
            if (col0 + 3u < cfg.d) {
                let p = lr * cfg.d + col0 + 3u;
                h[p] = h[p] + a3[ry];
            }
            if (col0 + 4u < cfg.d) {
                let p = lr * cfg.d + col0 + 4u;
                h[p] = h[p] + a4[ry];
            }
            if (col0 + 5u < cfg.d) {
                let p = lr * cfg.d + col0 + 5u;
                h[p] = h[p] + a5[ry];
            }
            if (col0 + 6u < cfg.d) {
                let p = lr * cfg.d + col0 + 6u;
                h[p] = h[p] + a6[ry];
            }
            if (col0 + 7u < cfg.d) {
                let p = lr * cfg.d + col0 + 7u;
                h[p] = h[p] + a7[ry];
            }
        }
    }
}
