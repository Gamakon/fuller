// grid 16 8 4 8
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 512>;
var<workgroup> wg_b: array<f32, 512>;

@compute @workgroup_size(16, 8, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let c0 = wid.x * 64u + lid.x * 4u;
    let lr0 = wid.y * 64u + lid.y * 8u;
    var acc: array<f32, 64>;

    for (var rr: u32 = 0u; rr < 8u; rr = rr + 1u) {
        for (var cc: u32 = 0u; cc < 4u; cc = cc + 1u) {
            acc[rr * 4u + cc] = 0.0;
        }
    }

    for (var t0: u32 = 0u; t0 < cfg.m; t0 = t0 + 8u) {
        for (var load: u32 = 0u; load < 8u; load = load + 1u) {
            let q = lidx + load * 128u;
            if (q < 512u) {
                let it = t0 + q / 64u;
                let gr = cfg.r_off + wid.y * 64u + q % 64u;
                if (it < cfg.m && gr < cfg.d) {
                    wg_a[q] = phi[it * cfg.d + gr];
                } else {
                    wg_a[q] = 0.0;
                }
            } else {
                let p = q - 512u;
                let it = t0 + p / 64u;
                let c = wid.x * 64u + p % 64u;
                if (it < cfg.m && c < cfg.d) {
                    wg_b[p] = phi[it * cfg.d + c];
                } else {
                    wg_b[p] = 0.0;
                }
            }
        }
        workgroupBarrier();

        for (var it: u32 = 0u; it < 8u; it = it + 1u) {
            for (var rr: u32 = 0u; rr < 8u; rr = rr + 1u) {
                let a = wg_a[it * 64u + lid.y * 8u + rr];
                for (var cc: u32 = 0u; cc < 4u; cc = cc + 1u) {
                    acc[rr * 4u + cc] = acc[rr * 4u + cc] + a * wg_b[it * 64u + lid.x * 4u + cc];
                }
            }
        }
        workgroupBarrier();
    }

    for (var rr: u32 = 0u; rr < 8u; rr = rr + 1u) {
        let lr = lr0 + rr;
        let gr = cfg.r_off + lr;
        for (var cc: u32 = 0u; cc < 4u; cc = cc + 1u) {
            let c = c0 + cc;
            if (lr < cfg.r_rows && gr < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + acc[rr * 4u + cc];
            }
        }
    }
}
