// grid 16 8 4 1
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };

@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 32>;
var<workgroup> wg_b: array<f32, 256>;

@compute @workgroup_size(16, 8, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lr = wid.y * 8u + lid.y;
    let gr = cfg.r_off + lr;
    let c0 = wid.x * 64u + lid.x * 4u;

    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);

    for (var t: u32 = 0u; t < cfg.m; t = t + 4u) {
        if (lidx < 32u) {
            let q = lidx / 8u;
            let rr = lidx % 8u;
            let local_row = wid.y * 8u + rr;
            let global_row = cfg.r_off + local_row;
            if (local_row < cfg.r_rows && global_row < cfg.d && t + q < cfg.m) {
                wg_a[lidx] = phi[(t + q) * cfg.d + global_row];
            } else {
                wg_a[lidx] = 0.0;
            }
        }

        if (lidx < 64u) {
            let c = wid.x * 64u + lidx;
            if (c < cfg.d && t < cfg.m) {
                wg_b[lidx] = phi[t * cfg.d + c];
            } else {
                wg_b[lidx] = 0.0;
            }
            if (c < cfg.d && t + 1u < cfg.m) {
                wg_b[64u + lidx] = phi[(t + 1u) * cfg.d + c];
            } else {
                wg_b[64u + lidx] = 0.0;
            }
            if (c < cfg.d && t + 2u < cfg.m) {
                wg_b[128u + lidx] = phi[(t + 2u) * cfg.d + c];
            } else {
                wg_b[128u + lidx] = 0.0;
            }
            if (c < cfg.d && t + 3u < cfg.m) {
                wg_b[192u + lidx] = phi[(t + 3u) * cfg.d + c];
            } else {
                wg_b[192u + lidx] = 0.0;
            }
        }

        workgroupBarrier();

        let p = lid.x * 4u;
        let b0 = vec4<f32>(wg_b[p], wg_b[p + 1u], wg_b[p + 2u], wg_b[p + 3u]);
        let b1 = vec4<f32>(wg_b[64u + p], wg_b[64u + p + 1u], wg_b[64u + p + 2u], wg_b[64u + p + 3u]);
        let b2 = vec4<f32>(wg_b[128u + p], wg_b[128u + p + 1u], wg_b[128u + p + 2u], wg_b[128u + p + 3u]);
        let b3 = vec4<f32>(wg_b[192u + p], wg_b[192u + p + 1u], wg_b[192u + p + 2u], wg_b[192u + p + 3u]);

        v0 = v0 + wg_a[lid.y] * b0;
        v1 = v1 + wg_a[8u + lid.y] * b1;
        v2 = v2 + wg_a[16u + lid.y] * b2;
        v3 = v3 + wg_a[24u + lid.y] * b3;

        workgroupBarrier();
    }

    if (lr < cfg.r_rows && gr < cfg.d) {
        let sum = (v0 + v1) + (v2 + v3);
        if (c0 < cfg.d) {
            let i = lr * cfg.d + c0;
            h[i] = h[i] + sum.x;
        }
        if (c0 + 1u < cfg.d) {
            let i = lr * cfg.d + c0 + 1u;
            h[i] = h[i] + sum.y;
        }
        if (c0 + 2u < cfg.d) {
            let i = lr * cfg.d + c0 + 2u;
            h[i] = h[i] + sum.z;
        }
        if (c0 + 3u < cfg.d) {
            let i = lr * cfg.d + c0 + 3u;
            h[i] = h[i] + sum.w;
        }
    }
}
