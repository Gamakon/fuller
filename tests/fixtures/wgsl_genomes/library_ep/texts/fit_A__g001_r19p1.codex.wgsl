// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> wg_a: array<f32, 256>;
var<workgroup> wg_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row0 = wid.y * 32u + lid.y;
    let col0 = wid.x * 32u + lid.x;
    let lane_item = lidx / 32u;
    let lane_feature = lidx % 32u;

    var s00: f32 = 0.0;
    var s01: f32 = 0.0;
    var s10: f32 = 0.0;
    var s11: f32 = 0.0;

    for (var t: u32 = 0u; t < cfg.m; t = t + 8u) {
        let item = t + lane_item;
        var av: f32 = 0.0;
        var bv: f32 = 0.0;
        let load_lr = wid.y * 32u + lane_feature;
        let load_c = wid.x * 32u + lane_feature;

        if (item < cfg.m) {
            if (load_lr < cfg.r_rows && cfg.r_off + load_lr < cfg.d) {
                av = phi[item * cfg.d + cfg.r_off + load_lr];
            }
            if (load_c < cfg.d) {
                bv = phi[item * cfg.d + load_c];
            }
        }

        wg_a[lidx] = av;
        wg_b[lidx] = bv;
        workgroupBarrier();

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let base = k * 32u;
            s00 = s00 + wg_a[base + lid.y] * wg_b[base + lid.x];
            s01 = s01 + wg_a[base + lid.y] * wg_b[base + lid.x + 16u];
            s10 = s10 + wg_a[base + lid.y + 16u] * wg_b[base + lid.x];
            s11 = s11 + wg_a[base + lid.y + 16u] * wg_b[base + lid.x + 16u];
        }
        workgroupBarrier();
    }

    if (row0 < cfg.r_rows && cfg.r_off + row0 < cfg.d) {
        if (col0 < cfg.d) {
            let idx = row0 * cfg.d + col0;
            h[idx] = h[idx] + s00;
        }
        if (col0 + 16u < cfg.d) {
            let idx = row0 * cfg.d + col0 + 16u;
            h[idx] = h[idx] + s01;
        }
    }
    if (row0 + 16u < cfg.r_rows && cfg.r_off + row0 + 16u < cfg.d) {
        if (col0 < cfg.d) {
            let idx = (row0 + 16u) * cfg.d + col0;
            h[idx] = h[idx] + s10;
        }
        if (col0 + 16u < cfg.d) {
            let idx = (row0 + 16u) * cfg.d + col0 + 16u;
            h[idx] = h[idx] + s11;
        }
    }
}
