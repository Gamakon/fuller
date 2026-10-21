// grid 16 16 8 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    let i0 = wid.x * 128u;
    let i1 = wid.y * 64u;
    let i2 = i0 + lid.x;
    let i3 = i1 + lid.y;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        pa[k] = 0.0;
    }
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            let q = lidx + (256u * k);
            let row = t + (q / 128u);
            let col = i0 + (q % 128u);
            var f0: f32 = 0.0;
            if (row < cfg.m) {
                if (col < cfg.d) {
                    f0 = phi[(row * cfg.d) + col];
                }
            }
            wg_a[q] = f0;
        }
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let q = lidx + (256u * k);
            let row = t + (q / 64u);
            let rl = i1 + (q % 64u);
            var f1: f32 = 0.0;
            if (row < cfg.m) {
                if (rl < cfg.r_rows) {
                    if ((cfg.r_off + rl) < cfg.d) {
                        f1 = phi[(row * cfg.d) + (cfg.r_off + rl)];
                    }
                }
            }
            wg_b[q] = f1;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let ia = (k * 128u) + lid.x;
            let ia2 = ia + 64u;
            let ib = (k * 64u) + lid.y;
            let a0 = wg_a[ia];
            let a1 = wg_a[ia + 16u];
            let a2 = wg_a[ia + 32u];
            let a3 = wg_a[ia + 48u];
            let a4 = wg_a[ia2];
            let a5 = wg_a[ia2 + 16u];
            let a6 = wg_a[ia2 + 32u];
            let a7 = wg_a[ia2 + 48u];
            let b0 = wg_b[ib];
            let b1 = wg_b[ib + 16u];
            let b2 = wg_b[ib + 32u];
            let b3 = wg_b[ib + 48u];
            pa[0u] = pa[0u] + (b0 * a0); pa[1u] = pa[1u] + (b0 * a1); pa[2u] = pa[2u] + (b0 * a2); pa[3u] = pa[3u] + (b0 * a3);
            pa[4u] = pa[4u] + (b0 * a4); pa[5u] = pa[5u] + (b0 * a5); pa[6u] = pa[6u] + (b0 * a6); pa[7u] = pa[7u] + (b0 * a7);
            pa[8u] = pa[8u] + (b1 * a0); pa[9u] = pa[9u] + (b1 * a1); pa[10u] = pa[10u] + (b1 * a2); pa[11u] = pa[11u] + (b1 * a3);
            pa[12u] = pa[12u] + (b1 * a4); pa[13u] = pa[13u] + (b1 * a5); pa[14u] = pa[14u] + (b1 * a6); pa[15u] = pa[15u] + (b1 * a7);
            pa[16u] = pa[16u] + (b2 * a0); pa[17u] = pa[17u] + (b2 * a1); pa[18u] = pa[18u] + (b2 * a2); pa[19u] = pa[19u] + (b2 * a3);
            pa[20u] = pa[20u] + (b2 * a4); pa[21u] = pa[21u] + (b2 * a5); pa[22u] = pa[22u] + (b2 * a6); pa[23u] = pa[23u] + (b2 * a7);
            pa[24u] = pa[24u] + (b3 * a0); pa[25u] = pa[25u] + (b3 * a1); pa[26u] = pa[26u] + (b3 * a2); pa[27u] = pa[27u] + (b3 * a3);
            pa[28u] = pa[28u] + (b3 * a4); pa[29u] = pa[29u] + (b3 * a5); pa[30u] = pa[30u] + (b3 * a6); pa[31u] = pa[31u] + (b3 * a7);
        }
        workgroupBarrier();
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let rr = i3 + (16u * k);
        for (var k2: u32 = 0u; k2 < 8u; k2 = k2 + 1u) {
            let cc = i2 + (16u * k2);
            if (rr < cfg.r_rows) {
                if ((cfg.r_off + rr) < cfg.d) {
                    if (cc < cfg.d) {
                        h[(rr * cfg.d) + cc] = h[(rr * cfg.d) + cc] + pa[(k * 8u) + k2];
                    }
                }
            }
        }
    }
}
