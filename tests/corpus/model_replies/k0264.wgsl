// grid 16 16 2 2
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var c00: f32 = 0.0;
    var c01: f32 = 0.0;
    var c10: f32 = 0.0;
    var c11: f32 = 0.0;
    var t: u32 = 0u;
    var k: u32 = 0u;
    let li = lid.y * 16u + lid.x;
    let kk = li >> 5u;
    let ii = li & 31u;
    let lrow = wid.y * 32u + ii;
    let gcol = wid.x * 32u + ii;
    for (t = 0u; t < cfg.m; t = t + 8u) {
        var f0: f32 = 0.0;
        var f1: f32 = 0.0;
        if (t + kk < cfg.m) {
            if (lrow < cfg.r_rows && cfg.r_off + lrow < cfg.d) {
                f0 = phi[(t + kk) * cfg.d + cfg.r_off + lrow];
            }
            if (gcol < cfg.d) {
                f1 = phi[(t + kk) * cfg.d + gcol];
            }
        }
        wg_a[li] = f0;
        wg_b[li] = f1;
        workgroupBarrier();
        for (k = 0u; k < 8u; k = k + 1u) {
            let a0 = wg_a[k * 32u + lid.y];
            let a1 = wg_a[k * 32u + lid.y + 16u];
            let b0 = wg_b[k * 32u + lid.x];
            let b1 = wg_b[k * 32u + lid.x + 16u];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
        }
        workgroupBarrier();
    }
    let r0 = wid.y * 32u + lid.y;
    let r1 = r0 + 16u;
    let q0 = wid.x * 32u + lid.x;
    let q1 = q0 + 16u;
    if (r0 < cfg.r_rows && cfg.r_off + r0 < cfg.d) {
        if (q0 < cfg.d) {
            h[r0 * cfg.d + q0] = h[r0 * cfg.d + q0] + c00;
        }
        if (q1 < cfg.d) {
            h[r0 * cfg.d + q1] = h[r0 * cfg.d + q1] + c01;
        }
    }
    if (r1 < cfg.r_rows && cfg.r_off + r1 < cfg.d) {
        if (q0 < cfg.d) {
            h[r1 * cfg.d + q0] = h[r1 * cfg.d + q0] + c10;
        }
        if (q1 < cfg.d) {
            h[r1 * cfg.d + q1] = h[r1 * cfg.d + q1] + c11;
        }
    }
}
