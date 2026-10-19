// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var t: u32 = 0u;
    var k: u32 = 0u;
    var k2: u32 = 0u;
    var k3: u32 = 0u;
    var f0: f32 = 0.0;
    var f1: f32 = 0.0;
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var b0: f32 = 0.0; var b1: f32 = 0.0; var b2: f32 = 0.0; var b3: f32 = 0.0;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k & 63u] = 0.0f;
    }
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let ty = lid.y * 4u;
    let tx = lid.x * 4u;
    for (t = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let e = lidx + k * 256u;
            let it = t * 32u + (e / 64u);
            let off = e % 64u;
            f0 = 0.0f;
            f1 = 0.0f;
            if (it < cfg.m) {
                let lr = row0 + off;
                if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
                    f0 = phi[it * cfg.d + cfg.r_off + lr];
                }
                let c = col0 + off;
                if (c < cfg.d) {
                    f1 = phi[it * cfg.d + c];
                }
            }
            wg_a[e & 2047u] = f0;
            wg_b[e & 2047u] = f1;
        }
        workgroupBarrier();
        for (k = 0u; k < 32u; k = k + 1u) {
            let ba = k * 64u + ty;
            let bb = k * 64u + tx;
            a0 = wg_a[ba & 2047u];
            a1 = wg_a[(ba + 1u) & 2047u];
            a2 = wg_a[(ba + 2u) & 2047u];
            a3 = wg_a[(ba + 3u) & 2047u];
            b0 = wg_b[bb & 2047u];
            b1 = wg_b[(bb + 1u) & 2047u];
            b2 = wg_b[(bb + 2u) & 2047u];
            b3 = wg_b[(bb + 3u) & 2047u];
            pa[0] = pa[0] + a0 * b0;
            pa[1] = pa[1] + a0 * b1;
            pa[2] = pa[2] + a0 * b2;
            pa[3] = pa[3] + a0 * b3;
            pa[4] = pa[4] + a1 * b0;
            pa[5] = pa[5] + a1 * b1;
            pa[6] = pa[6] + a1 * b2;
            pa[7] = pa[7] + a1 * b3;
            pa[8] = pa[8] + a2 * b0;
            pa[9] = pa[9] + a2 * b1;
            pa[10] = pa[10] + a2 * b2;
            pa[11] = pa[11] + a2 * b3;
            pa[12] = pa[12] + a3 * b0;
            pa[13] = pa[13] + a3 * b1;
            pa[14] = pa[14] + a3 * b2;
            pa[15] = pa[15] + a3 * b3;
        }
        workgroupBarrier();
    }
    for (k2 = 0u; k2 < 4u; k2 = k2 + 1u) {
        for (k3 = 0u; k3 < 4u; k3 = k3 + 1u) {
            let lr = gid.y * 4u + k2;
            let c = gid.x * 4u + k3;
            if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d && c < cfg.d) {
                let idx = lr * cfg.d + c;
                h[idx] = h[idx] + pa[(k2 * 4u + k3) & 63u];
            }
        }
    }
}
