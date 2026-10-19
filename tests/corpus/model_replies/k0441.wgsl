// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 2048>;
var<workgroup> wg_b: array<f32, 2048>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var pa: array<f32, 64>;
    var pb: array<f32, 64>;
    var ra: array<f32, 64>;
    var rb: array<f32, 64>;
    var ca: array<f32, 64>;
    var cb: array<f32, 64>;
    var t: u32 = 0u; var k: u32 = 0u; var j: u32 = 0u; var i: u32 = 0u;
    for (k = 0u; k < 16u; k = k + 1u) {
        pa[k] = 0.0f;
        pb[k] = 0.0f;
    }
    for (t = 0u; t < ((cfg.m + 31u) / 32u); t = t + 1u) {
        for (k = 0u; k < 8u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let item = t * 32u + idx / 64u;
            let cc = wid.x * 64u + (idx % 64u);
            let rr = wid.y * 64u + (idx % 64u);
            var va: f32 = 0.0f;
            if (item < cfg.m && cc < cfg.d) {
                va = phi[item * cfg.d + cc];
            }
            wg_a[idx & 2047u] = va;
            var vb: f32 = 0.0f;
            if (item < cfg.m && rr < cfg.r_rows && (cfg.r_off + rr) < cfg.d) {
                vb = phi[item * cfg.d + cfg.r_off + rr];
            }
            wg_b[idx & 2047u] = vb;
        }
        workgroupBarrier();
        for (k = 0u; k < 32u; k = k + 2u) {
            for (j = 0u; j < 4u; j = j + 1u) {
                ca[j] = wg_a[(k * 64u + lid.x + 16u * j) & 2047u];
                cb[j] = wg_a[((k + 1u) * 64u + lid.x + 16u * j) & 2047u];
                ra[j] = wg_b[(k * 64u + lid.y + 16u * j) & 2047u];
                rb[j] = wg_b[((k + 1u) * 64u + lid.y + 16u * j) & 2047u];
            }
            for (i = 0u; i < 4u; i = i + 1u) {
                for (j = 0u; j < 4u; j = j + 1u) {
                    pa[i * 4u + j] = pa[i * 4u + j] + ra[i] * ca[j];
                    pb[i * 4u + j] = pb[i * 4u + j] + rb[i] * cb[j];
                }
            }
        }
        workgroupBarrier();
    }
    for (i = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + lid.y + 16u * i;
        for (j = 0u; j < 4u; j = j + 1u) {
            let c = wid.x * 64u + lid.x + 16u * j;
            if (lr < cfg.r_rows && c < cfg.d && (cfg.r_off + lr) < cfg.d) {
                let o = lr * cfg.d + c;
                h[o] = h[o] + (pa[i * 4u + j] + pb[i * 4u + j]);
            }
        }
    }
}
