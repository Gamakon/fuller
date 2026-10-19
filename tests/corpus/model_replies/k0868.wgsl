// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let ly = lid.y * 4u;
    let lx = lid.x * 4u;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + k * 256u;
            let item = t + (e >> 6u);
            let o = e & 63u;
            var a: f32 = 0.0;
            var b: f32 = 0.0;
            if (item < cfg.m) {
                let lrow = row0 + o;
                if (lrow < cfg.r_rows && (cfg.r_off + lrow) < cfg.d) {
                    a = phi[item * cfg.d + cfg.r_off + lrow];
                }
                if (col0 + o < cfg.d) {
                    b = phi[item * cfg.d + col0 + o];
                }
            }
            wg_a[e & 1023u] = a;
            wg_b[e & 1023u] = b;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let base = k * 64u;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let av = wg_a[(base + ly + i) & 1023u];
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let n = (i * 4u + j) & 15u;
                    acc[n] = acc[n] + av * wg_b[(base + lx + j) & 1023u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ly + i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lx + j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
