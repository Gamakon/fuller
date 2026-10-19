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
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let col = tid & 63u;
    let item_sub = tid >> 6u;
    let cb = wid.x * 64u + col;
    let lrow = wid.y * 64u + col;
    let grow = cfg.r_off + lrow;
    let a_ok = (lrow < cfg.r_rows) && (grow < cfg.d);
    let b_ok = cb < cfg.d;
    for (var t: u32 = 0u; t < cfg.m; t = t + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let it = t + item_sub + 4u * k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (it < cfg.m) {
                let base = it * cfg.d;
                if (a_ok) { va = phi[base + grow]; }
                if (b_ok) { vb = phi[base + cb]; }
            }
            let si = (tid + 256u * k) & 1023u;
            wg_a[si] = va;
            wg_b[si] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let b0 = wg_b[(o + lx) & 1023u];
            let b1 = wg_b[(o + lx + 16u) & 1023u];
            let b2 = wg_b[(o + lx + 32u) & 1023u];
            let b3 = wg_b[(o + lx + 48u) & 1023u];
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let a = wg_a[(o + ly + 16u * i) & 1023u];
                let q = i * 4u;
                acc[q & 15u] = acc[q & 15u] + a * b0;
                acc[(q + 1u) & 15u] = acc[(q + 1u) & 15u] + a * b1;
                acc[(q + 2u) & 15u] = acc[(q + 2u) & 15u] + a * b2;
                acc[(q + 3u) & 15u] = acc[(q + 3u) & 15u] + a * b3;
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = wid.y * 64u + ly + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < cfg.d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = wid.x * 64u + lx + 16u * j;
                if (c < cfg.d) {
                    let idx = lr * cfg.d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
