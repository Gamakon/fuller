// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> wg_a: array<f32, 1024>;
var<workgroup> wg_b: array<f32, 1024>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    let d = cfg.d;
    let m = cfg.m;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        // cooperative, coalesced tile load: 16 items x 64 columns for each operand
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let idx = lidx + k * 256u;
            let it = idx >> 6u;
            let col = idx & 63u;
            let item = t0 + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let base = item * d;
                let c = colBase + col;
                if (c < d) {
                    vb = phi[base + c];
                }
                let lrow = rowBase + col;
                let gr = cfg.r_off + lrow;
                if (lrow < cfg.r_rows && gr < d) {
                    va = phi[base + gr];
                }
            }
            wg_a[idx & 1023u] = va;
            wg_b[idx & 1023u] = vb;
        }
        workgroupBarrier();
        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let o = tt * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = wg_a[(o + lid.y + 16u * i) & 1023u];
                b[i & 3u] = wg_b[(o + lid.x + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = fma(a[i & 3u], b[j & 3u], acc[q]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowBase + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colBase + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
