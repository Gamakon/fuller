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
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    for (var t: u32 = 0u; t < m; t = t + 16u) {
        // cooperative, coalesced loads: 64 consecutive floats per item row
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = lidx + 256u * k;
            let tt = e >> 6u;
            let cc = e & 63u;
            let item = t + tt;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (item < m) {
                let base = item * d;
                let c = col0 + cc;
                if (c < d) {
                    va = phi[base + c];
                }
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    vb = phi[base + gr];
                }
            }
            wg_a[e & 1023u] = va;
            wg_b[e & 1023u] = vb;
        }
        workgroupBarrier();

        for (var tt: u32 = 0u; tt < 16u; tt = tt + 1u) {
            let ob = tt * 64u;
            let a0 = wg_a[(ob + lid.x) & 1023u];
            let a1 = wg_a[(ob + lid.x + 16u) & 1023u];
            let a2 = wg_a[(ob + lid.x + 32u) & 1023u];
            let a3 = wg_a[(ob + lid.x + 48u) & 1023u];
            let b0 = wg_b[(ob + lid.y) & 1023u];
            let b1 = wg_b[(ob + lid.y + 16u) & 1023u];
            let b2 = wg_b[(ob + lid.y + 32u) & 1023u];
            let b3 = wg_b[(ob + lid.y + 48u) & 1023u];
            acc[0u] = acc[0u] + b0 * a0;
            acc[1u] = acc[1u] + b0 * a1;
            acc[2u] = acc[2u] + b0 * a2;
            acc[3u] = acc[3u] + b0 * a3;
            acc[4u] = acc[4u] + b1 * a0;
            acc[5u] = acc[5u] + b1 * a1;
            acc[6u] = acc[6u] + b1 * a2;
            acc[7u] = acc[7u] + b1 * a3;
            acc[8u] = acc[8u] + b2 * a0;
            acc[9u] = acc[9u] + b2 * a1;
            acc[10u] = acc[10u] + b2 * a2;
            acc[11u] = acc[11u] + b2 * a3;
            acc[12u] = acc[12u] + b3 * a0;
            acc[13u] = acc[13u] + b3 * a1;
            acc[14u] = acc[14u] + b3 * a2;
            acc[15u] = acc[15u] + b3 * a3;
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + lid.y + 16u * i;
        if (lr < cfg.r_rows && (cfg.r_off + lr) < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lid.x + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
