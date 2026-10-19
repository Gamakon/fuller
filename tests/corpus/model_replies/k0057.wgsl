// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tid = lid.y * 16u + lid.x;
    let colBase = wid.x * 64u;
    let rowBase = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i & 15u] = 0.0;
    }
    for (var t0 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q = 0u; q < 4u; q = q + 1u) {
            let e = tid + q * 256u;
            let k = e >> 6u;
            let j = e & 63u;
            let t = t0 + k;
            var va = 0.0;
            var vb = 0.0;
            if (t < m) {
                let lr = rowBase + j;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = colBase + j;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sa[(k * 64u + lid.y * 4u + i) & 1023u];
                b[i & 3u] = sb[(k * 64u + lid.x * 4u + i) & 1023u];
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                for (var i = 0u; i < 4u; i = i + 1u) {
                    let ix = (j * 4u + i) & 15u;
                    acc[ix] = acc[ix] + a[j & 3u] * b[i & 3u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
        let lr = rowBase + lid.y * 4u + j;
        if (lr < cfg.r_rows && cfg.r_off + lr < d) {
            for (var i = 0u; i < 4u; i = i + 1u) {
                let c = colBase + lid.x * 4u + i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
