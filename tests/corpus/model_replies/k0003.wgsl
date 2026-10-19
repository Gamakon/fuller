// grid 16 16 4 4
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 1024>;

struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let d = cfg.d;
    let m = cfg.m;
    let colbase = wg.x * 64u;
    let rowbase = wg.y * 64u;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let e = tid + 256u * k;
            let it = e >> 6u;
            let cc = e & 63u;
            let t = t0 + it;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let c = colbase + cc;
                if (c < d) {
                    vb = phi[t * d + c];
                }
                let lr = rowbase + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
            }
            sa[e & 1023u] = va;
            sb[e & 1023u] = vb;
        }
        workgroupBarrier();
        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                a[j & 3u] = sa[(o + ty + 16u * j) & 1023u];
                b[j & 3u] = sb[(o + tx + 16u * j) & 1023u];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    let q = (j * 4u + i) & 15u;
                    acc[q] = acc[q] + a[j & 3u] * b[i & 3u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = rowbase + ty + 16u * j;
        if (lr < cfg.r_rows) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = colbase + tx + 16u * i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(j * 4u + i) & 15u];
                }
            }
        }
    }
}
