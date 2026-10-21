// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sA: array<f32, 1024>;
var<workgroup> sB: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid3: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid3.x;
    let ty = lid3.y;
    let lid = ty * 16u + tx;
    let col0 = wid.x * 64u;
    let row0 = wid.y * 64u;

    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) { acc[q] = 0.0; }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var n: u32 = 0u; n < 4u; n = n + 1u) {
            let e = lid + n * 256u;
            let k = e >> 6u;
            let i = e & 63u;
            let t = t0 + k;
            var va: f32 = 0.0;
            var vb: f32 = 0.0;
            if (t < m) {
                let lr = row0 + i;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    va = phi[t * d + gr];
                }
                let c = col0 + i;
                if (c < d) {
                    vb = phi[t * d + c];
                }
            }
            sA[e] = va;
            sB[e] = vb;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let base = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                a[j] = sA[base + ty * 4u + j];
                b[j] = sB[base + tx * 4u + j];
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    acc[j * 4u + i] = acc[j * 4u + i] + a[j] * b[i];
                }
            }
        }
        workgroupBarrier();
    }

    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let lr = row0 + ty * 4u + j;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                let c = col0 + tx * 4u + i;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[j * 4u + i];
                }
            }
        }
    }
}
