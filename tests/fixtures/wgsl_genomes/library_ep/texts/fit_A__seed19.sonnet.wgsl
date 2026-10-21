// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> As: array<f32, 512>;
var<workgroup> Bs: array<f32, 512>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let lx = lid.x;
    let ly = lid.y;
    let tid = ly * 16u + lx;
    let row0 = wg.y * 64u;   // local row base
    let col0 = wg.x * 64u;   // column base

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var t0 = 0u; t0 < m; t0 = t0 + 8u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let k = tid + 256u * s;
            let kk = k >> 6u;
            let cc = k & 63u;
            let t = t0 + kk;
            var av = 0.0;
            var bv = 0.0;
            if (t < m) {
                let lr = row0 + cc;
                let gr = cfg.r_off + lr;
                if (lr < cfg.r_rows && gr < d) {
                    av = phi[t * d + gr];
                }
                let c = col0 + cc;
                if (c < d) {
                    bv = phi[t * d + c];
                }
            }
            As[k] = av;
            Bs[k] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 8u; kk = kk + 1u) {
            let base = kk * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                a[i] = As[base + ly + 16u * i];
                b[i] = Bs[base + lx + 16u * i];
            }
            for (var i = 0u; i < 4u; i = i + 1u) {
                for (var j = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ly + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j = 0u; j < 4u; j = j + 1u) {
                let c = col0 + lx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[i * 4u + j];
                }
            }
        }
    }
}
