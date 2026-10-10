// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sr: array<f32, 1024>;
var<workgroup> sc: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let col0 = wg.x * 64u;
    let row0 = wg.y * 64u;

    var acc: array<f32, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var q: u32 = 0u; q < 4u; q = q + 1u) {
            let idx = tid + 256u * q;
            let item = idx >> 6u;
            let pos = idx & 63u;
            let t = t0 + item;
            var rv: f32 = 0.0;
            var cv: f32 = 0.0;
            if (t < m) {
                let lrr = row0 + pos;
                let gr = cfg.r_off + lrr;
                if (lrr < cfg.r_rows && gr < d) {
                    rv = phi[t * d + gr];
                }
                let cc = col0 + pos;
                if (cc < d) {
                    cv = phi[t * d + cc];
                }
            }
            sr[idx] = rv;
            sc[idx] = cv;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let base = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i] = sr[base + ty + 16u * i];
                b[i] = sc[base + tx + 16u * i];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    acc[i * 4u + j] = acc[i * 4u + j] + a[i] * b[j];
                }
            }
        }
        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = row0 + ty + 16u * i;
        let gr = cfg.r_off + lr;
        if (lr < cfg.r_rows && gr < d) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = col0 + tx + 16u * j;
                if (c < d) {
                    let o = lr * d + c;
                    h[o] = h[o] + acc[i * 4u + j];
                }
            }
        }
    }
}
