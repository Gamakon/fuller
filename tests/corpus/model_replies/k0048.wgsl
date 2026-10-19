// grid 16 16 4 4
struct Cfg { d: u32, m: u32, r_rows: u32, r_off: u32, };
@group(0) @binding(0) var<storage, read> phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

var<workgroup> sr: array<f32, 1024>;
var<workgroup> sc: array<f32, 1024>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = cfg.d;
    let m = cfg.m;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    let colbase = wid.x * 64u;
    let rowbase = wid.y * 64u;
    var acc: array<f32, 16>;
    for (var q: u32 = 0u; q < 16u; q = q + 1u) {
        acc[q & 15u] = 0.0;
    }
    for (var t0: u32 = 0u; t0 < m; t0 = t0 + 16u) {
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * 256u;
            let k = idx >> 6u;
            let j = idx & 63u;
            let t = t0 + k;
            let col = colbase + j;
            let lr = rowbase + j;
            let gr = cfg.r_off + lr;
            var vc: f32 = 0.0;
            var vr: f32 = 0.0;
            if (t < m) {
                let base = t * d;
                if (col < d) { vc = phi[base + col]; }
                if (lr < cfg.r_rows && gr < d) { vr = phi[base + gr]; }
            }
            sc[idx & 1023u] = vc;
            sr[idx & 1023u] = vr;
        }
        workgroupBarrier();
        for (var k: u32 = 0u; k < 16u; k = k + 1u) {
            let o = k * 64u;
            var a: array<f32, 4>;
            var b: array<f32, 4>;
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                a[i & 3u] = sr[(o + ty + 16u * i) & 1023u];
                b[i & 3u] = sc[(o + tx + 16u * i) & 1023u];
            }
            for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                    let q = (i * 4u + j) & 15u;
                    acc[q] = acc[q] + a[i & 3u] * b[j & 3u];
                }
            }
        }
        workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let lr = rowbase + ty + 16u * i;
        if (lr < cfg.r_rows) {
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                let c = colbase + tx + 16u * j;
                if (c < d) {
                    let idx = lr * d + c;
                    h[idx] = h[idx] + acc[(i * 4u + j) & 15u];
                }
            }
        }
    }
}
